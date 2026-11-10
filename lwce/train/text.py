"""Training loop, evaluation and proxy search objective for the text domain (GoEmotions).

AdamW (lr 5e-4, weight decay 0.01), linear warm-up for WARMUP_EPOCHS then cosine decay
(stepped once per epoch), gradient clipping at norm 1.0. The checkpoint with the best
validation macro-F1 is kept. Gradient statistics are logged inside the loop (GradLogger)
because they cannot be recovered after training.
"""

import math

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from lwce.data.goemotions import BATCH_SIZE, MAX_LEN
from lwce.losses import get_clf_loss
from lwce.metrics import GradLogger, extended_metrics
from lwce.models.transformer import build_text_transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FINAL_EPOCHS = 50
WARMUP_EPOCHS = 3
PROXY_EPOCHS = 10
PROXY_SUBSET_RATIO = 0.20
PROXY_MIN_PER_CLASS = 5


def predict(model, loader, device=DEVICE):
    """(y_true, y_pred) arrays of argmax predictions."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yp.extend(model(xb.to(device)).argmax(dim=1).cpu().tolist())
            yt.extend(yb.tolist())
    return np.array(yt), np.array(yp)


def compute_val_metrics(model, loader, num_classes, class_counts_train=None, device=DEVICE):
    """Top1_Acc, F1_Macro, extended_metrics and, given the training counts, Many/Medium/Few
    accuracy (mean recall over the frequency tertiles of the classes)."""
    y_true, y_pred = predict(model, loader, device)
    result = {
        'Top1_Acc': float((y_true == y_pred).mean()),
        'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    result.update(extended_metrics(y_true, y_pred, num_classes))

    if class_counts_train is not None:
        pca = np.array(result['Per_Class_Acc'])
        sorted_idx = np.argsort(np.array(class_counts_train))[::-1]   # many -> few
        n = len(sorted_idx)
        result['Many_Acc']   = float(pca[sorted_idx[:n // 3]].mean())
        result['Medium_Acc'] = float(pca[sorted_idx[n // 3:2 * n // 3]].mean())
        result['Few_Acc']    = float(pca[sorted_idx[2 * n // 3:]].mean())
    return result


def compute_val_acc(model, loader, device=DEVICE):
    """Macro-F1: the model-selection and search objective (not balanced accuracy)."""
    y_true, y_pred = predict(model, loader, device)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def train_model(loss_name, class_counts, train_loader, val_loader, num_classes, vocab_size,
                alpha=1.0, gamma=2.0, eps=0.1, tau=1.0,
                epochs=FINAL_EPOCHS, lr=5e-4, seed=42, device=DEVICE):
    """Train a TextTransformer and return (best-val-F1 model, history, best val macro-F1)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_text_transformer(vocab_size, num_classes, pad_idx=0, max_len=MAX_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # linear warm-up (WARMUP_EPOCHS) -> cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = get_clf_loss(loss_name, class_counts, alpha=alpha, gamma=gamma,
                             eps=eps, tau=tau)
    glog = GradLogger(class_counts, num_classes, device)

    best_val_f1, best_state = 0.0, None
    # 'val_balanced_acc' holds validation macro-F1; the legacy key name is kept so that
    # the saved histories match the paper-time result files.
    history = {'epoch': [], 'train_loss': [], 'val_balanced_acc': [],
               'grad_norm': [], 'grad_many': [], 'grad_few': [], 'grad_ratio': []}

    for epoch in range(epochs):
        model.train()
        glog.reset()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            logits.retain_grad()          # per-sample logit gradients for GradLogger
            loss = criterion(logits, yb)
            loss.backward()
            # clip_grad_norm_ returns the total norm before clipping; log it as is
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            glog.update(logits, yb, grad_norm=total_norm)
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
        train_loss /= len(train_loader.dataset)

        val_f1 = compute_val_acc(model, val_loader, device)        # macro-F1
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_balanced_acc'].append(val_f1)
        for k, v in glog.epoch_end().items():
            history[k].append(v)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_val_f1


def proxy_positions(labels, ratio=PROXY_SUBSET_RATIO, min_per_class=PROXY_MIN_PER_CLASS,
                    seed=42):
    """Stratified proxy subset of the training set: per class
    max(round(ratio * n_c), min(n_c, min_per_class)) samples, so tail classes survive."""
    rng = np.random.RandomState(seed)
    proxy_pos = []
    for c in np.unique(labels):
        pos_c = np.where(labels == c)[0]
        n_take = max(int(round(ratio * len(pos_c))), min(len(pos_c), min_per_class))
        n_take = min(n_take, len(pos_c))
        proxy_pos.extend(rng.choice(pos_c, size=n_take, replace=False).tolist())
    rng.shuffle(proxy_pos)
    return proxy_pos


def make_proxy_loader(train_loader, batch_size=BATCH_SIZE):
    """Shuffled loader over the proxy subset of train_loader's TensorDataset."""
    pos = proxy_positions(train_loader.dataset.tensors[1].numpy())
    return DataLoader(Subset(train_loader.dataset, pos), batch_size=batch_size,
                      shuffle=True, num_workers=0)


def proxy_score(loss_name, class_counts, proxy_loader, val_loader, num_classes, vocab_size,
                epochs=PROXY_EPOCHS, device=DEVICE, **params):
    """Search objective: validation macro-F1 after training on the proxy subset (seed 42)."""
    model, _, _ = train_model(loss_name, class_counts, proxy_loader, val_loader, num_classes,
                              vocab_size, epochs=epochs, device=device, **params)
    return compute_val_acc(model, val_loader, device)
