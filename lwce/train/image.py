"""Training and evaluation of ResNet-32 on CIFAR-LT.

SGD (lr 0.1, momentum 0.9, weight decay 2e-4), MultiStepLR with milestones [160, 180]
and factor 0.01, and the checkpoint with the best validation macro-F1 is kept. LDAM
alone gets the cosine head and a 5-epoch linear learning-rate warm-up.
"""

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Subset

from lwce.losses import get_clf_loss
from lwce.metrics import GradLogger, extended_metrics
from lwce.models.resnet32 import build_resnet32

PROXY_SUBSET_RATIO = 0.20
PROXY_MIN_PER_CLASS = 5


def predict(model, loader, device):
    """Labels and argmax predictions over a loader."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            yp.extend(model(imgs.to(device)).argmax(dim=1).cpu().tolist())
            yt.extend(labels.tolist())
    return np.array(yt), np.array(yp)


def group_indices(class_counts):
    """Many, medium and few classes: tertiles of the training counts, most frequent first."""
    sorted_idx = np.argsort(np.array(class_counts))[::-1]
    n = len(sorted_idx)
    return sorted_idx[:n // 3], sorted_idx[n // 3:2 * n // 3], sorted_idx[2 * n // 3:]


def evaluate_metrics(model, loader, num_classes, class_counts_train, device):
    """Top-1, macro-F1, per-class recall, G-mean, worst and balanced accuracy, group recalls."""
    y_true, y_pred = predict(model, loader, device)
    result = {
        'Top1_Acc': float((y_true == y_pred).mean()),
        'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    result.update(extended_metrics(y_true, y_pred, num_classes))

    pca = np.array(result['Per_Class_Acc'])
    many_idx, medium_idx, few_idx = group_indices(class_counts_train)
    result['Many_Acc'] = float(pca[many_idx].mean())
    result['Medium_Acc'] = float(pca[medium_idx].mean())
    result['Few_Acc'] = float(pca[few_idx].mean())   # minority-class recall
    return result


def val_macro_f1(model, loader, device):
    """Model-selection and grid-search score: macro-F1."""
    y_true, y_pred = predict(model, loader, device)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def train_model(loss_name, class_counts, train_loader, val_loader, num_classes, device,
                alpha=1.0, gamma=2.0, eps=0.1, tau=1.0, epochs=200, lr=0.1, seed=42):
    """Train ResNet-32 and restore the epoch with the best validation macro-F1.

    Returns:
        (model, history, best_val_f1). history['val_balanced_acc'] holds the validation
        macro-F1; the key name is kept for compatibility with the saved results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_resnet32(num_classes, normed_head=(loss_name == 'ldam')).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    # Fixed milestones, also for the 20-epoch proxy runs (kept for reproducibility).
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 180], gamma=0.01)

    criterion = get_clf_loss(loss_name, class_counts, alpha=alpha, gamma=gamma,
                             eps=eps, tau=tau)
    glog = GradLogger(class_counts, num_classes, device)

    best_val_acc = 0.0
    best_model_state = None
    history = {'epoch': [], 'train_loss': [], 'val_balanced_acc': [],
               'grad_norm': [], 'grad_many': [], 'grad_few': [], 'grad_ratio': []}

    for epoch in range(epochs):
        model.train()
        # LDAM (s = 30) diverges under SGD without warm-up: 5-epoch linear warm-up, LDAM only.
        if loss_name == 'ldam' and epoch < 5:
            for g in optimizer.param_groups:
                g['lr'] = lr * (epoch + 1) / 5
        glog.reset()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            logits.retain_grad()                 # per-sample logit gradients for GradLogger
            loss = criterion(logits, labels)
            loss.backward()
            glog.update(logits, labels, model)   # before optimizer.step()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)
        val_f1 = val_macro_f1(model, val_loader, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_balanced_acc'].append(val_f1)
        for k, v in glog.epoch_end().items():
            history[k].append(v)

        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    return model, history, best_val_acc


def proxy_subset(train_subset, seed=42):
    """Stratified 20% of the training split, at least 5 images per class (or all of them).

    `train_subset` is the training loader's Subset; the result indexes into it.
    """
    all_targets = np.array(train_subset.dataset.targets)
    sub_labels = all_targets[np.array(train_subset.indices)]
    rng = np.random.RandomState(seed)
    proxy_pos = []
    for c in np.unique(sub_labels):
        pos_c = np.where(sub_labels == c)[0]
        n_take = max(int(round(PROXY_SUBSET_RATIO * len(pos_c))),
                     min(len(pos_c), PROXY_MIN_PER_CLASS))
        n_take = min(n_take, len(pos_c))
        proxy_pos.extend(rng.choice(pos_c, size=n_take, replace=False).tolist())
    rng.shuffle(proxy_pos)
    return Subset(train_subset, proxy_pos)


def proxy_evaluate(loss_name, class_counts, proxy_loader, val_loader, num_classes, epochs,
                   device, **params):
    """Grid-search objective: validation macro-F1 of the best proxy-run checkpoint (seed 42)."""
    model, _, _ = train_model(loss_name, class_counts, proxy_loader, val_loader, num_classes,
                              device, epochs=epochs, **params)
    return val_macro_f1(model, val_loader, device)
