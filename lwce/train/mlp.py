"""MLP training, proxy hyperparameter search and the experiment loop (tabular, network).

Paper settings: Adam (lr 1e-3, weight decay 1e-4) with cosine annealing to 1e-5 stepped
once per epoch, batch size 512, no gradient clipping, no early stopping; the model after
the last epoch is evaluated on the test set. Final runs: 100 epochs (tabular) or 50
(network), seeds 42-46. Proxy search: 20 / 10 epochs on a stratified 20% subset of the
training set, scored by macro-F1 on its held-out 20%.
"""

import gc
import json
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from lwce.losses import get_clf_loss
from lwce.metrics import GradLogger, extended_metrics
from lwce.models.mlp import MLP
from lwce.search import DEFAULT_PARAMS, GRIDS, run_grid_search

BATCH_SIZE = 512
SEEDS = [42, 43, 44, 45, 46]
PROXY_RATIO = 0.20


def compute_metrics(y_true, y_pred, class_counts):
    """Accuracy, balanced accuracy, macro/weighted F1, per-class F1, Head/Mid/Tail F1,
    plus per-class recall, G-mean and worst-class recall (lwce.metrics.extended_metrics).

    Head/Mid/Tail group the classes by training count: for two classes Tail is the rarer
    class, Head the other and Mid the macro-F1; otherwise Tail is the n // 3 rarest.
    """
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    # No labels= argument: covers only the classes present in y_true or y_pred
    # (kept for reproducibility).
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    n = len(class_counts)
    idx = np.argsort(class_counts)   # minority -> majority
    f1a = np.array(f1_pc)

    if n == 2:
        tail_f1 = float(f1a[idx[0]])
        head_f1 = float(f1a[idx[1]])
        mid_f1 = float(f1m)
    else:
        t = max(1, n // 3)
        tail_f1 = float(f1a[idx[:t]].mean())
        mid_f1 = float(f1a[idx[t:2 * t]].mean()) if 2 * t <= n else tail_f1
        head_f1 = float(f1a[idx[2 * t:]].mean()) if 2 * t < n else float(f1a[idx[-t:]].mean())

    out = dict(Accuracy=acc, Balanced_Accuracy=bacc,
               F1_Macro=f1m, F1_Weighted=f1w,
               Per_Class_F1=f1_pc,
               Head_F1=head_f1, Mid_F1=mid_f1, Tail_F1=tail_f1)
    out.update(extended_metrics(y_true, y_pred, n))
    return out


def train_model(X_tr, y_tr, X_te, y_te, class_counts, num_classes, loss_name,
                alpha=1.0, gamma=2.0, eps=0.1, tau=1.0, epochs=100, seed=42,
                test_batch_size=2048, device='cpu'):
    """Train one MLP and evaluate the last-epoch model on the test set.

    test_batch_size was 2048 (tabular) and 4096 (network).

    Returns:
        (metrics, history); history holds per-epoch train_loss and the GradLogger
        statistics grad_norm, grad_many, grad_few, grad_ratio.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ld = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True, num_workers=0)
    te_ld = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=test_batch_size, shuffle=False, num_workers=0)

    model = MLP(X_tr.shape[1], num_classes).to(device)
    criterion = get_clf_loss(loss_name, class_counts, alpha=alpha, gamma=gamma,
                             eps=eps, tau=tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    glog = GradLogger(class_counts, num_classes, device)
    history = {'epoch': [], 'train_loss': [],
               'grad_norm': [], 'grad_many': [], 'grad_few': [], 'grad_ratio': []}

    for ep in range(epochs):
        model.train()
        glog.reset()
        run_loss, n_seen = 0.0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            logits.retain_grad()               # per-sample logit gradients for GradLogger
            loss = criterion(logits, yb)
            loss.backward()
            glog.update(logits, yb, model)     # before optimizer.step()
            optimizer.step()
            run_loss += float(loss) * yb.size(0)
            n_seen += yb.size(0)
        scheduler.step()

        history['epoch'].append(ep)
        history['train_loss'].append(run_loss / max(1, n_seen))
        for k, v in glog.epoch_end().items():
            history[k].append(v)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_ld:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = compute_metrics(y_te, y_pred, class_counts)
    del model, criterion, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, history


def proxy_split(y_tr, ratio=PROXY_RATIO):
    """Indices (proxy-train, proxy-val) into the training set.

    A stratified `ratio` subset of the training set, split 80/20 (stratified). When a
    stratified split is impossible the subset is drawn at random and the 80/20 split is
    taken in order.
    """
    n_proxy = int(ratio * len(y_tr))
    try:
        proxy_idx, _ = train_test_split(
            np.arange(len(y_tr)), train_size=n_proxy, stratify=y_tr, random_state=42)
    except ValueError:
        proxy_idx = np.random.RandomState(42).choice(len(y_tr), n_proxy, replace=False)
    try:
        ptr_idx, pval_idx = train_test_split(
            proxy_idx, test_size=0.2, stratify=y_tr[proxy_idx], random_state=42)
    except ValueError:
        split = int(0.8 * len(proxy_idx))
        ptr_idx, pval_idx = proxy_idx[:split], proxy_idx[split:]
    return ptr_idx, pval_idx


def proxy_score(X_ptr, y_ptr, X_pval, y_pval, class_counts, num_classes, loss_name,
                epochs, device='cpu', **params):
    """Proxy-validation macro-F1 of one (loss, params) setting.

    Fixed seed 42 and the same Adam optimiser as train_model but no learning-rate
    schedule; class_counts are those of the full training set.
    """
    torch.manual_seed(42)
    model = MLP(X_ptr.shape[1], num_classes).to(device)
    crit = get_clf_loss(loss_name, class_counts, **params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(
        TensorDataset(torch.from_numpy(X_ptr), torch.from_numpy(y_ptr)),
        batch_size=min(BATCH_SIZE, len(X_ptr)), shuffle=True, num_workers=0)
    model.train()
    for _ in range(epochs):
        for xb, yb in ld:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_pval).to(device)).argmax(1).cpu().numpy()
    score = f1_score(y_pval, preds, average='macro', zero_division=0)
    del model, crit, opt
    gc.collect()
    torch.cuda.empty_cache()
    return score


def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _dump(obj, path, indent=2):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def _search(X_tr, y_tr, class_counts, num_classes, pending, best, trials, proxy_epochs,
            device):
    """Grid search of the `pending` losses on one proxy split; returns updated copies."""
    ptr_idx, pval_idx = proxy_split(y_tr)
    X_ptr, y_ptr = X_tr[ptr_idx], y_tr[ptr_idx]
    X_pval, y_pval = X_tr[pval_idx], y_tr[pval_idx]
    bp, ts = dict(best), dict(trials)
    for loss_name in pending:
        evaluate = partial(proxy_score, X_ptr, y_ptr, X_pval, y_pval, class_counts,
                           num_classes, loss_name, proxy_epochs, device)
        bp[loss_name], ts[loss_name], _ = run_grid_search(
            GRIDS[loss_name], evaluate, cached_trials=ts.get(loss_name), label=loss_name)
    return bp, ts


def _all_rows(out_dir, datasets):
    """Rows of every {dataset}_results.json in `out_dir`, in the order of `datasets`."""
    rows = []
    for ds in datasets:
        rows.extend(_load(os.path.join(out_dir, f'{ds}_results.json'), []))
    return rows


def run_domain(datasets, load_fn, out_dir, all_results_name, losses, seeds, epochs,
               proxy_epochs, test_batch_size, device, all_datasets=None):
    """Per dataset: load -> grid search of the tuned losses -> final runs -> save.

    Files in `out_dir` (names and layouts of the original results):
        _best_params.json     {dataset: {loss: {param: value}}}
        _optuna_trials.json   {dataset: {loss: [{'params': {...}, 'value': f1}, ...]}}
        {dataset}_results.json   one row per (loss, seed)
        _histories.json       {dataset: {'{loss}_s{seed}': history}}
        _class_counts.json    {dataset: training class counts}
        `all_results_name`    rows of every {dataset}_results.json in `out_dir`, in the
                              order of `all_datasets`; rewritten after each dataset (the
                              original also wrote an identical _checkpoint.json)

    Resuming: (loss, seed) runs already in {dataset}_results.json and losses already in
    _best_params.json are skipped; a dataset with nothing left to do is not loaded.

    Args:
        load_fn: callable(dataset) -> (X_tr, X_te, y_tr, y_te, class_counts, num_classes).
        all_datasets: order of the datasets in the all-results file, which also keeps the
            datasets of earlier calls (default: `datasets`).
    """
    order = all_datasets if all_datasets is not None else datasets
    os.makedirs(out_dir, exist_ok=True)
    bp_path = os.path.join(out_dir, '_best_params.json')
    trials_path = os.path.join(out_dir, '_optuna_trials.json')
    hist_path = os.path.join(out_dir, '_histories.json')
    counts_path = os.path.join(out_dir, '_class_counts.json')
    all_path = os.path.join(out_dir, all_results_name)

    best_params = _load(bp_path, {})
    optuna_trials = _load(trials_path, {})
    all_histories = _load(hist_path, {})
    all_counts = _load(counts_path, {})
    all_results = _all_rows(out_dir, order)

    for ds in datasets:
        ds_json = os.path.join(out_dir, f'{ds}_results.json')
        ds_rows = _load(ds_json, [])
        done = {(r['Loss'], r['Seed']) for r in ds_rows}
        pending = [ln for ln in GRIDS if ln in losses and ln not in best_params.get(ds, {})]
        if not pending and all((loss, s) in done for loss in losses for s in seeds):
            print(f'[{ds}] done -- skip ({len(ds_rows)} runs)')
            all_results = _all_rows(out_dir, order)
            _dump(all_results, all_path)
            continue

        print(f'\n[{ds}] loading...')
        X_tr, X_te, y_tr, y_te, class_counts, num_classes = load_fn(ds)
        if pending:
            print(f'[{ds}] grid search (pending: {pending})')
            best_params[ds], optuna_trials[ds] = _search(
                X_tr, y_tr, class_counts, num_classes, pending, best_params.get(ds, {}),
                optuna_trials.get(ds, {}), proxy_epochs, device)
            _dump(best_params, bp_path)
            _dump(optuna_trials, trials_path)
        bp = best_params.get(ds, {})

        all_histories.setdefault(ds, {})
        all_counts[ds] = list(map(int, class_counts))
        _dump(all_counts, counts_path)

        for loss in losses:
            # Tuned parameters of this loss, defaults for the rest (stored in every row).
            p = bp.get(loss, {})
            kw = {k: p.get(k, v) for k, v in DEFAULT_PARAMS.items()}
            new_run = False
            for seed in seeds:
                if (loss, seed) in done:
                    continue
                m, hist = train_model(X_tr, y_tr, X_te, y_te, class_counts, num_classes,
                                      loss_name=loss, epochs=epochs, seed=seed,
                                      test_batch_size=test_batch_size, device=device, **kw)
                ds_rows.append(dict(
                    Dataset=ds, Loss=loss, Seed=seed,
                    Accuracy=m['Accuracy'], Balanced_Accuracy=m['Balanced_Accuracy'],
                    F1_Macro=m['F1_Macro'], F1_Weighted=m['F1_Weighted'],
                    Head_F1=m['Head_F1'], Mid_F1=m['Mid_F1'], Tail_F1=m['Tail_F1'],
                    G_Mean=m['G_Mean'], Worst_Acc=m['Worst_Acc'],
                    Per_Class_Acc=m['Per_Class_Acc'], Per_Class_F1=m['Per_Class_F1'],
                    **kw))
                all_histories[ds][f'{loss}_s{seed}'] = hist
                new_run = True

            rows = [r for r in ds_rows if r['Loss'] == loss]
            f1s = [r['F1_Macro'] for r in rows]
            tail = np.mean([r['Tail_F1'] for r in rows])
            gm = np.mean([r.get('G_Mean', 0) for r in rows])
            print(f'{ds:22s}  {loss:9s}  F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  '
                  f'Tail={tail:.4f}  G-Mean={gm:.4f}{"" if new_run else "  (cached)"}')
            if new_run:   # save after every (dataset, loss)
                _dump(ds_rows, ds_json)
                _dump(all_histories, hist_path, indent=None)

        all_results = _all_rows(out_dir, order)
        _dump(all_results, all_path)
        del X_tr, X_te, y_tr, y_te
        gc.collect()
        torch.cuda.empty_cache()

    print('\nAll experiments complete.')
    return all_results
