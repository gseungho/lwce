"""Text domain (GoEmotions, compact Transformer): grid search, final runs and aggregation.

1. Grid search for each tuned loss (lwce.search.GRIDS): every grid point is trained for
   --proxy-epochs on a stratified 20% proxy subset of the training split and scored by
   macro-F1 on the full validation split. The test split is never used for selection.
2. Final runs: every loss x seed is trained for --epochs on the full training split with
   the selected parameters (defaults for untuned losses) and evaluated on the test split.
3. Aggregation over seeds: mean and population standard deviation (np.std, ddof=0).

Outputs in --out-dir, with the names and structures of the paper-time result files:
    optuna_trials.json      {loss: [{'params': {...}, 'value': val macro-F1}, ...]}
    optuna_checkpoint.json  {loss: selected params}
    results_checkpoint.json {'{loss}_s{seed}': {'metrics': {...}, 'history': {...}}}
    results_per_seed.json   {loss: {seed: metrics}}
    results_agg.json        {loss: {'<metric>_mean': ..., '<metric>_std': ...}}
Searches and runs already present in these files are skipped, so an interrupted run resumes.

    python scripts/run_text.py
"""

import argparse
import functools
import json
import os

import numpy as np
import torch

from lwce.data.goemotions import NUM_CLASSES, load_goemotions
from lwce.losses import get_clf_loss
from lwce.runs import check_run_config, runs_in_checkpoint
from lwce.search import DEFAULT_PARAMS, GRIDS, LOSSES, run_grid_search
from lwce.train.text import (
    DEVICE,
    FINAL_EPOCHS,
    PROXY_EPOCHS,
    compute_val_metrics,
    make_proxy_loader,
    proxy_score,
    train_model,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 43, 44, 45, 46]
METRIC_KEYS = ['Top1_Acc', 'Balanced_Acc', 'F1_Macro', 'G_Mean', 'Worst_Acc',
               'Many_Acc', 'Medium_Acc', 'Few_Acc']


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _dump_json(obj, path, indent=None):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def search(losses, class_counts, train_loader, val_loader, num_classes, vocab_size, out_dir,
           proxy_epochs=PROXY_EPOCHS, device=DEVICE):
    """Grid search for the tuned losses that are not yet in optuna_checkpoint.json.

    Returns:
        {loss: selected params} for every searched loss found or run.
    """
    best_path = os.path.join(out_dir, 'optuna_checkpoint.json')
    trials_path = os.path.join(out_dir, 'optuna_trials.json')
    best, trials = _load_json(best_path), _load_json(trials_path)

    pending = [k for k in GRIDS if k in losses and k not in best]
    if not pending:
        return best
    proxy_loader = make_proxy_loader(train_loader)
    print(f'search: {pending} | proxy {len(proxy_loader.dataset)} samples')

    for loss in pending:
        evaluate = functools.partial(proxy_score, loss, class_counts, proxy_loader, val_loader,
                                     num_classes, vocab_size, epochs=proxy_epochs, device=device)
        # grid points with exactly the same params as a saved trial are reused
        best[loss], trials[loss], _ = run_grid_search(
            GRIDS[loss], evaluate, cached_trials=trials.get(loss), label=loss)
        _dump_json(best, best_path, indent=2)
        _dump_json(trials, trials_path, indent=2)
    return best


def final_runs(losses, seeds, best, class_counts, train_loader, val_loader, test_loader,
               num_classes, vocab_size, out_dir, epochs=FINAL_EPOCHS, device=DEVICE):
    """Train every loss x seed not yet in results_checkpoint.json; return the checkpoint."""
    path = os.path.join(out_dir, 'results_checkpoint.json')
    ckpt = _load_json(path)

    for loss in losses:
        tuned = best.get(loss, {})
        params = {k: tuned.get(k, v) for k, v in DEFAULT_PARAMS.items()}
        for seed in seeds:
            run_key = f'{loss}_s{seed}'
            if run_key in ckpt:
                continue
            print(f'[{loss}] seed={seed} ' + ', '.join(f'{k}={v:.2f}' for k, v in params.items()))
            model, history, _ = train_model(
                loss, class_counts, train_loader, val_loader, num_classes, vocab_size,
                **params, epochs=epochs, seed=seed, device=device)

            metrics = compute_val_metrics(model, test_loader, num_classes,
                                          class_counts_train=class_counts, device=device)
            metrics.update(params)
            # normalised class weights actually used (None for unweighted losses)
            w = get_clf_loss(loss, class_counts, **params).get_weights()
            metrics['class_weights'] = None if w is None else w.tolist()

            ckpt[run_key] = {'metrics': metrics, 'history': history}
            _dump_json(ckpt, path)
            print(f'  F1={metrics["F1_Macro"]:.4f} | G-Mean={metrics["G_Mean"]:.4f} | '
                  f'Few={metrics["Few_Acc"]:.4f} | grad_ratio={history["grad_ratio"][-1]:.2f}')
    return ckpt


def aggregate(ckpt, losses, seeds, out_dir):
    """Write results_agg.json and results_per_seed.json; return (agg, per_seed)."""
    results = {loss: {s: ckpt[f'{loss}_s{s}']['metrics'] for s in seeds
                      if f'{loss}_s{s}' in ckpt} for loss in losses}
    agg = {}
    for loss in losses:
        seed_metrics = [results[loss][s] for s in seeds if s in results[loss]]
        if not seed_metrics:
            print(f'{loss:10s} | not finished')
            continue
        a = {}
        for key in METRIC_KEYS:
            vals = [m.get(key, 0.0) for m in seed_metrics]
            a[f'{key}_mean'] = float(np.mean(vals))
            a[f'{key}_std'] = float(np.std(vals))
        agg[loss] = a
        print(f'{loss:10s} | F1 {a["F1_Macro_mean"]:.4f}+-{a["F1_Macro_std"]:.4f} | '
              f'Top1 {a["Top1_Acc_mean"]:.4f} | BA {a["Balanced_Acc_mean"]:.4f} | '
              f'G-Mean {a["G_Mean_mean"]:.4f} | Few {a["Few_Acc_mean"]:.4f} | '
              f'n={len(seed_metrics)}')

    per_seed = {loss: {str(s): results[loss].get(s, {}) for s in seeds} for loss in losses}
    _dump_json(agg, os.path.join(out_dir, 'results_agg.json'), indent=2)
    _dump_json(per_seed, os.path.join(out_dir, 'results_per_seed.json'), indent=2)
    return agg, per_seed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--losses', nargs='+', default=list(LOSSES), choices=LOSSES)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=FINAL_EPOCHS)
    ap.add_argument('--proxy-epochs', type=int, default=PROXY_EPOCHS)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'results', 'text'))
    ap.add_argument('--cache-dir', default=None, help='Hugging Face datasets cache')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    check_run_config(args.out_dir, {'epochs': args.epochs, 'proxy_epochs': args.proxy_epochs})

    train_loader, val_loader, test_loader, class_counts, _, itos, _ = load_goemotions(
        cache_dir=args.cache_dir)
    vocab_size = len(itos)
    nz = [c for c in class_counts if c > 0]
    print(f'device {device} | vocab {vocab_size} | train/val/test '
          f'{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} | '
          f'IR {max(class_counts) / max(1, min(nz)):.1f}')

    best = search(args.losses, class_counts, train_loader, val_loader, NUM_CLASSES, vocab_size,
                  args.out_dir, proxy_epochs=args.proxy_epochs, device=device)
    ckpt = final_runs(args.losses, args.seeds, best, class_counts, train_loader, val_loader,
                      test_loader, NUM_CLASSES, vocab_size, args.out_dir, epochs=args.epochs,
                      device=device)
    # Aggregate every run in the folder, so a partial rerun never shrinks the files.
    losses = list(args.losses) + [ln for ln in LOSSES if ln not in args.losses]
    aggregate(ckpt, *runs_in_checkpoint(ckpt, losses, args.seeds), args.out_dir)


if __name__ == '__main__':
    main()
