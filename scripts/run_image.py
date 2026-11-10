"""CIFAR-10-LT / CIFAR-100-LT with ResNet-32: grid search, final runs and aggregation.

For each imbalance ratio IR:
  1. grid search of the tuned losses on a proxy task (20 epochs on a stratified 20% of
     the training split, seed 42, score = validation macro-F1);
  2. final runs: 200 epochs per (loss, seed), best-validation-macro-F1 checkpoint,
     evaluated on the balanced test set;
  3. mean and standard deviation over seeds.

Every finished run is saved at once, and runs already in the checkpoint files are skipped,
so an interrupted job can simply be restarted. Run from the repository root after
`pip install -e .`, e.g.

    python scripts/run_image.py --dataset cifar100 --data-dir data
"""

import argparse
import functools
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from lwce.data.cifar_lt import SPECS, load_cifar_lt, lt_class_counts
from lwce.losses import get_clf_loss
from lwce.runs import check_run_config, runs_in_checkpoint
from lwce.search import DEFAULT_PARAMS, GRIDS, LOSSES, run_grid_search
from lwce.train.image import (
    evaluate_metrics,
    group_indices,
    proxy_evaluate,
    proxy_subset,
    train_model,
)

BATCH_SIZE = 128
NUM_WORKERS = 0
PROXY_SEED = 42
AGG_KEYS = ['Top1_Acc', 'Balanced_Acc', 'F1_Macro', 'G_Mean', 'Worst_Acc',
            'Many_Acc', 'Medium_Acc', 'Few_Acc']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    p.add_argument('--irs', type=int, nargs='+', default=[10, 50, 100])
    p.add_argument('--losses', nargs='+', default=LOSSES, choices=LOSSES)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--proxy-epochs', type=int, default=20)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default=None,
                   help='default: results/cifar10_lt or results/cifar100_lt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = os.path.join('results', f'{args.dataset}_lt')
    return args


def _load_json(path, int_keys=False):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()} if int_keys else data


def _save_json(obj, path, indent=None):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def run_search(args, device):
    """Grid search per (IR, loss); returns {ir: {loss: best_params}}."""
    path_best = os.path.join(args.out_dir, 'optuna_checkpoint.json')
    path_trials = os.path.join(args.out_dir, 'optuna_trials.json')
    best = _load_json(path_best, int_keys=True)
    trials = _load_json(path_trials, int_keys=True)
    num_classes = SPECS[args.dataset]['num_classes']

    for ir in args.irs:
        pending = [ln for ln in GRIDS if ln in args.losses and ln not in best.get(ir, {})]
        if not pending:
            print(f'[IR={ir}] grid search: nothing pending')
            continue
        print(f'[IR={ir}] grid search: {pending}')

        train_loader, val_loader, _, class_counts = load_cifar_lt(
            args.dataset, ir, args.data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        proxy_ds = proxy_subset(train_loader.dataset, seed=PROXY_SEED)
        proxy_loader = DataLoader(proxy_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print(f'  proxy: {len(proxy_ds)} images')

        best_ir = dict(best.get(ir, {}))
        trials_ir = dict(trials.get(ir, {}))
        for loss_name in pending:
            evaluate = functools.partial(proxy_evaluate, loss_name, class_counts, proxy_loader,
                                         val_loader, num_classes, args.proxy_epochs, device)
            best_ir[loss_name], trials_ir[loss_name], _ = run_grid_search(
                GRIDS[loss_name], evaluate, cached_trials=trials_ir.get(loss_name),
                label=loss_name)

        best[ir] = best_ir
        trials[ir] = trials_ir
        _save_json({str(k): v for k, v in best.items()}, path_best, indent=2)
        _save_json({str(k): v for k, v in trials.items()}, path_trials, indent=2)
    return best


def run_final(args, best, device):
    """Final runs; returns {'IR{ir}_{loss}_s{seed}': {'metrics': ..., 'history': ...}}."""
    path = os.path.join(args.out_dir, 'results_checkpoint.json')
    ckpt = _load_json(path)
    num_classes = SPECS[args.dataset]['num_classes']

    for ir in args.irs:
        train_loader, val_loader, test_loader, class_counts = load_cifar_lt(
            args.dataset, ir, args.data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        for loss_name in args.losses:
            p = best.get(ir, {}).get(loss_name, {})
            kw = {k: p.get(k, v) for k, v in DEFAULT_PARAMS.items()}

            for seed in args.seeds:
                run_key = f'IR{ir}_{loss_name}_s{seed}'
                if run_key in ckpt:
                    continue
                print(f'[IR={ir}] {loss_name} seed={seed} ' +
                      ', '.join(f'{k}={v:.4g}' for k, v in kw.items()))

                model, history, _ = train_model(
                    loss_name, class_counts, train_loader, val_loader, num_classes, device,
                    epochs=args.epochs, seed=seed, **kw)

                metrics = evaluate_metrics(model, test_loader, num_classes, class_counts, device)
                metrics.update(kw)
                w = get_clf_loss(loss_name, class_counts, **kw).get_weights()
                metrics['class_weights'] = None if w is None else w.tolist()

                ckpt[run_key] = {'metrics': metrics, 'history': history}
                _save_json(ckpt, path)
                print(f'  Top1={metrics["Top1_Acc"]:.4f}  F1={metrics["F1_Macro"]:.4f}  '
                      f'Few={metrics["Few_Acc"]:.4f}')
    return ckpt


def aggregate(args, ckpt):
    """Write IR{ir}/results_per_seed.json and IR{ir}/results_agg.json (std with ddof=0)."""
    results = {ir: {ln: {} for ln in args.losses} for ir in args.irs}
    for run_key, run_data in ckpt.items():
        ir_str, loss_str, seed_str = run_key.split('_', 2)
        ir, seed = int(ir_str[2:]), int(seed_str[1:])
        if ir in results and loss_str in results[ir]:
            results[ir][loss_str][seed] = run_data['metrics']

    for ir in args.irs:
        many_idx, medium_idx, few_idx = group_indices(lt_class_counts(args.dataset, ir))
        agg_ir = {}
        print(f'\nIR={ir}: macro-F1 mean +- std over seeds')
        for loss_name in args.losses:
            seed_metrics = [results[ir][loss_name][s]
                            for s in args.seeds if s in results[ir][loss_name]]
            if not seed_metrics:
                continue
            # Group recalls recomputed from Per_Class_Acc as in the original aggregation
            # (same values for fresh runs; kept for reproducibility).
            for m in seed_metrics:
                pca = np.array(m.get('Per_Class_Acc', []))
                if len(pca) > 0:
                    m['Many_Acc'] = float(pca[many_idx].mean())
                    m['Medium_Acc'] = float(pca[medium_idx].mean())
                    m['Few_Acc'] = float(pca[few_idx].mean())
            agg = {}
            for key in AGG_KEYS:
                vals = [m.get(key, 0.0) for m in seed_metrics]
                agg[f'{key}_mean'] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
            agg_ir[loss_name] = agg
            print(f'  {loss_name:9s} {agg["F1_Macro_mean"]:.4f} +- {agg["F1_Macro_std"]:.4f}'
                  f'  (n={len(seed_metrics)})')

        ir_dir = os.path.join(args.out_dir, f'IR{ir}')
        os.makedirs(ir_dir, exist_ok=True)
        _save_json(agg_ir, os.path.join(ir_dir, 'results_agg.json'), indent=2)
        per_seed = {ln: {str(s): results[ir][ln].get(s, {}) for s in args.seeds}
                    for ln in args.losses}
        _save_json(per_seed, os.path.join(ir_dir, 'results_per_seed.json'), indent=2)


def aggregate_all(args, ckpt):
    """Aggregate every run in the checkpoint, so a partial rerun never shrinks the files."""
    losses = list(args.losses) + [ln for ln in LOSSES if ln not in args.losses]
    irs_found = {int(k.split('_', 1)[0][2:]) for k in ckpt}
    irs = [ir for ir in args.irs if ir in irs_found] + sorted(irs_found - set(args.irs))
    for ir in irs:
        ir_losses, ir_seeds = runs_in_checkpoint(ckpt, losses, args.seeds, prefix=f'IR{ir}_')
        aggregate(argparse.Namespace(**{**vars(args), 'irs': [ir], 'losses': ir_losses,
                                        'seeds': ir_seeds}), ckpt)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    check_run_config(args.out_dir, {'dataset': args.dataset, 'epochs': args.epochs,
                                    'proxy_epochs': args.proxy_epochs})
    best = run_search(args, device)
    ckpt = run_final(args, best, device)
    aggregate_all(args, ckpt)


if __name__ == '__main__':
    main()
