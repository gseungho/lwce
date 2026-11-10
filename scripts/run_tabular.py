"""Tabular domain: 11 datasets x 13 losses x 5 seeds with the MLP.

Per dataset: load -> proxy grid search of the tuned losses -> final runs -> save. Rerunning
the same command resumes: finished runs are skipped. Raw file names are listed in
lwce/data/tabular.py. Run from the repository root after `pip install -e .`, e.g.

    python scripts/run_tabular.py --data-dir data/tabular
"""

import argparse

import torch

from lwce.data.tabular import DATASETS, load_tabular_dataset
from lwce.runs import check_run_config
from lwce.search import LOSSES
from lwce.train.mlp import SEEDS, run_domain

TEST_BATCH_SIZE = 2048


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    p.add_argument('--losses', nargs='+', default=LOSSES, choices=LOSSES)
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--proxy-epochs', type=int, default=20)
    p.add_argument('--data-dir', default='data/tabular')
    p.add_argument('--out-dir', default='results/tabular')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_run_config(args.out_dir, {'domain': 'tabular', 'epochs': args.epochs,
                                    'proxy_epochs': args.proxy_epochs})
    run_domain(args.datasets, lambda ds: load_tabular_dataset(ds, args.data_dir),
               args.out_dir, 'mlp_all_results.json', args.losses, args.seeds,
               args.epochs, args.proxy_epochs, TEST_BATCH_SIZE, torch.device(args.device),
               all_datasets=DATASETS)


if __name__ == '__main__':
    main()
