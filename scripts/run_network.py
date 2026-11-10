"""Network-intrusion domain: 8 datasets x 13 losses x 5 seeds with the MLP.

Per dataset: download (kagglehub) and load -> proxy grid search of the tuned losses ->
final runs -> save. Rerunning the same command resumes: finished runs are skipped and a
finished dataset is not downloaded again. Run from the repository root after
`pip install -e .[network]`, e.g.

    python scripts/run_network.py
"""

import argparse

import torch

from lwce.data.network import DATASETS, load_network_dataset
from lwce.runs import check_run_config
from lwce.search import LOSSES
from lwce.train.mlp import SEEDS, run_domain

TEST_BATCH_SIZE = 4096


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    p.add_argument('--losses', nargs='+', default=LOSSES, choices=LOSSES)
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--proxy-epochs', type=int, default=10)
    p.add_argument('--out-dir', default='results/network')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_run_config(args.out_dir, {'domain': 'network', 'epochs': args.epochs,
                                    'proxy_epochs': args.proxy_epochs})
    run_domain(args.datasets, lambda ds: load_network_dataset(ds)[:6],
               args.out_dir, 'network_mlp_all_results.json', args.losses, args.seeds,
               args.epochs, args.proxy_epochs, TEST_BATCH_SIZE, torch.device(args.device),
               all_datasets=DATASETS)


if __name__ == '__main__':
    main()
