"""Per-domain analysis of the MLP runs (tabular or network).

Reads the files written by run_tabular.py / run_network.py and writes
<results-dir>/analysis/analysis_summary.json with
    grad_summary    final-epoch gradient norm and few/many gradient ratio (mean over seeds)
    ablation        macro-F1 mean/std, worst-class recall and G-mean of the ablation rows
    weight_ratios   w_rare / w_freq of every re-weighting loss at its selected parameters
The original notebook cell also drew PNG figures; they are not reproduced here. Run from
the repository root after `pip install -e .`, e.g.

    python scripts/analyze_mlp.py --domain tabular
"""

import argparse
import json
import os

import numpy as np

from lwce.data import network, tabular
from lwce.losses import get_clf_loss
from lwce.search import ABLATION_ROWS, LOSSES
from lwce.train.mlp import SEEDS

DOMAINS = {'tabular': (tabular.DATASETS, 'mlp_all_results.json'),
           'network': (network.DATASETS, 'network_mlp_all_results.json')}

THEORY = {'wce': 'rho', 'pwce': 'rho^alpha', 'sqce': 'sqrt(rho)',
          'lwce': 'O(log rho)', 'plwce': 'O((log rho)^alpha)',
          'eslwce': '<= LWCE', 'combined': 'O((log rho)^alpha)', 'cb': '-'}


def grad_summary(all_histories, ds_list):
    print('Gradient statistics (last epoch, mean over seeds)')
    print(f'{"Dataset":22s} | {"Loss":10s} | {"grad_norm":>10} | {"Few/Many":>9}')
    print('-' * 60)
    out = {}
    for ds in ds_list:
        for loss_name in LOSSES:
            hs = [all_histories[ds][f'{loss_name}_s{s}'] for s in SEEDS
                  if f'{loss_name}_s{s}' in all_histories[ds]]
            if not hs or 'grad_norm' not in hs[0]:
                continue
            gn = float(np.mean([h['grad_norm'][-1] for h in hs]))
            gr = float(np.nanmean([h['grad_ratio'][-1] for h in hs]))
            out[f'{ds}_{loss_name}'] = {'grad_norm_final': gn, 'grad_ratio_final': gr}
            print(f'{ds:22s} | {loss_name:10s} | {gn:10.4f} | {gr:9.3f}')
    return out


def weight_ratios(raw_counts, all_results, ds_list):
    print('\nWeight dynamic range w_rare / w_freq')
    print(f'{"Dataset":22s} | {"IR":>8} | {"Loss":10s} | {"w_rare/w_freq":>14} | theory')
    print('-' * 78)
    rows = []
    for ds in ds_list:
        cc = raw_counts[ds]
        order = np.argsort(np.array(cc))[::-1]
        ir = max(cc) / max(1, min(cc))
        for loss_name in LOSSES:
            # Parameters as stored in the first result row of this (dataset, loss).
            row = next((r for r in all_results
                        if r['Dataset'] == ds and r['Loss'] == loss_name), None)
            if row is None:
                continue
            kw = {k: row[k] for k in ('alpha', 'gamma', 'eps', 'tau') if k in row}
            w = get_clf_loss(loss_name, cc, **kw).get_weights()
            if w is not None:   # None: unweighted (ce, focal, logitadj, ldam)
                ratio = w[order[-1]] / w[order[0]]
                print(f'{ds:22s} | {ir:8.1f} | {loss_name:10s} | {ratio:14.2f} | '
                      f'{THEORY.get(loss_name, "-")}')
                rows.append({'Dataset': ds, 'IR': float(ir), 'Loss': loss_name,
                             'w_rare_over_w_freq': float(ratio)})
    return rows


def ablation(all_results, ds_list):
    print('\nAblation  [log compression / power alpha / smoothing eps]')
    print(f'{"Dataset":22s} | {"Method":10s} | {"LC":^3} | {"a":^3} | {"e":^3} | '
          f'{"F1-Macro":^15} | {"Worst":^6} | {"G-Mean":^6}')
    print('-' * 90)
    rows = []
    for ds in ds_list:
        for name, lc, pw, sm in ABLATION_ROWS:
            rs = [r for r in all_results if r['Dataset'] == ds and r['Loss'] == name]
            if not rs:
                continue
            f1 = [r['F1_Macro'] for r in rs]
            wo = float(np.mean([r.get('Worst_Acc', 0) for r in rs]))
            gm = float(np.mean([r.get('G_Mean', 0) for r in rs]))
            print(f'{ds:22s} | {name:10s} | {lc:^3} | {pw:^3} | {sm:^3} | '
                  f'{np.mean(f1):.4f}+/-{np.std(f1):.4f} | {wo:6.4f} | {gm:6.4f}')
            rows.append({'Dataset': ds, 'Method': name, 'LogCompression': lc,
                         'Power_alpha': pw, 'Smoothing_eps': sm,
                         'F1_Macro_mean': float(np.mean(f1)),
                         'F1_Macro_std': float(np.std(f1)),
                         'Worst_Acc': wo, 'G_Mean': gm})
    return rows


def analyze(results_dir, domain):
    datasets, all_results_name = DOMAINS[domain]
    with open(os.path.join(results_dir, '_class_counts.json')) as f:
        raw_counts = json.load(f)
    with open(os.path.join(results_dir, '_histories.json')) as f:
        all_histories = json.load(f)
    with open(os.path.join(results_dir, all_results_name)) as f:
        all_results = json.load(f)
    ds_list = [d for d in datasets if d in all_histories and d in raw_counts]

    grads = grad_summary(all_histories, ds_list)
    weights = weight_ratios(raw_counts, all_results, ds_list)
    summary = {'grad_summary': grads, 'ablation': ablation(all_results, ds_list),
               'weight_ratios': weights}
    analysis_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {analysis_dir}/analysis_summary.json')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', required=True, choices=list(DOMAINS))
    p.add_argument('--results-dir', help='default: results/<domain>')
    args = p.parse_args(argv)
    analyze(args.results_dir or os.path.join('results', args.domain), args.domain)


if __name__ == '__main__':
    main()
