"""Hyperparameter grids and the exhaustive grid search used for every tuned loss.

Each tuned parameter is searched over 30 candidates (Combined: 10 x 6 = 60) scored by
proxy-validation macro-F1. Ties go to the first maximum; the grids are in ascending
order, so ties are broken toward the smaller value.
"""

import itertools

import numpy as np

# Losses run in every domain: the twelve of the main comparison plus Combined, which
# appears only in the ablation.
LOSSES = ['ce', 'wce', 'pwce', 'sqce', 'lwce', 'plwce', 'eslwce', 'combined', 'cb',
          'focal', 'logitadj', 'ldam', 'brlwce']

# Ablation rows: (loss, log compression, power alpha, smoothing eps).
ABLATION_ROWS = [('wce', 'X', 'X', 'X'), ('lwce', 'O', 'X', 'X'), ('plwce', 'O', 'O', 'X'),
                 ('eslwce', 'O', 'X', 'O'), ('combined', 'O', 'O', 'O')]


def _with_anchor(values, anchor):
    """Replace the grid point nearest to `anchor` by `anchor` itself.

    PLWCE at alpha = 1, ES-LWCE at eps = 0 and Combined at (1, 0) equal LWCE exactly;
    the anchors put that special case inside each grid.
    """
    v = list(map(float, values))
    v[int(np.argmin(np.abs(np.asarray(v) - anchor)))] = float(anchor)
    return sorted(v)


GRIDS = {
    'pwce': {'alpha': [0.0] + np.geomspace(0.01, 5.0, 29).tolist()},
    'plwce': {'alpha': _with_anchor(np.linspace(0.5, 6.0, 30), 1.0)},
    'focal': {'gamma': np.linspace(1.0, 5.0, 30).tolist()},
    'eslwce': {'eps': _with_anchor(np.logspace(-1, 1, 30), 0.0)},
    'logitadj': {'tau': np.linspace(0.25, 2.0, 30).tolist()},
    'combined': {'alpha': _with_anchor(np.linspace(0.5, 6.0, 10), 1.0),
                 'eps': _with_anchor(np.logspace(-1, 1, 6), 0.0)},
}

# Defaults used for the losses and parameters that are not tuned.
DEFAULT_PARAMS = {'alpha': 1.0, 'gamma': 2.0, 'eps': 0.1, 'tau': 1.0}


def run_grid_search(grid, evaluate, cached_trials=None, label=''):
    """Exhaustive search over the product of the grid axes.

    Trials in `cached_trials` with exactly the same parameters are reused instead of
    re-evaluated. This is valid because every proxy evaluation resets its seed, so a
    (loss, params) pair always gets the same score; discard the cache whenever the
    proxy split, seed, schedule or model changes.

    Args:
        grid: {param_name: [values]}.
        evaluate: callable(**params) -> float, larger is better.
        cached_trials: [{'params': {...}, 'value': float}, ...] from an earlier run.
        label: prefix for the progress line; nothing is printed if empty.

    Returns:
        (best_params, trials, n_reused).
    """
    keys = list(grid.keys())
    cache = {}
    for t in (cached_trials or []):
        params, value = t.get('params') or {}, t.get('value')
        if value is None or set(params) != set(keys):
            continue
        cache[tuple(round(float(params[k]), 12) for k in keys)] = float(value)

    trials, n_reused = [], 0
    for combo in itertools.product(*[list(grid[k]) for k in keys]):
        params = {k: float(v) for k, v in zip(keys, combo, strict=True)}
        key = tuple(round(params[k], 12) for k in keys)
        if key in cache:
            value, n_reused = cache[key], n_reused + 1
        else:
            value = float(evaluate(**params))
        trials.append({'params': params, 'value': value})

    best = max(trials, key=lambda t: t['value'])
    if label:
        print(f'  {label:9s} ' + ', '.join(f'{k}={v:.4g}' for k, v in best['params'].items())
              + f'  F1={best["value"]:.4f}  ({len(trials)} trials, {n_reused} reused)')
    return dict(best['params']), trials, n_reused
