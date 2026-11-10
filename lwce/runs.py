"""Bookkeeping shared by the experiment scripts."""

import json
import os


def check_run_config(out_dir, config):
    """Refuse to resume an output folder that was filled with other training settings.

    Finished runs and grid-search trials are reused by name only, so a folder must hold
    runs made with one set of settings. The settings are stored in run_config.json the
    first time a folder is used; a later call with different settings stops here. A folder
    that already holds result files but no run_config.json is refused as well, because
    the settings of those results are unknown.
    """
    path = os.path.join(out_dir, 'run_config.json')
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved != config:
            raise SystemExit(f'{out_dir} holds runs made with {saved}, but this call uses '
                             f'{config}. Use another --out-dir or the same settings.')
        return
    if os.path.isdir(out_dir):   # result files may sit in subfolders (IR10/, analysis/)
        existing = sorted(os.path.relpath(os.path.join(d, f), out_dir)
                          for d, _, files in os.walk(out_dir) for f in files
                          if f.endswith('.json'))
        if existing:
            raise SystemExit(f'{out_dir} already holds results ({", ".join(existing[:3])}, ...) '
                             'but no run_config.json, so their training settings are unknown. '
                             'Use another --out-dir, or write run_config.json with the settings '
                             'those results were made with.')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def runs_in_checkpoint(keys, losses, seeds, prefix=''):
    """Losses and seeds of the finished runs stored under keys '{prefix}{loss}_s{seed}'.

    Used to aggregate every run in an output folder, not only those of the current call,
    so that a partial rerun never shrinks the result files. Losses keep the order of
    `losses` (keys of other losses are ignored); seeds keep the order of `seeds`, and seeds
    not in `seeds` follow in ascending order.
    """
    found_losses, found_seeds = set(), set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        loss, _, seed = key[len(prefix):].rpartition('_s')
        if loss in losses and seed.isdigit():
            found_losses.add(loss)
            found_seeds.add(int(seed))
    ordered_seeds = [s for s in seeds if s in found_seeds] + sorted(found_seeds - set(seeds))
    return [loss for loss in losses if loss in found_losses], ordered_seeds
