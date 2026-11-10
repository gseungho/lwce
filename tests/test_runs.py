"""Output-folder bookkeeping: settings guard and aggregation over every finished run."""

import importlib.util
import json
import os

import numpy as np
import pytest

from lwce.runs import check_run_config, runs_in_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved_config(folder):
    with open(os.path.join(ROOT, 'results', folder, 'run_config.json')) as f:
        return json.load(f)


def test_check_run_config_refuses_other_settings(tmp_path):
    check_run_config(str(tmp_path), {'epochs': 2, 'proxy_epochs': 1})
    check_run_config(str(tmp_path), {'epochs': 2, 'proxy_epochs': 1})
    with pytest.raises(SystemExit):
        check_run_config(str(tmp_path), {'epochs': 3, 'proxy_epochs': 1})


def test_check_run_config_refuses_results_without_a_record(tmp_path):
    """A folder with results but no run_config.json (review 2026-11-10, re-review F1)."""
    (tmp_path / '_best_params.json').write_text('{}')
    with pytest.raises(SystemExit):
        check_run_config(str(tmp_path), {'epochs': 2, 'proxy_epochs': 1})
    assert not (tmp_path / 'run_config.json').exists()
    check_run_config(str(tmp_path / 'new'), {'epochs': 2, 'proxy_epochs': 1})   # a new folder
    assert (tmp_path / 'new' / 'run_config.json').exists()


def test_check_run_config_looks_into_subfolders(tmp_path):
    """Only IR10/ aggregates and no record (final review 2026-11-10, F1)."""
    (tmp_path / 'IR10').mkdir()
    (tmp_path / 'IR10' / 'results_agg.json').write_text('{}')
    with pytest.raises(SystemExit):
        check_run_config(str(tmp_path), {'dataset': 'cifar10', 'epochs': 2, 'proxy_epochs': 1})
    assert not (tmp_path / 'run_config.json').exists()
    (tmp_path / 'empty_sub').mkdir()   # a folder with only empty subfolders is still new
    check_run_config(str(tmp_path / 'empty_sub'), {'epochs': 2, 'proxy_epochs': 1})


def test_image_cli_checks_the_settings_before_loading_data(tmp_path):
    image = _script('run_image')
    check_run_config(str(tmp_path), {'dataset': 'cifar10', 'epochs': 200, 'proxy_epochs': 20})
    with pytest.raises(SystemExit):
        image.main(['--dataset', 'cifar10', '--epochs', '2', '--out-dir', str(tmp_path)])


@pytest.mark.parametrize('domain, names', [('tabular', ['german_credit', 'yeast']),
                                           ('network', ['NSL-KDD', 'TON_IoT'])])
def test_mlp_cli_keeps_every_dataset_and_checks_the_settings(tmp_path, monkeypatch, domain,
                                                             names):
    """run_tabular.py / run_network.py main(): re-review 2026-11-10, F2."""
    script = _script(f'run_{domain}')
    data = {ds: _arrays(k + 2, 120) for k, ds in enumerate(names)}
    if domain == 'tabular':
        monkeypatch.setattr(script, 'load_tabular_dataset', lambda ds, data_dir: data[ds])
    else:
        monkeypatch.setattr(script, 'load_network_dataset', lambda ds: data[ds] + (['c'],))
    out = tmp_path / 'out'
    base = ['--losses', 'ce', 'lwce', '--seeds', '42', '--out-dir', str(out), '--device', 'cpu']
    settings = ['--epochs', '2', '--proxy-epochs', '1']
    for ds in reversed(names):   # one call per dataset, in reverse order
        script.main(['--datasets', ds, *settings, *base])
    all_name = 'mlp_all_results.json' if domain == 'tabular' else 'network_mlp_all_results.json'
    rows = json.loads((out / all_name).read_text())
    assert [r['Dataset'] for r in rows] == [ds for ds in names for _ in ('ce', 'lwce')]

    import lwce.train.mlp as mlp

    def fail(*a, **k):
        raise AssertionError('a finished run was trained again')

    monkeypatch.setattr(mlp, 'train_model', fail)
    script.main(['--datasets', names[0], *settings, *base])   # same settings: reuse
    assert json.loads((out / all_name).read_text()) == rows
    for changed in (['--epochs', '3', '--proxy-epochs', '1'],
                    ['--epochs', '2', '--proxy-epochs', '2']):
        with pytest.raises(SystemExit):
            script.main(['--datasets', names[0], *changed, *base])


def _arrays(num_classes, n_tr, n_te=60, d=6):
    """Small separable data in the layout the loaders return."""
    rng = np.random.default_rng(num_classes)
    y_tr = np.arange(n_tr) % num_classes
    y_te = np.arange(n_te) % num_classes
    centers = rng.normal(scale=2.0, size=(num_classes, d))
    X_tr = (centers[y_tr] + rng.normal(size=(n_tr, d))).astype(np.float32)
    X_te = (centers[y_te] + rng.normal(size=(n_te, d))).astype(np.float32)
    counts = [int((y_tr == c).sum()) for c in range(num_classes)]
    return X_tr, X_te, y_tr.astype(np.int64), y_te.astype(np.int64), counts, num_classes


def test_shipped_results_match_the_script_defaults():
    """A default call on the shipped results folders resumes instead of stopping."""
    image = _script('run_image')
    for ds in ('cifar10', 'cifar100'):
        a = image.parse_args(['--dataset', ds])
        assert _saved_config(f'{ds}_lt') == {'dataset': ds, 'epochs': a.epochs,
                                             'proxy_epochs': a.proxy_epochs}
    for domain in ('tabular', 'network'):
        a = _script(f'run_{domain}').parse_args([])
        assert _saved_config(domain) == {'domain': domain, 'epochs': a.epochs,
                                         'proxy_epochs': a.proxy_epochs}
    text = _script('run_text')
    assert _saved_config('text') == {'epochs': text.FINAL_EPOCHS,
                                     'proxy_epochs': text.PROXY_EPOCHS}


def test_runs_in_checkpoint_keeps_the_requested_order():
    keys = ['IR10_lwce_s43', 'IR10_ce_s42', 'IR100_ce_s44', 'IR10_rflog_s42', 'IR10_ce_s7']
    assert runs_in_checkpoint(keys, ['ce', 'lwce'], [43, 42], prefix='IR10_') == (
        ['ce', 'lwce'], [43, 42, 7])


def test_image_partial_rerun_keeps_every_run_in_the_aggregates(tmp_path):
    image = _script('run_image')

    def metrics(f):
        keys = ['Top1_Acc', 'Balanced_Acc', 'F1_Macro', 'G_Mean', 'Worst_Acc']
        return {**{k: f for k in keys}, 'Per_Class_Acc': [f] * 10}

    ckpt = {f'IR10_{loss}_s{seed}': {'metrics': metrics(seed / 100), 'history': {}}
            for loss in ('ce', 'lwce') for seed in (42, 43)}
    args = image.parse_args(['--dataset', 'cifar10', '--irs', '10', '--losses', 'ce',
                             '--seeds', '42', '--out-dir', str(tmp_path)])
    image.aggregate_all(args, ckpt)
    with open(tmp_path / 'IR10' / 'results_agg.json') as f:
        agg = json.load(f)
    with open(tmp_path / 'IR10' / 'results_per_seed.json') as f:
        per_seed = json.load(f)
    assert list(agg) == ['ce', 'lwce']
    assert agg['ce']['F1_Macro_mean'] == pytest.approx(0.425)
    assert {loss: list(v) for loss, v in per_seed.items()} == {'ce': ['42', '43'],
                                                              'lwce': ['42', '43']}
