"""The saved results reproduce every cross-domain number printed in the paper."""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_paper_tables_have_no_mismatch():
    spec = importlib.util.spec_from_file_location(
        'paper_tables', os.path.join(ROOT, 'scripts', 'paper_tables.py'))
    pt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pt)
    rows = pt.compare(pt.compute(pt.load_settings()))
    bad = [r for r in rows if r[3] != 'ok']
    assert len(rows) == 147
    assert not bad, bad
