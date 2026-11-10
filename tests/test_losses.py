"""Properties of the proposed weights stated in the paper."""

import numpy as np
import pytest
import torch

from lwce.losses import PAPER_LOSSES, calculate_weights, get_clf_loss

COUNTS = [5000, 2000, 300, 40, 3]


def test_weights_have_mean_one():
    for mode in ['wce', 'pwce', 'sqce', 'lwce', 'plwce', 'eslwce', 'brlwce', 'combined', 'cb']:
        assert calculate_weights(COUNTS, mode).mean().item() == pytest.approx(1.0, abs=1e-6)


def test_lwce_weight_is_inverse_log():
    w = calculate_weights(COUNTS, 'lwce').numpy()
    raw = 1.0 / np.log1p(np.array(COUNTS, dtype=np.float64))
    assert np.allclose(w, raw / raw.mean(), rtol=1e-5)


def test_extensions_reduce_to_lwce():
    lwce = calculate_weights(COUNTS, 'lwce')
    assert torch.allclose(calculate_weights(COUNTS, 'plwce', alpha=1.0), lwce)
    assert torch.allclose(calculate_weights(COUNTS, 'eslwce', eps=0.0), lwce)
    assert torch.allclose(calculate_weights(COUNTS, 'combined', alpha=1.0, eps=0.0), lwce)


def test_brlwce_is_replication_invariant_and_lwce_is_not():
    for k in (2, 10, 100):
        scaled = [k * n for n in COUNTS]
        for mode, invariant in [('brlwce', True), ('lwce', False)]:
            same = torch.allclose(calculate_weights(scaled, mode), calculate_weights(COUNTS, mode))
            assert same == invariant, (mode, k)


def test_eslwce_is_finite_for_an_empty_class():
    w = calculate_weights([100, 10, 0], 'eslwce', eps=0.1)
    assert torch.isfinite(w).all()


@pytest.mark.parametrize('name', PAPER_LOSSES + ['combined'])
def test_every_loss_returns_a_finite_scalar(name):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(16, len(COUNTS), generator=g, requires_grad=True)
    targets = torch.randint(0, len(COUNTS), (16,), generator=g)
    loss = get_clf_loss(name, COUNTS)(logits, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
