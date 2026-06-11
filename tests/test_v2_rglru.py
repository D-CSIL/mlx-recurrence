"""Parity tests for the v2 RG-LRU scan (tiny shapes only — safe during training)."""

import mlx.core as mx
import pytest

from mlx_recurrence import rglru_scan, rglru_scan_with_state
from mlx_recurrence.rglru import rglru_scan_reference
from mlx_recurrence._chassis import parity_check


def _make_inputs(B, L, D, seed=0):
    mx.random.seed(seed)
    # a in (0, 1) so the recurrence is contractive (as RG-LRU's exp(-.) gate is)
    a = mx.sigmoid(mx.random.normal((B, L, D)) + 2.0)
    x = mx.random.normal((B, L, D)) * 0.5
    mx.eval(a, x)
    return a, x


@pytest.mark.parametrize("B,L,D,seg,seed", [
    (2, 64, 64, 32, 0),
    (1, 128, 32, 32, 1),
    (2, 96, 32, 16, 2),
])
def test_rglru_parity(B, L, D, seg, seed):
    a, x = _make_inputs(B, L, D, seed)

    ok, report = parity_check(
        kernel_fn=lambda *args: rglru_scan(*args, seg=seg),
        reference_fn=rglru_scan_reference,
        inputs=(a, x),
        arg_names=["a", "x"],
        grad_argnums=(0, 1),
        label=f"rglru[B{B} L{L} D{D} seg{seg}]",
    )
    assert ok, f"RG-LRU parity failed: {report}"


def test_rglru_final_state():
    B, L, D, seg = 2, 64, 64, 32
    a, x = _make_inputs(B, L, D, seed=3)

    _y, fs = rglru_scan_with_state(a, x, seg=seg)

    h = mx.zeros((B, D))
    for t in range(L):
        h = a[:, t, :] * h + x[:, t, :]
    mx.eval(fs, h)

    diff = float(mx.max(mx.abs(fs - h)))
    assert diff < 1e-3, f"RG-LRU final-state diff {diff:.2e}"


def test_rglru_invalid_shapes():
    a, x = _make_inputs(1, 30, 32)
    with pytest.raises(ValueError):
        mx.eval(rglru_scan(a, x, seg=32))

    a, x = _make_inputs(1, 32, 16)
    with pytest.raises(ValueError):
        mx.eval(rglru_scan(a, x, seg=32))


def test_rglru_parity_negative_gates():
    """The kernel must be range-agnostic in ``a`` — it only multiplies.

    Consumers (e.g. HELIX's oscillator-gated HSL) drive ``a`` in (-1, 1),
    outside RG-LRU's native (0, 1) envelope; this pins forward + gradient
    parity for sign-flipping gates in the package's own suite."""
    B, L, D, seg = 2, 64, 64, 32
    mx.random.seed(7)
    a = mx.random.uniform(low=-0.99, high=0.99, shape=(B, L, D))
    x = mx.random.normal(shape=(B, L, D)) * 0.5
    mx.eval(a, x)

    ok, report = parity_check(
        kernel_fn=lambda *args: rglru_scan(*args, seg=seg),
        reference_fn=rglru_scan_reference,
        inputs=(a, x),
        arg_names=["a", "x"],
        grad_argnums=(0, 1),
        label="rglru[negative gates]",
    )
    assert ok, f"RG-LRU negative-gate parity failed: {report}"
