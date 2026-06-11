"""Parity tests for the v2 GLA scan (tiny shapes only — safe during training)."""

import mlx.core as mx
import pytest

from mlx_recurrence import gla_scan, gla_scan_with_state
from mlx_recurrence.gla import gla_scan_reference
from mlx_recurrence._chassis import parity_check


def _make_inputs(B, L, H, Dh, seed=0):
    mx.random.seed(seed)
    q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)
    k     = mx.random.normal((B, L, H, Dh)) * 0.5
    v     = mx.random.normal((B, L, H, Dh)) * 0.5
    gates = mx.sigmoid(mx.random.normal((B, L, H)) + 2.0)
    mx.eval(q, k, v, gates)
    return q, k, v, gates


@pytest.mark.parametrize("B,L,H,Dh,seg,seed", [
    (2, 64, 2, 64, 32, 0),
    (1, 128, 3, 32, 32, 1),
    (2, 96, 2, 32, 16, 2),
])
def test_gla_parity(B, L, H, Dh, seg, seed):
    q, k, v, gates = _make_inputs(B, L, H, Dh, seed)

    ok, report = parity_check(
        kernel_fn=lambda *a: gla_scan(*a, seg=seg),
        reference_fn=gla_scan_reference,
        inputs=(q, k, v, gates),
        arg_names=["q", "k", "v", "gates"],
        grad_argnums=(0, 1, 2, 3),
        label=f"gla[B{B} L{L} H{H} Dh{Dh} seg{seg}]",
    )
    assert ok, f"GLA parity failed: {report}"


def test_gla_final_state():
    B, L, H, Dh, seg = 2, 64, 2, 64, 32
    q, k, v, gates = _make_inputs(B, L, H, Dh, seed=3)

    _y, fs = gla_scan_with_state(q, k, v, gates, seg=seg)

    h = mx.zeros((B, H, Dh, Dh))
    for t in range(L):
        h = (gates[:, t, :, None, None] * h
             + k[:, t, :, :, None] * v[:, t, :, None, :])
    mx.eval(fs, h)

    diff = float(mx.max(mx.abs(fs - h)))
    assert diff < 1e-3, f"GLA final-state diff {diff:.2e}"


def test_gla_invalid_shapes():
    q, k, v, gates = _make_inputs(1, 30, 2, 32)
    with pytest.raises(ValueError):
        mx.eval(gla_scan(q, k, v, gates, seg=32))

    q, k, v, gates = _make_inputs(1, 32, 2, 16)
    with pytest.raises(ValueError):
        mx.eval(gla_scan(q, k, v, gates, seg=32))
