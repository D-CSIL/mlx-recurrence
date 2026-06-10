"""Parity tests for the v2 SSD scan (tiny shapes only — safe during training).

Shapes are kept small (B<=2, L<=128, dims<=64, fp32) so these allocate only
a few MB of GPU and run in seconds, per the live-training memory constraint.
"""

import mlx.core as mx
import pytest

from mlx_recurrence import ssd_scan, ssd_scan_with_state
from mlx_recurrence.ssd import ssd_scan_reference
from mlx_recurrence._chassis import parity_check


def _make_inputs(B, L, H, Dh, N, seed=0):
    mx.random.seed(seed)
    u     = mx.random.normal((B, L, H, Dh)) * 0.5
    delta = mx.abs(mx.random.normal((B, L, H))) * 0.05 + 0.001
    B_in  = mx.random.normal((B, L, H, N)) * 0.5
    C_in  = mx.random.normal((B, L, H, N)) * 0.5
    A_neg = -mx.abs(mx.random.normal((H, N))) - 0.1
    mx.eval(u, delta, B_in, C_in, A_neg)
    return u, delta, B_in, C_in, A_neg


@pytest.mark.parametrize("B,L,H,Dh,N,seg,seed", [
    (2, 64, 2, 64, 8, 32, 0),
    (1, 128, 3, 32, 16, 32, 1),
    (2, 96, 2, 32, 8, 16, 2),
])
def test_ssd_parity(B, L, H, Dh, N, seg, seed):
    u, delta, B_in, C_in, A_neg = _make_inputs(B, L, H, Dh, N, seed)

    ok, report = parity_check(
        kernel_fn=lambda *a: ssd_scan(*a, seg=seg),
        reference_fn=ssd_scan_reference,
        inputs=(u, delta, B_in, C_in, A_neg),
        arg_names=["u", "delta", "B", "C", "A_neg"],
        grad_argnums=(0, 1, 2, 3, 4),
        label=f"ssd[B{B} L{L} H{H} Dh{Dh} N{N} seg{seg}]",
    )
    assert ok, f"SSD parity failed: {report}"


def test_ssd_final_state():
    B, L, H, Dh, N, seg = 2, 64, 2, 64, 8, 32
    u, delta, B_in, C_in, A_neg = _make_inputs(B, L, H, Dh, N, seed=3)

    _y, fs = ssd_scan_with_state(u, delta, B_in, C_in, A_neg, seg=seg)

    # reference final state: [B, H, Dh, N]
    h = mx.zeros((B, H, Dh, N))
    for t in range(L):
        dt = delta[:, t, :, None, None]
        decay = mx.exp(dt * A_neg[None, :, None, :])
        h = decay * h + dt * B_in[:, t, :, None, :] * u[:, t, :, :, None]
    mx.eval(fs, h)

    diff = float(mx.max(mx.abs(fs - h)))
    assert diff < 1e-3, f"SSD final-state diff {diff:.2e}"


def test_ssd_invalid_shapes():
    # L not divisible by seg
    u, delta, B_in, C_in, A_neg = _make_inputs(1, 30, 2, 32, 8)
    with pytest.raises(ValueError):
        mx.eval(ssd_scan(u, delta, B_in, C_in, A_neg, seg=32))

    # Dh not a multiple of 32
    u, delta, B_in, C_in, A_neg = _make_inputs(1, 32, 2, 16, 8)
    with pytest.raises(ValueError):
        mx.eval(ssd_scan(u, delta, B_in, C_in, A_neg, seg=32))
