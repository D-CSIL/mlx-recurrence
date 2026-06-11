"""Parity tests for the rotational LRU plug-in (rotlru).

Tiny shapes throughout (< 10 MB) — safe beside a live training job.

Coverage mirrors the other v2 plug-ins:
  - forward + EVERY gradient vs the pure-MLX reference (two shape configs,
    multi-segment)
  - negative magnitude gates (kernel must be range-agnostic)
  - theta = 0 reduces exactly to the RG-LRU recurrence
  - final-state variant matches the last step of y
  - shape-constraint validation raises
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_recurrence import parity_check
from mlx_recurrence.rotlru import (
    rotlru_scan,
    rotlru_scan_with_state,
    rotlru_scan_reference,
)
from mlx_recurrence.rglru import rglru_scan_reference

ARGS = ("a", "cs", "sn", "x")
GRADS = (0, 1, 2, 3)


def _inputs(B, L, Dp, seed=0, a_low=0.0, a_high=0.99):
    mx.random.seed(seed)
    theta = mx.random.uniform(low=0.0, high=math.pi, shape=(B, L, Dp))
    a = mx.random.uniform(low=a_low, high=a_high, shape=(B, L, Dp))
    cs = mx.cos(theta)
    sn = mx.sin(theta)
    x = mx.random.normal(shape=(B, L, 2 * Dp)) * 0.5
    return a, cs, sn, x


@pytest.mark.parametrize(
    "B,L,Dp,seg",
    [
        (2, 64, 32, 32),   # D=64, two segments
        (1, 96, 16, 32),   # D=32 edge, three segments
    ],
)
def test_parity_forward_and_grads(B, L, Dp, seg):
    inputs = _inputs(B, L, Dp, seed=B * 100 + L)
    ok, report = parity_check(
        lambda a, cs, sn, x: rotlru_scan(a, cs, sn, x, seg=seg),
        rotlru_scan_reference,
        inputs,
        ARGS,
        GRADS,
        label=f"rotlru B{B} L{L} Dp{Dp}",
    )
    assert ok, report


def test_parity_negative_gates():
    """a in (-0.99, 0.99): rotation + sign-flipping magnitude."""
    inputs = _inputs(2, 64, 32, seed=7, a_low=-0.99, a_high=0.99)
    ok, report = parity_check(
        lambda a, cs, sn, x: rotlru_scan(a, cs, sn, x),
        rotlru_scan_reference,
        inputs,
        ARGS,
        GRADS,
        label="rotlru negative-a",
    )
    assert ok, report


def test_theta_zero_reduces_to_rglru():
    """R(0) = I, so rotlru with cs=1, sn=0 must equal the RG-LRU scan on the
    same per-channel gates (each pair behaves as two independent channels)."""
    B, L, Dp = 2, 64, 32
    mx.random.seed(3)
    a = mx.random.uniform(low=-0.9, high=0.9, shape=(B, L, Dp))
    x = mx.random.normal(shape=(B, L, 2 * Dp))
    cs = mx.ones((B, L, Dp))
    sn = mx.zeros((B, L, Dp))

    y_rot = rotlru_scan(a, cs, sn, x)

    # Same gate applied to both members of each pair, channel-expanded.
    a_full = mx.repeat(a[..., None], 2, axis=-1).reshape(B, L, 2 * Dp)
    y_rglru = rglru_scan_reference(a_full, x)

    mx.eval(y_rot, y_rglru)
    assert float(mx.max(mx.abs(y_rot - y_rglru))) < 1e-4


def test_pure_rotation_preserves_norm():
    """With a = 1 and zero drive after t=0, the pair norm must be conserved
    (rotation is an isometry) — catches any cos/sin orientation bug."""
    B, L, Dp = 1, 64, 32
    mx.random.seed(5)
    theta = mx.random.uniform(low=0.1, high=3.0, shape=(B, L, Dp))
    a = mx.ones((B, L, Dp))
    cs, sn = mx.cos(theta), mx.sin(theta)
    x = mx.zeros((B, L, 2 * Dp))
    x0 = mx.random.normal(shape=(B, 1, 2 * Dp))
    x = mx.concatenate([x0, x[:, 1:]], axis=1)

    y = rotlru_scan(a, cs, sn, x)
    yp = y.reshape(B, L, Dp, 2)
    norms = mx.sqrt(yp[..., 0] ** 2 + yp[..., 1] ** 2)   # (B, L, Dp)
    mx.eval(norms)
    first = norms[:, 0, :]
    last = norms[:, -1, :]
    assert float(mx.max(mx.abs(last - first))) < 1e-4


def test_with_state_matches_last_step():
    a, cs, sn, x = _inputs(2, 64, 32, seed=11)
    y, final = rotlru_scan_with_state(a, cs, sn, x)
    mx.eval(y, final)
    assert final.shape == (2, 64)
    assert float(mx.max(mx.abs(final - y[:, -1, :]))) < 1e-6


def test_constraint_violations_raise():
    a, cs, sn, x = _inputs(1, 48, 16, seed=1)      # L=48 not divisible by 32
    with pytest.raises(ValueError):
        rotlru_scan(a, cs, sn, x)
    a, cs, sn, x = _inputs(1, 64, 8, seed=1)       # D=16 not divisible by 32
    with pytest.raises(ValueError):
        rotlru_scan(a, cs, sn, x)
    a, cs, sn, x = _inputs(1, 64, 16, seed=1)
    with pytest.raises(ValueError):
        rotlru_scan(a, cs, sn, x[:, :, :-2])       # x/Dp mismatch
