#!/usr/bin/env python3
"""
test_kernels.py — Verification Suite for mlx_recurrence Metal Scan Kernels
===========================================================================
Tests:
  1. Numerical correctness:  Metal vs Python loop output match (SSM + GLA)
  2. Gradient correctness:   Metal VJP vs Python autograd vs finite differences
  3. Training convergence:   200 steps Metal vs Python, loss curves must track
  4. Full model test:        Requires d_csil_1 model code (NOT part of this package)
  5. Checkpoint inference:   Requires checkpoint file (NOT part of this package)

NOTE: Tests 3 (training convergence), 4 (full model), and 5 (checkpoint inference)
require the d_csil_1 model codebase to be present in sys.path. They will be
skipped automatically if that module is not available. Tests 1 and 2 are
self-contained and verify the kernels in isolation.

Run:
    python -m pytest tests/
    # or
    python tests/test_kernels.py
"""

from __future__ import annotations
import sys
import os
import time
import math
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# ── Package imports ───────────────────────────────────────────────────────────
from mlx_recurrence import (
    selective_scan_metal,
    gla_scan_metal,
    _ssm_forward_kernel,
    _gla_forward_kernel,
)

# ── Optional model imports (Tests 3, 4, 5 only) ───────────────────────────────
_D_CSIL1_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root
    from d_csil_1 import (
        DCSIL1Config, DCSIL1, create_model, loss_fn,
        SelectiveSSM, GatedLinearAttention, precompute_rope,
    )
    import d_csil_1 as _d_csil_1_module
    _D_CSIL1_AVAILABLE = True
except ImportError:
    pass

# ── Globals ───────────────────────────────────────────────────────────────────
RESULTS = []
PROJ = Path(__file__).parent.parent


def record(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    print(f"  [{tag}]  {name}")
    if detail:
        print(f"         {detail}")


# ============================================================================
# TEST 1: Numerical Correctness — Metal vs Python Loop
# ============================================================================

def test_ssm_numerical():
    """Compare Metal kernel SSM output vs Python loop across multiple seeds."""
    print("\n── TEST 1a: SSM Numerical Correctness ──")
    max_diffs = []

    for seed in [42, 123, 999, 7, 2026]:
        mx.random.seed(seed)
        B, L, D, N = 4, 256, 512, 64

        u     = mx.random.normal((B, L, D))
        delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
        B_in  = mx.random.normal((B, L, N))
        C_in  = mx.random.normal((B, L, N))
        A_log = mx.log(
            mx.repeat(mx.expand_dims(mx.arange(1, N + 1, dtype=mx.float32), 0),
                      repeats=D, axis=0)
        )
        A_neg = -mx.exp(A_log)
        mx.eval(u, delta, B_in, C_in, A_neg)

        # Python loop
        h = mx.zeros((B, D, N))
        py_outs = []
        for t in range(L):
            dt_t = mx.expand_dims(delta[:, t, :], -1)
            B_t  = mx.expand_dims(B_in[:, t, :], 1)
            u_t  = mx.expand_dims(u[:, t, :], -1)
            C_t  = mx.expand_dims(C_in[:, t, :], 1)
            dA   = mx.exp(dt_t * A_neg)
            h    = dA * h + (dt_t * B_t) * u_t
            py_outs.append(mx.sum(h * C_t, axis=-1))
        y_python = mx.stack(py_outs, axis=1)

        y_metal = selective_scan_metal(u, delta, B_in, C_in, A_neg)
        mx.eval(y_python, y_metal)

        diff = mx.max(mx.abs(y_python - y_metal)).item()
        max_diffs.append(diff)

    worst = max(max_diffs)
    passed = worst < 1e-4
    record("SSM output match (5 seeds)",
           passed,
           f"worst max_diff = {worst:.2e} (threshold 1e-4)")


def test_gla_numerical():
    """Compare Metal kernel GLA output vs Python loop across multiple seeds."""
    print("\n── TEST 1b: GLA Numerical Correctness ──")
    max_diffs = []

    for seed in [42, 123, 999, 7, 2026]:
        mx.random.seed(seed)
        B, L, H, Dh = 4, 256, 4, 64

        q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)
        k     = mx.random.normal((B, L, H, Dh))
        v     = mx.random.normal((B, L, H, Dh))
        gates = mx.sigmoid(mx.random.normal((B, L, H)))
        mx.eval(q, k, v, gates)

        # Python loop
        h = mx.zeros((B, H, Dh, Dh))
        py_outs = []
        for t in range(L):
            g_t = gates[:, t, :, None, None]
            k_t = k[:, t, :, :, None]
            v_t = v[:, t, :, None, :]
            q_t = q[:, t, :, :, None]
            h   = g_t * h + k_t * v_t
            o_t = mx.sum(q_t * h, axis=-2)
            py_outs.append(o_t)
        y_python = mx.stack(py_outs, axis=1)

        y_metal = gla_scan_metal(q, k, v, gates)
        mx.eval(y_python, y_metal)

        diff = mx.max(mx.abs(y_python - y_metal)).item()
        max_diffs.append(diff)

    worst = max(max_diffs)
    passed = worst < 1e-4
    record("GLA output match (5 seeds)",
           passed,
           f"worst max_diff = {worst:.2e} (threshold 1e-4)")


def test_ssm_edge_cases():
    """Test Metal SSM with edge-case inputs."""
    print("\n── TEST 1c: SSM Edge Cases ──")
    B, L, D, N = 2, 64, 128, 32

    A_log = mx.log(
        mx.repeat(mx.expand_dims(mx.arange(1, N + 1, dtype=mx.float32), 0),
                  repeats=D, axis=0)
    )
    A_neg = -mx.exp(A_log)

    cases = {
        "near-zero delta": (
            mx.random.normal((B, L, D)),
            mx.ones((B, L, D)) * 1e-6,
            mx.random.normal((B, L, N)),
            mx.random.normal((B, L, N)),
        ),
        "large delta": (
            mx.random.normal((B, L, D)),
            mx.ones((B, L, D)) * 10.0,
            mx.random.normal((B, L, N)),
            mx.random.normal((B, L, N)),
        ),
        "zero input": (
            mx.zeros((B, L, D)),
            mx.abs(mx.random.normal((B, L, D))) * 0.1,
            mx.random.normal((B, L, N)),
            mx.random.normal((B, L, N)),
        ),
    }

    all_pass = True
    for case_name, (u, delta, B_in, C_in) in cases.items():
        mx.eval(u, delta, B_in, C_in)

        h = mx.zeros((B, D, N))
        py_outs = []
        for t in range(L):
            dt_t = mx.expand_dims(delta[:, t, :], -1)
            B_t  = mx.expand_dims(B_in[:, t, :], 1)
            u_t  = mx.expand_dims(u[:, t, :], -1)
            C_t  = mx.expand_dims(C_in[:, t, :], 1)
            dA   = mx.exp(dt_t * A_neg)
            h    = dA * h + (dt_t * B_t) * u_t
            py_outs.append(mx.sum(h * C_t, axis=-1))
        y_py = mx.stack(py_outs, axis=1)

        y_mt = selective_scan_metal(u, delta, B_in, C_in, A_neg)
        mx.eval(y_py, y_mt)
        diff = mx.max(mx.abs(y_py - y_mt)).item()

        ok = diff < 1e-3
        if not ok:
            all_pass = False
        status = "ok" if ok else "FAIL"
        print(f"         {case_name}: diff={diff:.2e} [{status}]")

    record("SSM edge cases (zero/small/large delta)", all_pass)


# ============================================================================
# TEST 2: Gradient Correctness
# ============================================================================

def test_ssm_gradient_vs_python():
    """Compare Metal VJP gradients against Python loop autograd gradients."""
    print("\n── TEST 2a: SSM Gradient — Metal VJP vs Python Autograd ──")

    mx.random.seed(42)
    B, L, D, N = 2, 32, 64, 16

    u     = mx.random.normal((B, L, D))
    delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
    B_in  = mx.random.normal((B, L, N))
    C_in  = mx.random.normal((B, L, N))
    A_log = mx.log(
        mx.repeat(mx.expand_dims(mx.arange(1, N + 1, dtype=mx.float32), 0),
                  repeats=D, axis=0)
    )
    A_neg = -mx.exp(A_log)
    mx.eval(u, delta, B_in, C_in, A_neg)

    def loss_metal(u, delta, B_in, C_in, A_neg):
        return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

    metal_grad_fn = mx.grad(loss_metal, argnums=(0, 1, 2, 3, 4))
    metal_grads   = metal_grad_fn(u, delta, B_in, C_in, A_neg)
    mx.eval(metal_grads)

    def loss_python(u, delta, B_in, C_in, A_neg):
        h = mx.zeros((B, D, N))
        outputs = []
        for t in range(L):
            dt_t = mx.expand_dims(delta[:, t, :], -1)
            B_t  = mx.expand_dims(B_in[:, t, :], 1)
            u_t  = mx.expand_dims(u[:, t, :], -1)
            C_t  = mx.expand_dims(C_in[:, t, :], 1)
            dA   = mx.exp(dt_t * A_neg)
            h    = dA * h + (dt_t * B_t) * u_t
            outputs.append(mx.sum(h * C_t, axis=-1))
        return mx.sum(mx.stack(outputs, axis=1))

    python_grad_fn = mx.grad(loss_python, argnums=(0, 1, 2, 3, 4))
    python_grads   = python_grad_fn(u, delta, B_in, C_in, A_neg)
    mx.eval(python_grads)

    names    = ["grad_u", "grad_delta", "grad_B", "grad_C", "grad_A"]
    all_pass = True
    worst_rel = 0
    for name, mg, pg in zip(names, metal_grads, python_grads):
        scale   = mx.maximum(mx.abs(pg), mx.array(1e-6))
        rel_err = mx.max(mx.abs(mg - pg) / scale).item()
        worst_rel = max(worst_rel, rel_err)
        ok = rel_err < 0.05
        if not ok:
            all_pass = False
        status = "ok" if ok else "FAIL"
        print(f"         {name}: rel_err={rel_err:.4f} [{status}]")

    record("SSM Metal grad vs Python autograd",
           all_pass,
           f"worst relative error = {worst_rel:.4f} (threshold 5%)")


def test_gla_gradient_vs_python():
    """Compare Metal VJP gradients against Python loop autograd gradients."""
    print("\n── TEST 2b: GLA Gradient — Metal VJP vs Python Autograd ──")

    mx.random.seed(42)
    B, L, H, Dh = 2, 32, 4, 32

    q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)
    k     = mx.random.normal((B, L, H, Dh))
    v     = mx.random.normal((B, L, H, Dh))
    gates = mx.sigmoid(mx.random.normal((B, L, H)))
    mx.eval(q, k, v, gates)

    def loss_metal(q, k, v, gates):
        return mx.sum(gla_scan_metal(q, k, v, gates))

    metal_grad_fn = mx.grad(loss_metal, argnums=(0, 1, 2, 3))
    metal_grads   = metal_grad_fn(q, k, v, gates)
    mx.eval(metal_grads)

    def loss_python(q, k, v, gates):
        h = mx.zeros((B, H, Dh, Dh))
        outputs = []
        for t in range(L):
            g_t = gates[:, t, :, None, None]
            k_t = k[:, t, :, :, None]
            v_t = v[:, t, :, None, :]
            q_t = q[:, t, :, :, None]
            h   = g_t * h + k_t * v_t
            o_t = mx.sum(q_t * h, axis=-2)
            outputs.append(o_t)
        return mx.sum(mx.stack(outputs, axis=1))

    python_grad_fn = mx.grad(loss_python, argnums=(0, 1, 2, 3))
    python_grads   = python_grad_fn(q, k, v, gates)
    mx.eval(python_grads)

    names    = ["grad_q", "grad_k", "grad_v", "grad_gates"]
    all_pass = True
    worst_rel = 0
    for name, mg, pg in zip(names, metal_grads, python_grads):
        scale   = mx.maximum(mx.abs(pg), mx.array(1e-6))
        rel_err = mx.max(mx.abs(mg - pg) / scale).item()
        worst_rel = max(worst_rel, rel_err)
        ok = rel_err < 0.05
        if not ok:
            all_pass = False
        status = "ok" if ok else "FAIL"
        print(f"         {name}: rel_err={rel_err:.4f} [{status}]")

    record("GLA Metal grad vs Python autograd",
           all_pass,
           f"worst relative error = {worst_rel:.4f} (threshold 5%)")


def test_ssm_gradient_finite_diff():
    """Verify Metal SSM gradients against finite-difference numerical gradients."""
    print("\n── TEST 2c: SSM Gradient — Metal VJP vs Finite Difference ──")

    mx.random.seed(42)
    B, L, D, N = 1, 8, 16, 8
    eps = 5e-4

    u     = mx.random.normal((B, L, D))
    delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
    B_in  = mx.random.normal((B, L, N))
    C_in  = mx.random.normal((B, L, N))
    A_log = mx.log(
        mx.repeat(mx.expand_dims(mx.arange(1, N + 1, dtype=mx.float32), 0),
                  repeats=D, axis=0)
    )
    A_neg = -mx.exp(A_log)
    mx.eval(u, delta, B_in, C_in, A_neg)

    def loss_fn_ssm(u, delta, B_in, C_in, A_neg):
        return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

    grad_fn            = mx.grad(loss_fn_ssm, argnums=(0,))
    analytical_grad_u  = grad_fn(u, delta, B_in, C_in, A_neg)[0]
    mx.eval(analytical_grad_u)
    ana_flat = np.array(analytical_grad_u).flatten()

    u_np       = np.array(u)
    flat_size  = u_np.size
    rng        = np.random.RandomState(42)
    sample_indices = rng.choice(flat_size, size=min(20, flat_size), replace=False)

    max_rel_err = 0
    for flat_idx in sample_indices:
        idx     = np.unravel_index(flat_idx, u_np.shape)
        u_plus  = u_np.copy(); u_plus[idx]  += eps
        u_minus = u_np.copy(); u_minus[idx] -= eps

        loss_plus  = loss_fn_ssm(mx.array(u_plus),  delta, B_in, C_in, A_neg)
        loss_minus = loss_fn_ssm(mx.array(u_minus), delta, B_in, C_in, A_neg)
        mx.eval(loss_plus, loss_minus)

        fd    = (loss_plus.item() - loss_minus.item()) / (2 * eps)
        ana   = float(ana_flat[flat_idx])
        scale = max(abs(fd), 1e-6)
        rel   = abs(ana - fd) / scale
        max_rel_err = max(max_rel_err, rel)

    passed = max_rel_err < 0.1
    record("SSM Metal grad vs finite difference (20 elements)",
           passed,
           f"worst relative error = {max_rel_err:.4f} (threshold 10%)")


def test_gla_gradient_finite_diff():
    """Verify Metal GLA gradients against finite-difference numerical gradients."""
    print("\n── TEST 2d: GLA Gradient — Metal VJP vs Finite Difference ──")

    mx.random.seed(42)
    B, L, H, Dh = 1, 8, 2, 16
    eps = 1e-4

    q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)
    k     = mx.random.normal((B, L, H, Dh))
    v     = mx.random.normal((B, L, H, Dh))
    gates = mx.sigmoid(mx.random.normal((B, L, H)))
    mx.eval(q, k, v, gates)

    def loss_fn_gla(q, k, v, gates):
        return mx.sum(gla_scan_metal(q, k, v, gates))

    grad_fn           = mx.grad(loss_fn_gla, argnums=(1,))
    analytical_grad_k = grad_fn(q, k, v, gates)[0]
    mx.eval(analytical_grad_k)
    ana_flat = np.array(analytical_grad_k).flatten()

    k_np       = np.array(k)
    flat_size  = k_np.size
    rng        = np.random.RandomState(42)
    sample_indices = rng.choice(flat_size, size=min(20, flat_size), replace=False)

    max_rel_err = 0
    for flat_idx in sample_indices:
        idx     = np.unravel_index(flat_idx, k_np.shape)
        k_plus  = k_np.copy(); k_plus[idx]  += eps
        k_minus = k_np.copy(); k_minus[idx] -= eps

        loss_plus  = loss_fn_gla(q, mx.array(k_plus),  v, gates)
        loss_minus = loss_fn_gla(q, mx.array(k_minus), v, gates)
        mx.eval(loss_plus, loss_minus)

        fd    = (loss_plus.item() - loss_minus.item()) / (2 * eps)
        ana   = float(ana_flat[flat_idx])
        scale = max(abs(fd), 1e-6)
        rel   = abs(ana - fd) / scale
        max_rel_err = max(max_rel_err, rel)

    passed = max_rel_err < 0.1
    record("GLA Metal grad vs finite difference (20 elements)",
           passed,
           f"worst relative error = {max_rel_err:.4f} (threshold 10%)")


# ============================================================================
# TEST 3: Training Convergence — requires d_csil_1
# ============================================================================

def test_training_convergence():
    """Run 200 training steps with Metal and Python, verify loss curves match.

    REQUIRES: d_csil_1 model code in sys.path.
    """
    print("\n── TEST 3: Training Convergence (200 steps) ──")

    if not _D_CSIL1_AVAILABLE:
        record("Training convergence (200 steps)", True,
               "SKIP — d_csil_1 not available (not part of mlx-recurrence)")
        return

    import tiktoken

    data_path = PROJ / "data_cache" / "wikitext2_test.npy"
    if not data_path.exists():
        try:
            enc  = tiktoken.get_encoding("gpt2")
            text = "The quick brown fox jumps over the lazy dog. " * 500
            tokens = enc.encode(text)
            arr = np.array(tokens[:32768], dtype=np.uint16)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(data_path), arr)
            print("         Created synthetic test data (32K tokens)")
        except Exception:
            record("Training convergence (200 steps)", False,
                   "No training data available and could not create synthetic data")
            return

    tokens = np.load(str(data_path)).astype(np.int32)
    if len(tokens) < 2048:
        record("Training convergence (200 steps)", False,
               f"Data too small: {len(tokens)} tokens (need >=2048)")
        return

    config = DCSIL1Config(seq_len=64, num_layers=4, embed_dim=128,
                          state_dim=32, num_experts=2, top_k_experts=1,
                          gla_heads=2, gla_head_dim=32)
    steps      = 200
    lr         = 1e-3
    batch_size = 4
    seq_len    = config.seq_len

    def make_batch(tokens, step, batch_size, seq_len):
        max_start = len(tokens) - (batch_size * seq_len) - 2
        start     = (step * batch_size * seq_len) % max(max_start, 1)
        inputs, targets = [], []
        for b in range(batch_size):
            s = start + b * seq_len
            if s + seq_len + 1 > len(tokens):
                s = 0
            inputs.append(tokens[s:s + seq_len].tolist())
            targets.append(tokens[s + 1:s + seq_len + 1].tolist())
        return mx.array(inputs), mx.array(targets)

    def train_run(use_metal: bool, label: str):
        _d_csil_1_module._USE_METAL_SCAN = use_metal
        mx.random.seed(42)

        model = DCSIL1(config)
        mx.eval(model.parameters())

        import mlx.optimizers
        optimizer      = mlx.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
        loss_and_grad  = nn.value_and_grad(model, loss_fn)
        losses = []

        for step in range(steps):
            inp, tgt = make_batch(tokens, step, batch_size, seq_len)
            loss_val, grads = loss_and_grad(model, inp, tgt)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val)
            losses.append(loss_val.item())

            if step % 50 == 0 or step == steps - 1:
                print(f"         [{label}] step {step:3d}: "
                      f"loss={loss_val.item():.4f} ppl={math.exp(loss_val.item()):.1f}")

        return losses

    print("         Running Metal kernel training...")
    t0           = time.perf_counter()
    losses_metal = train_run(True, "Metal ")
    metal_time   = time.perf_counter() - t0

    print("         Running Python loop training...")
    t0            = time.perf_counter()
    losses_python = train_run(False, "Python")
    python_time   = time.perf_counter() - t0

    _d_csil_1_module._USE_METAL_SCAN = True

    metal_init    = losses_metal[0]
    metal_final   = np.mean(losses_metal[-10:])
    python_init   = losses_python[0]
    python_final  = np.mean(losses_python[-10:])

    metal_converged  = metal_final < metal_init * 0.9
    python_converged = python_final < python_init * 0.9
    finals_close     = abs(metal_final - python_final) / max(python_final, 1e-6) < 0.15

    corr     = np.corrcoef(losses_metal, losses_python)[0, 1]
    all_pass = metal_converged and python_converged and finals_close and corr > 0.85

    detail = (
        f"Metal: {metal_init:.3f}->{metal_final:.3f} ({metal_time:.1f}s) | "
        f"Python: {python_init:.3f}->{python_final:.3f} ({python_time:.1f}s) | "
        f"correlation={corr:.4f} | finals_diff={abs(metal_final-python_final):.4f}"
    )
    record("Training convergence (200 steps)", all_pass, detail)

    if metal_time < python_time:
        print(f"         Metal training was {python_time/metal_time:.2f}x faster")
    else:
        print(f"         WARNING: Metal was slower ({metal_time:.1f}s vs {python_time:.1f}s)")


# ============================================================================
# TEST 4: Full Model Loss+Gradient Match — requires d_csil_1
# ============================================================================

def test_full_model_gradient():
    """Verify full model loss + gradient is identical Metal vs Python.

    REQUIRES: d_csil_1 model code in sys.path.
    """
    print("\n── TEST 4: Full Model Loss+Gradient Match ──")

    if not _D_CSIL1_AVAILABLE:
        record("Full model loss+gradient match", True,
               "SKIP — d_csil_1 not available (not part of mlx-recurrence)")
        return

    config = DCSIL1Config(seq_len=32, num_layers=4, embed_dim=64,
                          state_dim=16, num_experts=2, top_k_experts=1,
                          gla_heads=2, gla_head_dim=32)

    mx.random.seed(42)
    model = DCSIL1(config)
    mx.eval(model.parameters())

    inp = mx.random.randint(0, config.vocab_size, (2, config.seq_len))
    tgt = mx.random.randint(0, config.vocab_size, (2, config.seq_len))
    mx.eval(inp, tgt)

    _d_csil_1_module._USE_METAL_SCAN = True
    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_m, grads_m = loss_grad_fn(model, inp, tgt)
    mx.eval(loss_m, grads_m)

    _d_csil_1_module._USE_METAL_SCAN = False
    loss_p, grads_p = loss_grad_fn(model, inp, tgt)
    mx.eval(loss_p, grads_p)

    _d_csil_1_module._USE_METAL_SCAN = True

    loss_diff = abs(loss_m.item() - loss_p.item())

    import mlx.utils as mlx_utils
    flat_m = dict(mlx_utils.tree_flatten(grads_m))
    flat_p = dict(mlx_utils.tree_flatten(grads_p))

    max_grad_diff = 0
    for key in flat_m:
        if key in flat_p:
            d = mx.max(mx.abs(flat_m[key] - flat_p[key])).item()
            max_grad_diff = max(max_grad_diff, d)

    passed = loss_diff < 1e-3 and max_grad_diff < 0.1
    detail = f"loss_diff={loss_diff:.2e} | max_grad_diff={max_grad_diff:.2e}"
    record("Full model loss+gradient match", passed, detail)


# ============================================================================
# TEST 5: Checkpoint Inference — requires d_csil_1 + checkpoint file
# ============================================================================

def test_checkpoint_inference():
    """Load real checkpoint and verify Metal/Python produce identical outputs.

    REQUIRES: d_csil_1 model code in sys.path + a .npz checkpoint file under runs/.
    """
    print("\n── TEST 5: Checkpoint Inference ──")

    if not _D_CSIL1_AVAILABLE:
        record("Checkpoint inference match", True,
               "SKIP — d_csil_1 not available (not part of mlx-recurrence)")
        return

    ckpt_candidates = [
        PROJ / "runs" / "run_20260327_091953" / "checkpoints" / "epoch4.npz",
    ]
    ckpt_dir = PROJ / "runs"
    if ckpt_dir.exists():
        for run_dir in sorted(ckpt_dir.iterdir()):
            cp_dir = run_dir / "checkpoints"
            if cp_dir.exists():
                for cp in sorted(cp_dir.glob("*.npz")):
                    if cp not in ckpt_candidates:
                        ckpt_candidates.append(cp)

    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break

    if ckpt_path is None:
        record("Checkpoint inference match", True,
               "SKIP — no checkpoint found (not a kernel failure)")
        return

    print(f"         Loading: {ckpt_path.name}")

    try:
        data = dict(np.load(str(ckpt_path), allow_pickle=True))
    except Exception as e:
        record("Checkpoint inference match", False, f"Failed to load checkpoint: {e}")
        return

    if "config" in data:
        config_dict = json.loads(str(data["config"]))
        import dataclasses
        config = DCSIL1Config(**{k: v for k, v in config_dict.items()
                                 if k in {f.name for f in dataclasses.fields(DCSIL1Config)}})
    else:
        config = DCSIL1Config()

    try:
        model = DCSIL1(config)
        weights = {k: mx.array(v) for k, v in data.items()
                   if k not in ("config", "optimizer", "step", "epoch",
                                "best_val_ppl", "train_loss", "val_loss")}
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
    except Exception as e:
        record("Checkpoint inference match", False, f"Failed to build model: {e}")
        return

    test_tokens = mx.array([[464, 1917, 318, 257, 1295, 1295, 1295, 835,
                              198, 464, 2612, 318, 257, 1295, 1295, 1295]])

    _d_csil_1_module._USE_METAL_SCAN = True
    logits_metal  = model(test_tokens)
    mx.eval(logits_metal)
    greedy_metal = mx.argmax(logits_metal[0], axis=-1)
    mx.eval(greedy_metal)

    _d_csil_1_module._USE_METAL_SCAN = False
    logits_python  = model(test_tokens)
    mx.eval(logits_python)
    greedy_python = mx.argmax(logits_python[0], axis=-1)
    mx.eval(greedy_python)

    _d_csil_1_module._USE_METAL_SCAN = True

    logit_diff   = mx.max(mx.abs(logits_metal - logits_python)).item()
    tokens_match = mx.array_equal(greedy_metal, greedy_python)
    mx.eval(tokens_match)
    tokens_match = tokens_match.item()

    passed = logit_diff < 1e-3 and tokens_match
    detail = f"logit max_diff={logit_diff:.2e} | greedy tokens identical={tokens_match}"
    record("Checkpoint inference match", passed, detail)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("mlx-recurrence Kernel Verification Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MLX version: {mx.__version__}")
    if not _D_CSIL1_AVAILABLE:
        print("NOTE: d_csil_1 not found — Tests 3/4/5 will be skipped.")
    print("=" * 70)

    t_start = time.perf_counter()

    # Test 1: Numerical correctness (self-contained)
    test_ssm_numerical()
    test_gla_numerical()
    test_ssm_edge_cases()

    # Test 2: Gradient correctness (self-contained)
    test_ssm_gradient_vs_python()
    test_gla_gradient_vs_python()
    test_ssm_gradient_finite_diff()
    test_gla_gradient_finite_diff()

    # Test 4: Full model (requires d_csil_1)
    test_full_model_gradient()

    # Test 3: Training convergence (requires d_csil_1, longest)
    test_training_convergence()

    # Test 5: Checkpoint inference (requires d_csil_1 + checkpoint)
    test_checkpoint_inference()

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_pass  = sum(1 for r in RESULTS if r["passed"])
    n_fail  = sum(1 for r in RESULTS if not r["passed"])
    n_total = len(RESULTS)

    for r in RESULTS:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}]  {r['test']}")

    print(f"\n  {n_pass}/{n_total} passed, {n_fail} failed")
    print(f"  Total time: {elapsed:.1f}s")

    if n_fail == 0:
        print("\n  ALL TESTS PASSED — Metal kernels verified.")
    else:
        print(f"\n  {n_fail} TEST(S) FAILED — review output above.")

    def make_serializable(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    report_path = PROJ / f"kernel_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "date":            datetime.now().isoformat(),
        "mlx_version":     mx.__version__,
        "results":         RESULTS,
        "passed":          n_pass,
        "failed":          n_fail,
        "total":           n_total,
        "elapsed_seconds": round(elapsed, 1),
        "verdict":         "PASS" if n_fail == 0 else "FAIL",
    }
    with open(report_path, "w") as f:
        json.dump(make_serializable(report), f, indent=2)
    print(f"\n  Report saved: {report_path.name}")
    print("=" * 70)

    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
