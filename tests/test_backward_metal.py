#!/usr/bin/env python3
"""
test_backward_metal.py — Metal Backward Kernel Verification
============================================================
Tests:
  1. SSM: Metal backward gradients match pure-MLX chunked backward
  2. GLA: Metal backward gradients match pure-MLX chunked backward
  3. SSM: Metal backward gradients match finite differences
  4. GLA: Metal backward gradients match finite differences
  5. Benchmark: Metal backward vs chunked backward timing

Run:
    python tests/test_backward_metal.py
"""

from __future__ import annotations
import sys
import time
import traceback
import mlx.core as mx
import numpy as np

# ── Package imports ──────────────────────────────────────────────────────────
from mlx_recurrence import (
    selective_scan_metal,
    selective_scan_chunked,
    gla_scan_metal,
    gla_scan_chunked,
    _ssm_forward_kernel,
    _ssm_backward_chunked,
    _ssm_backward_metal,
    _gla_forward_kernel,
    _gla_backward_chunked,
    _gla_backward_metal,
)

PASS = 0
FAIL = 0


def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


# =============================================================================
# Test 1: SSM Metal backward vs chunked backward
# =============================================================================

def test_ssm_backward_match():
    """Compare Metal backward output to chunked MLX backward."""
    print("\n── Test 1: SSM Metal backward vs Chunked backward ──")

    for B, L, D, N in [(1, 32, 16, 4), (2, 64, 32, 8), (4, 128, 48, 16)]:
        mx.random.seed(42)
        u     = mx.random.normal((B, L, D)) * 0.1
        delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
        B_in  = mx.random.normal((B, L, N)) * 0.1
        C_in  = mx.random.normal((B, L, N)) * 0.1
        A_neg = -mx.abs(mx.random.normal((D, N))) * 0.5
        mx.eval(u, delta, B_in, C_in, A_neg)

        # Forward pass to get h_all
        y_metal, h_all = _ssm_forward_kernel(u, delta, B_in, C_in, A_neg)
        mx.eval(y_metal, h_all)

        # Random gradient
        grad_y = mx.random.normal(y_metal.shape) * 0.1
        mx.eval(grad_y)

        # Metal backward
        gu_m, gd_m, gB_m, gC_m, gA_m = _ssm_backward_metal(
            grad_y, h_all, u, delta, B_in, C_in, A_neg
        )
        mx.eval(gu_m, gd_m, gB_m, gC_m, gA_m)

        # Chunked backward
        gu_c, gd_c, gB_c, gC_c, gA_c = _ssm_backward_chunked(
            grad_y, h_all, u, delta, B_in, C_in, A_neg
        )
        mx.eval(gu_c, gd_c, gB_c, gC_c, gA_c)

        label = f"B={B} L={L} D={D} N={N}"
        atol = 1e-3
        rtol = 1e-3

        pairs = [
            ("grad_u",     gu_m, gu_c),
            ("grad_delta", gd_m, gd_c),
            ("grad_B",     gB_m, gB_c),
            ("grad_C",     gC_m, gC_c),
            ("grad_A_neg", gA_m, gA_c),
        ]

        all_ok = True
        for gname, gm, gc in pairs:
            diff = float(mx.max(mx.abs(gm - gc)))
            scale = float(mx.max(mx.abs(gc))) + 1e-10
            rel = diff / scale
            ok = diff < atol or rel < rtol
            if not ok:
                all_ok = False
                print(f"    {gname}: max_abs_diff={diff:.6f}  rel={rel:.6f}")

        report(label, all_ok, f"atol={atol}")


# =============================================================================
# Test 2: GLA Metal backward vs chunked backward
# =============================================================================

def test_gla_backward_match():
    """Compare Metal backward output to chunked MLX backward."""
    print("\n── Test 2: GLA Metal backward vs Chunked backward ──")

    for B, L, H, Dh in [(1, 32, 2, 8), (2, 64, 4, 16), (2, 48, 4, 24)]:
        mx.random.seed(42)
        q     = mx.random.normal((B, L, H, Dh)) * 0.1
        k     = mx.random.normal((B, L, H, Dh)) * 0.1
        v     = mx.random.normal((B, L, H, Dh)) * 0.1
        gates = mx.sigmoid(mx.random.normal((B, L, H)))
        mx.eval(q, k, v, gates)

        # Forward pass to get h_all
        y_metal, h_all = _gla_forward_kernel(q, k, v, gates)
        mx.eval(y_metal, h_all)

        # Random gradient
        grad_out = mx.random.normal(y_metal.shape) * 0.1
        mx.eval(grad_out)

        # Metal backward
        gq_m, gk_m, gv_m, gg_m = _gla_backward_metal(
            grad_out, h_all, q, k, v, gates
        )
        mx.eval(gq_m, gk_m, gv_m, gg_m)

        # Chunked backward
        gq_c, gk_c, gv_c, gg_c = _gla_backward_chunked(
            grad_out, h_all, q, k, v, gates
        )
        mx.eval(gq_c, gk_c, gv_c, gg_c)

        label = f"B={B} L={L} H={H} Dh={Dh}"
        atol = 1e-3
        rtol = 1e-3

        pairs = [
            ("grad_q",     gq_m, gq_c),
            ("grad_k",     gk_m, gk_c),
            ("grad_v",     gv_m, gv_c),
            ("grad_gates", gg_m, gg_c),
        ]

        all_ok = True
        for gname, gm, gc in pairs:
            diff = float(mx.max(mx.abs(gm - gc)))
            scale = float(mx.max(mx.abs(gc))) + 1e-10
            rel = diff / scale
            ok = diff < atol or rel < rtol
            if not ok:
                all_ok = False
                print(f"    {gname}: max_abs_diff={diff:.6f}  rel={rel:.6f}")

        report(label, all_ok, f"atol={atol}")


# =============================================================================
# Test 3: SSM end-to-end gradient through mx.grad
# =============================================================================

def test_ssm_grad_e2e():
    """Verify Metal scan gradients are correct via finite differences.

    NOTE: We compare Metal against finite-diff (not chunked) because Metal
    and chunked forward paths produce slightly different float outputs
    (GPU vs CPU accumulation order). Both are correct for their own forward;
    the finite-diff test is the gold standard.
    """
    print("\n── Test 3: SSM end-to-end mx.grad vs finite-diff ──")

    B, L, D, N = 1, 16, 8, 4
    mx.random.seed(123)
    u     = mx.random.normal((B, L, D)) * 0.1
    delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
    B_in  = mx.random.normal((B, L, N)) * 0.1
    C_in  = mx.random.normal((B, L, N)) * 0.1
    A_neg = -mx.abs(mx.random.normal((D, N))) * 0.5
    mx.eval(u, delta, B_in, C_in, A_neg)

    def loss_fn(u, delta, B_in, C_in, A_neg):
        return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

    # Analytical gradients via Metal backward
    grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(u, delta, B_in, C_in, A_neg)
    mx.eval(*grads)

    # Finite-diff for each parameter (spot-check 5 random elements each)
    names = ["grad_u", "grad_delta", "grad_B", "grad_C", "grad_A_neg"]
    params = [u, delta, B_in, C_in, A_neg]
    all_ok = True
    eps = 1e-4

    for pname, param, grad_analytical in zip(names, params, grads):
        p_np = np.array(param)
        g_np = np.array(grad_analytical)
        rng = np.random.RandomState(42)
        indices = [tuple(rng.randint(0, s) for s in p_np.shape) for _ in range(5)]

        for idx in indices:
            p_plus = p_np.copy(); p_plus[idx] += eps
            p_minus = p_np.copy(); p_minus[idx] -= eps

            # Rebuild args with perturbed parameter
            args_plus = list(params)
            args_minus = list(params)
            pi = names.index(pname)
            args_plus[pi] = mx.array(p_plus)
            args_minus[pi] = mx.array(p_minus)

            l_plus = float(loss_fn(*args_plus))
            l_minus = float(loss_fn(*args_minus))
            fd = (l_plus - l_minus) / (2 * eps)
            analytical = float(g_np[idx])
            err = abs(fd - analytical) / (abs(fd) + 1e-10)

            if err > 0.10:
                all_ok = False
                print(f"    {pname}{idx}: fd={fd:.6f} analytical={analytical:.6f} rel_err={err:.4f}")

    report("SSM all grads match finite-diff (<10%)", all_ok)


# =============================================================================
# Test 4: GLA end-to-end gradient through mx.grad
# =============================================================================

def test_gla_grad_e2e():
    """Verify GLA Metal scan gradients are correct via finite differences."""
    print("\n── Test 4: GLA end-to-end mx.grad vs finite-diff ──")

    B, L, H, Dh = 1, 16, 1, 4
    mx.random.seed(123)
    q     = mx.random.normal((B, L, H, Dh)) * 0.1
    k     = mx.random.normal((B, L, H, Dh)) * 0.1
    v     = mx.random.normal((B, L, H, Dh)) * 0.1
    gates = mx.sigmoid(mx.random.normal((B, L, H)))
    mx.eval(q, k, v, gates)

    def loss_fn(q, k, v, gates):
        return mx.sum(gla_scan_metal(q, k, v, gates))

    grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3))(q, k, v, gates)
    mx.eval(*grads)

    names = ["grad_q", "grad_k", "grad_v", "grad_gates"]
    params = [q, k, v, gates]
    all_ok = True
    eps = 1e-4

    for pname, param, grad_analytical in zip(names, params, grads):
        p_np = np.array(param)
        g_np = np.array(grad_analytical)
        rng = np.random.RandomState(42)
        indices = [tuple(rng.randint(0, s) for s in p_np.shape) for _ in range(5)]

        for idx in indices:
            p_plus = p_np.copy(); p_plus[idx] += eps
            p_minus = p_np.copy(); p_minus[idx] -= eps

            args_plus = list(params)
            args_minus = list(params)
            pi = names.index(pname)
            args_plus[pi] = mx.array(p_plus)
            args_minus[pi] = mx.array(p_minus)

            l_plus = float(loss_fn(*args_plus))
            l_minus = float(loss_fn(*args_minus))
            fd = (l_plus - l_minus) / (2 * eps)
            analytical = float(g_np[idx])
            err = abs(fd - analytical) / (abs(fd) + 1e-10)

            if err > 0.05:
                all_ok = False
                print(f"    {pname}{idx}: fd={fd:.6f} analytical={analytical:.6f} rel_err={err:.4f}")

    report("GLA all grads match finite-diff (<5%)", all_ok)


# =============================================================================
# Test 5: SSM finite-difference gradient check
# =============================================================================

def test_ssm_finite_diff():
    """Verify Metal backward against finite differences."""
    print("\n── Test 5: SSM finite-difference gradient check ──")

    B, L, D, N = 1, 8, 4, 2
    mx.random.seed(77)
    u     = mx.random.normal((B, L, D)) * 0.1
    delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
    B_in  = mx.random.normal((B, L, N)) * 0.1
    C_in  = mx.random.normal((B, L, N)) * 0.1
    A_neg = -mx.abs(mx.random.normal((D, N))) * 0.5
    mx.eval(u, delta, B_in, C_in, A_neg)

    def loss_fn(u, delta, B_in, C_in, A_neg):
        return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

    grad_u_analytical = mx.grad(loss_fn, argnums=0)(u, delta, B_in, C_in, A_neg)
    mx.eval(grad_u_analytical)

    # Finite differences on u
    eps = 1e-4
    grad_u_fd = mx.zeros_like(u)
    u_np = np.array(u)
    fd_np = np.zeros_like(u_np)

    for idx in np.ndindex(u_np.shape):
        u_plus = u_np.copy()
        u_minus = u_np.copy()
        u_plus[idx] += eps
        u_minus[idx] -= eps
        l_plus = float(loss_fn(mx.array(u_plus), delta, B_in, C_in, A_neg))
        l_minus = float(loss_fn(mx.array(u_minus), delta, B_in, C_in, A_neg))
        fd_np[idx] = (l_plus - l_minus) / (2 * eps)

    grad_u_fd = mx.array(fd_np)
    diff = float(mx.max(mx.abs(grad_u_analytical - grad_u_fd)))
    scale = float(mx.max(mx.abs(grad_u_fd))) + 1e-10
    rel = diff / scale
    ok = diff < 1e-2 or rel < 1e-2
    report("SSM grad_u finite-diff", ok, f"max_diff={diff:.6f} rel={rel:.6f}")


# =============================================================================
# Test 6: GLA finite-difference gradient check
# =============================================================================

def test_gla_finite_diff():
    """Verify Metal backward against finite differences."""
    print("\n── Test 6: GLA finite-difference gradient check ──")

    B, L, H, Dh = 1, 8, 1, 4
    mx.random.seed(77)
    q     = mx.random.normal((B, L, H, Dh)) * 0.1
    k     = mx.random.normal((B, L, H, Dh)) * 0.1
    v     = mx.random.normal((B, L, H, Dh)) * 0.1
    gates = mx.sigmoid(mx.random.normal((B, L, H)))
    mx.eval(q, k, v, gates)

    def loss_fn(q, k, v, gates):
        return mx.sum(gla_scan_metal(q, k, v, gates))

    grad_q_analytical = mx.grad(loss_fn, argnums=0)(q, k, v, gates)
    mx.eval(grad_q_analytical)

    eps = 1e-4
    q_np = np.array(q)
    fd_np = np.zeros_like(q_np)

    for idx in np.ndindex(q_np.shape):
        q_plus = q_np.copy()
        q_minus = q_np.copy()
        q_plus[idx] += eps
        q_minus[idx] -= eps
        l_plus = float(loss_fn(mx.array(q_plus), k, v, gates))
        l_minus = float(loss_fn(mx.array(q_minus), k, v, gates))
        fd_np[idx] = (l_plus - l_minus) / (2 * eps)

    grad_q_fd = mx.array(fd_np)
    diff = float(mx.max(mx.abs(grad_q_analytical - grad_q_fd)))
    scale = float(mx.max(mx.abs(grad_q_fd))) + 1e-10
    rel = diff / scale
    ok = diff < 1e-2 or rel < 1e-2
    report("GLA grad_q finite-diff", ok, f"max_diff={diff:.6f} rel={rel:.6f}")


# =============================================================================
# Test 7: Benchmark — Metal backward vs chunked backward
# =============================================================================

def test_benchmark():
    """Time Metal backward vs chunked backward."""
    print("\n── Test 7: Backward Benchmark ──")

    # SSM benchmark
    B, L, D, N = 4, 256, 192, 16
    mx.random.seed(42)
    u     = mx.random.normal((B, L, D)) * 0.1
    delta = mx.abs(mx.random.normal((B, L, D))) * 0.1 + 0.01
    B_in  = mx.random.normal((B, L, N)) * 0.1
    C_in  = mx.random.normal((B, L, N)) * 0.1
    A_neg = -mx.abs(mx.random.normal((D, N))) * 0.5
    mx.eval(u, delta, B_in, C_in, A_neg)

    y_fwd, h_all = _ssm_forward_kernel(u, delta, B_in, C_in, A_neg)
    grad_y = mx.random.normal(y_fwd.shape) * 0.1
    mx.eval(y_fwd, h_all, grad_y)

    # Warmup
    for _ in range(3):
        _ssm_backward_metal(grad_y, h_all, u, delta, B_in, C_in, A_neg)
        _ssm_backward_chunked(grad_y, h_all, u, delta, B_in, C_in, A_neg)
    mx.eval()

    trials = 10

    t0 = time.perf_counter()
    for _ in range(trials):
        res = _ssm_backward_metal(grad_y, h_all, u, delta, B_in, C_in, A_neg)
        mx.eval(*res)
    ssm_metal_ms = (time.perf_counter() - t0) / trials * 1000

    t0 = time.perf_counter()
    for _ in range(trials):
        res = _ssm_backward_chunked(grad_y, h_all, u, delta, B_in, C_in, A_neg)
        mx.eval(*res)
    ssm_chunked_ms = (time.perf_counter() - t0) / trials * 1000

    ssm_speedup = ssm_chunked_ms / ssm_metal_ms
    print(f"  SSM [{B}x{L}x{D}, N={N}]:")
    print(f"    Metal:   {ssm_metal_ms:7.2f} ms")
    print(f"    Chunked: {ssm_chunked_ms:7.2f} ms")
    print(f"    Speedup: {ssm_speedup:.2f}x")
    report("SSM backward Metal within 2x", ssm_speedup > 0.5,
           f"{ssm_speedup:.2f}x")

    # GLA benchmark
    B, L, H, Dh = 4, 256, 4, 48
    mx.random.seed(42)
    q     = mx.random.normal((B, L, H, Dh)) * 0.1
    k     = mx.random.normal((B, L, H, Dh)) * 0.1
    v     = mx.random.normal((B, L, H, Dh)) * 0.1
    gates = mx.sigmoid(mx.random.normal((B, L, H)))
    mx.eval(q, k, v, gates)

    y_fwd, h_all_g = _gla_forward_kernel(q, k, v, gates)
    grad_out = mx.random.normal(y_fwd.shape) * 0.1
    mx.eval(y_fwd, h_all_g, grad_out)

    # Warmup
    for _ in range(3):
        _gla_backward_metal(grad_out, h_all_g, q, k, v, gates)
        _gla_backward_chunked(grad_out, h_all_g, q, k, v, gates)
    mx.eval()

    t0 = time.perf_counter()
    for _ in range(trials):
        res = _gla_backward_metal(grad_out, h_all_g, q, k, v, gates)
        mx.eval(*res)
    gla_metal_ms = (time.perf_counter() - t0) / trials * 1000

    t0 = time.perf_counter()
    for _ in range(trials):
        res = _gla_backward_chunked(grad_out, h_all_g, q, k, v, gates)
        mx.eval(*res)
    gla_chunked_ms = (time.perf_counter() - t0) / trials * 1000

    gla_speedup = gla_chunked_ms / gla_metal_ms
    print(f"\n  GLA [{B}x{L}x{H}x{Dh}]:")
    print(f"    Metal:   {gla_metal_ms:7.2f} ms")
    print(f"    Chunked: {gla_chunked_ms:7.2f} ms")
    print(f"    Speedup: {gla_speedup:.2f}x")
    report("GLA backward Metal within 2x", gla_speedup > 0.5,
           f"{gla_speedup:.2f}x")


# =============================================================================
# Test 8: Training convergence — Metal vs chunked loss curves
# =============================================================================

def test_training_convergence():
    """Run 50 steps of toy training with both backends, compare loss curves."""
    print("\n── Test 8: Training convergence (Metal vs Chunked) ──")

    B, L, D, N = 2, 32, 16, 4
    steps = 100
    lr = 0.01

    def run_training(scan_fn, seed):
        mx.random.seed(seed)
        # "Learnable" parameters — use small A_neg so states don't decay to zero
        u_param     = mx.random.normal((B, L, D)) * 0.5
        delta_param = mx.abs(mx.random.normal((B, L, D))) * 0.05 + 0.01
        A_neg       = -mx.abs(mx.random.normal((D, N))) * 0.05
        B_in        = mx.random.normal((B, L, N)) * 0.3
        C_in        = mx.random.normal((B, L, N)) * 0.3
        target      = mx.ones((B, L, D)) * 1.0  # clear target away from init
        mx.eval(u_param, delta_param, A_neg, B_in, C_in, target)

        losses = []
        for step in range(steps):
            def loss_fn(u_p, d_p, A_p):
                y = scan_fn(u_p, d_p, B_in, C_in, A_p)
                return mx.mean((y - target) ** 2)

            loss_val, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
                u_param, delta_param, A_neg
            )
            mx.eval(loss_val, *grads)
            losses.append(float(loss_val))

            u_param     = u_param - lr * grads[0]
            delta_param = delta_param - lr * grads[1]
            A_neg       = A_neg - lr * grads[2]
            mx.eval(u_param, delta_param, A_neg)

        return losses

    losses_metal   = run_training(selective_scan_metal, seed=42)
    losses_chunked = run_training(selective_scan_chunked, seed=42)

    # Both should decrease (any amount proves backprop is working)
    metal_decreased   = losses_metal[-1] < losses_metal[0] - 1e-6
    chunked_decreased = losses_chunked[-1] < losses_chunked[0] - 1e-6

    # Loss curves should track each other
    max_divergence = max(
        abs(m - c) / (abs(c) + 1e-10)
        for m, c in zip(losses_metal, losses_chunked)
    )

    print(f"    Metal   loss: {losses_metal[0]:.4f} -> {losses_metal[-1]:.4f}")
    print(f"    Chunked loss: {losses_chunked[0]:.4f} -> {losses_chunked[-1]:.4f}")
    print(f"    Max relative divergence: {max_divergence:.6f}")

    report("Both losses decrease", metal_decreased and chunked_decreased)
    report("Loss curves track (<5% divergence)", max_divergence < 0.05,
           f"div={max_divergence:.4f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  mlx-recurrence Metal Backward Kernel Tests")
    print("=" * 60)

    tests = [
        ("SSM backward match",    test_ssm_backward_match),
        ("GLA backward match",    test_gla_backward_match),
        ("SSM e2e grad",          test_ssm_grad_e2e),
        ("GLA e2e grad",          test_gla_grad_e2e),
        ("SSM finite-diff",       test_ssm_finite_diff),
        ("GLA finite-diff",       test_gla_finite_diff),
        ("Benchmark",             test_benchmark),
        ("Training convergence",  test_training_convergence),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            FAIL += 1
            print(f"\n  [FAIL] {name} — EXCEPTION:")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)
