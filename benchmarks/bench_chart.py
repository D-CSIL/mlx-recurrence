#!/usr/bin/env python3
"""
bench_chart.py — Generate benchmark charts for mlx-recurrence README.

Sweeps sequence lengths and compares Metal kernels vs Python fallback
for both forward and backward passes on SSM and GLA.

Outputs:
    benchmarks/ssm_benchmark.png
    benchmarks/gla_benchmark.png

Run:
    python benchmarks/bench_chart.py
"""

import time
import mlx.core as mx
import mlx.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from mlx_recurrence import (
    selective_scan_metal,
    selective_scan_chunked,
    gla_scan_metal,
    gla_scan_chunked,
)
from mlx_recurrence.ssm_scan import _ssm_forward_kernel, _ssm_backward_chunked
from mlx_recurrence.gla_scan import _gla_forward_kernel, _gla_backward_chunked


# ── Config ───────────────────────────────────────────────────────────
SEQ_LENS = [128, 256, 512, 1024, 2048]
BATCH = 2
SSM_INNER = 512
SSM_STATE = 64
GLA_HEADS = 12
GLA_HEAD_DIM = 64
WARMUP = 3
TRIALS = 8


def time_fn(fn, *args, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # ms


# ── SSM Benchmark ────────────────────────────────────────────────────

def ssm_python_loop(u, delta, B_in, C_in, A_neg):
    B_b, L, D = u.shape
    N = A_neg.shape[-1]
    h = mx.zeros((B_b, D, N))
    outputs = []
    for t in range(L):
        dt_t = mx.expand_dims(delta[:, t, :], -1)
        B_t = mx.expand_dims(B_in[:, t, :], 1)
        u_t = mx.expand_dims(u[:, t, :], -1)
        C_t = mx.expand_dims(C_in[:, t, :], 1)
        dA = mx.exp(dt_t * A_neg)
        h = dA * h + (dt_t * B_t) * u_t
        outputs.append(mx.sum(h * C_t, axis=-1))
    return mx.stack(outputs, axis=1)


def bench_ssm():
    print("Benchmarking SSM selective scan...")
    fwd_metal, fwd_python = [], []
    bwd_metal, bwd_python = [], []

    for L in SEQ_LENS:
        print(f"  seq_len={L}")
        u = mx.random.normal((BATCH, L, SSM_INNER))
        delta = mx.abs(mx.random.normal((BATCH, L, SSM_INNER))) * 0.1
        B_in = mx.random.normal((BATCH, L, SSM_STATE))
        C_in = mx.random.normal((BATCH, L, SSM_STATE))
        A_neg = -mx.exp(mx.log(
            mx.repeat(mx.expand_dims(mx.arange(1, SSM_STATE + 1, dtype=mx.float32), 0),
                      repeats=SSM_INNER, axis=0)
        ))
        mx.eval(u, delta, B_in, C_in, A_neg)

        # Forward: Metal
        fwd_metal.append(time_fn(selective_scan_metal, u, delta, B_in, C_in, A_neg))

        # Forward: Python loop
        fwd_python.append(time_fn(ssm_python_loop, u, delta, B_in, C_in, A_neg))

        # Backward: Metal (full forward+backward via grad)
        def loss_metal(u, delta, B_in, C_in, A_neg):
            return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

        grad_metal = mx.grad(loss_metal, argnums=(0, 1, 2, 3, 4))
        bwd_metal.append(time_fn(grad_metal, u, delta, B_in, C_in, A_neg))

        # Backward: Python fallback (forward + chunked backward via grad on chunked)
        def loss_chunked(u, delta, B_in, C_in, A_neg):
            return mx.sum(selective_scan_chunked(u, delta, B_in, C_in, A_neg))

        grad_chunked = mx.grad(loss_chunked, argnums=(0, 1, 2, 3, 4))
        bwd_python.append(time_fn(grad_chunked, u, delta, B_in, C_in, A_neg))

        print(f"    Fwd: Metal {fwd_metal[-1]:.1f}ms  Python {fwd_python[-1]:.1f}ms  "
              f"({fwd_python[-1]/fwd_metal[-1]:.1f}x)")
        print(f"    Bwd: Metal {bwd_metal[-1]:.1f}ms  Python {bwd_python[-1]:.1f}ms  "
              f"({bwd_python[-1]/bwd_metal[-1]:.1f}x)")

    return fwd_metal, fwd_python, bwd_metal, bwd_python


# ── GLA Benchmark ────────────────────────────────────────────────────

def gla_python_loop(q, k, v, gates):
    B_b, L, H, Dh = q.shape
    h = mx.zeros((B_b, H, Dh, Dh))
    outputs = []
    for t in range(L):
        g_t = gates[:, t, :, None, None]
        k_t = k[:, t, :, :, None]
        v_t = v[:, t, :, None, :]
        q_t = q[:, t, :, :, None]
        h = g_t * h + k_t * v_t
        o_t = mx.sum(q_t * h, axis=-2)
        outputs.append(o_t)
    return mx.stack(outputs, axis=1)


def bench_gla():
    print("\nBenchmarking GLA recurrence...")
    fwd_metal, fwd_python = [], []
    bwd_metal, bwd_python = [], []

    for L in SEQ_LENS:
        print(f"  seq_len={L}")
        q = mx.random.normal((BATCH, L, GLA_HEADS, GLA_HEAD_DIM)) * (GLA_HEAD_DIM ** -0.5)
        k = mx.random.normal((BATCH, L, GLA_HEADS, GLA_HEAD_DIM))
        v = mx.random.normal((BATCH, L, GLA_HEADS, GLA_HEAD_DIM))
        gates = mx.sigmoid(mx.random.normal((BATCH, L, GLA_HEADS)))
        mx.eval(q, k, v, gates)

        # Forward: Metal
        fwd_metal.append(time_fn(gla_scan_metal, q, k, v, gates))

        # Forward: Python loop
        fwd_python.append(time_fn(gla_python_loop, q, k, v, gates))

        # Backward: Metal
        def loss_metal(q, k, v, gates):
            return mx.sum(gla_scan_metal(q, k, v, gates))

        grad_metal = mx.grad(loss_metal, argnums=(0, 1, 2, 3))
        bwd_metal.append(time_fn(grad_metal, q, k, v, gates))

        # Backward: Python fallback
        def loss_chunked(q, k, v, gates):
            return mx.sum(gla_scan_chunked(q, k, v, gates))

        grad_chunked = mx.grad(loss_chunked, argnums=(0, 1, 2, 3))
        bwd_python.append(time_fn(grad_chunked, q, k, v, gates))

        print(f"    Fwd: Metal {fwd_metal[-1]:.1f}ms  Python {fwd_python[-1]:.1f}ms  "
              f"({fwd_python[-1]/fwd_metal[-1]:.1f}x)")
        print(f"    Bwd: Metal {bwd_metal[-1]:.1f}ms  Python {bwd_python[-1]:.1f}ms  "
              f"({bwd_python[-1]/bwd_metal[-1]:.1f}x)")

    return fwd_metal, fwd_python, bwd_metal, bwd_python


# ── Chart ────────────────────────────────────────────────────────────

def make_chart(seq_lens, fwd_metal, fwd_python, bwd_metal, bwd_python,
               title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    x = np.arange(len(seq_lens))
    width = 0.35

    # Forward
    bars1 = ax1.bar(x - width/2, fwd_python, width, label='MLX Fallback',
                    color='#e74c3c', alpha=0.85)
    bars2 = ax1.bar(x + width/2, fwd_metal, width, label='Metal Kernel',
                    color='#2ecc71', alpha=0.85)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Forward Pass', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lens)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add speedup labels
    for i, (p, m) in enumerate(zip(fwd_python, fwd_metal)):
        speedup = p / m
        ax1.annotate(f'{speedup:.1f}x', xy=(i + width/2, m),
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color='#27ae60')

    # Backward
    bars3 = ax2.bar(x - width/2, bwd_python, width, label='MLX Fallback',
                    color='#e74c3c', alpha=0.85)
    bars4 = ax2.bar(x + width/2, bwd_metal, width, label='Metal Kernel',
                    color='#2ecc71', alpha=0.85)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Forward + Backward (Training)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_lens)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    for i, (p, m) in enumerate(zip(bwd_python, bwd_metal)):
        speedup = p / m
        ax2.annotate(f'{speedup:.1f}x', xy=(i + width/2, m),
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color='#27ae60')

    fig.text(0.5, -0.02, 'M3 Max 36GB  |  Batch=2  |  mlx-recurrence v0.2.0',
             ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filename}")
    plt.close()


if __name__ == "__main__":
    ssm_fm, ssm_fp, ssm_bm, ssm_bp = bench_ssm()
    gla_fm, gla_fp, gla_bm, gla_bp = bench_gla()

    make_chart(SEQ_LENS, ssm_fm, ssm_fp, ssm_bm, ssm_bp,
               "SSM Selective Scan — Metal Kernel vs MLX Fallback",
               "benchmarks/ssm_benchmark.png")

    make_chart(SEQ_LENS, gla_fm, gla_fp, gla_bm, gla_bp,
               "GLA Recurrence — Metal Kernel vs MLX Fallback",
               "benchmarks/gla_benchmark.png")

    print("\nDone! Add these to your README:")
    print("  ![SSM Benchmark](benchmarks/ssm_benchmark.png)")
    print("  ![GLA Benchmark](benchmarks/gla_benchmark.png)")
