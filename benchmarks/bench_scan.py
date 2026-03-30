#!/usr/bin/env python3
"""
bench_scan.py — Performance Benchmark for mlx_recurrence Scan Kernels

Run:
    python -m mlx_recurrence.benchmarks.bench_scan
    python benchmarks/bench_scan.py
"""

from __future__ import annotations
import time

import mlx.core as mx

from mlx_recurrence import (
    selective_scan_metal,
    selective_scan_chunked,
    gla_scan_metal,
    gla_scan_chunked,
)


def benchmark_scan(batch_size=4, seq_len=256, inner_dim=512, state_dim=64,
                   num_heads=4, head_dim=64, warmup=3, trials=10):
    """
    Compare Python loop vs Metal kernel vs Chunked scan performance.

    Args:
        batch_size: batch size B
        seq_len:    sequence length L
        inner_dim:  SSM inner/feature dimension D
        state_dim:  SSM state dimension N
        num_heads:  GLA number of heads H
        head_dim:   GLA head dimension Dh
        warmup:     warmup iterations (discarded)
        trials:     timed iterations (averaged)
    """
    print("=" * 70)
    print("mlx-recurrence Scan Kernel Benchmark")
    print("=" * 70)

    # ---- SSM Benchmark ----
    print(f"\n--- SSM Selective Scan ---")
    print(f"    Shape: B={batch_size}, L={seq_len}, D={inner_dim}, N={state_dim}")

    u     = mx.random.normal((batch_size, seq_len, inner_dim))
    delta = mx.abs(mx.random.normal((batch_size, seq_len, inner_dim))) * 0.1
    B_in  = mx.random.normal((batch_size, seq_len, state_dim))
    C_in  = mx.random.normal((batch_size, seq_len, state_dim))
    A_neg = -mx.exp(mx.log(
        mx.repeat(mx.expand_dims(mx.arange(1, state_dim + 1, dtype=mx.float32), 0),
                  repeats=inner_dim, axis=0)
    ))
    mx.eval(u, delta, B_in, C_in, A_neg)

    # Python loop baseline
    def ssm_python_loop(u, delta, B_in, C_in, A_neg):
        B_b, L, D = u.shape
        N = A_neg.shape[-1]
        h = mx.zeros((B_b, D, N))
        outputs = []
        for t in range(L):
            dt_t = mx.expand_dims(delta[:, t, :], -1)
            B_t  = mx.expand_dims(B_in[:, t, :], 1)
            u_t  = mx.expand_dims(u[:, t, :], -1)
            C_t  = mx.expand_dims(C_in[:, t, :], 1)
            dA   = mx.exp(dt_t * A_neg)
            h    = dA * h + (dt_t * B_t) * u_t
            outputs.append(mx.sum(h * C_t, axis=-1))
        return mx.stack(outputs, axis=1)

    for _ in range(warmup):
        y_py = ssm_python_loop(u, delta, B_in, C_in, A_neg); mx.eval(y_py)

    times_py = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y_py = ssm_python_loop(u, delta, B_in, C_in, A_neg); mx.eval(y_py)
        times_py.append(time.perf_counter() - t0)
    avg_py = sum(times_py) / len(times_py)
    print(f"    Python loop:   {avg_py*1000:.1f} ms")

    # Metal kernel
    try:
        for _ in range(warmup):
            y_metal = selective_scan_metal(u, delta, B_in, C_in, A_neg); mx.eval(y_metal)

        times_metal = []
        for _ in range(trials):
            t0 = time.perf_counter()
            y_metal = selective_scan_metal(u, delta, B_in, C_in, A_neg); mx.eval(y_metal)
            times_metal.append(time.perf_counter() - t0)
        avg_metal = sum(times_metal) / len(times_metal)
        print(f"    Metal kernel:  {avg_metal*1000:.1f} ms  ({avg_py/avg_metal:.1f}x faster)")

        diff = mx.abs(y_py - y_metal); mx.eval(diff)
        print(f"    Max diff (Python vs Metal): {diff.max().item():.2e}")
    except Exception as e:
        print(f"    Metal kernel FAILED: {e}")

    # Chunked scan
    for _ in range(warmup):
        y_chunk = selective_scan_chunked(u, delta, B_in, C_in, A_neg, chunk_size=32)
        mx.eval(y_chunk)

    times_chunk = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y_chunk = selective_scan_chunked(u, delta, B_in, C_in, A_neg, chunk_size=32)
        mx.eval(y_chunk)
        times_chunk.append(time.perf_counter() - t0)
    avg_chunk = sum(times_chunk) / len(times_chunk)
    print(f"    Chunked scan:  {avg_chunk*1000:.1f} ms  ({avg_py/avg_chunk:.1f}x faster)")

    diff_c = mx.abs(y_py - y_chunk); mx.eval(diff_c)
    print(f"    Max diff (Python vs Chunked): {diff_c.max().item():.2e}")

    # ---- GLA Benchmark ----
    print(f"\n--- GLA Recurrence ---")
    print(f"    Shape: B={batch_size}, L={seq_len}, H={num_heads}, Dh={head_dim}")

    q     = mx.random.normal((batch_size, seq_len, num_heads, head_dim)) * (head_dim ** -0.5)
    k     = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    v     = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    gates = mx.sigmoid(mx.random.normal((batch_size, seq_len, num_heads)))
    mx.eval(q, k, v, gates)

    # Python loop baseline
    def gla_python_loop(q, k, v, gates):
        B_b, L, H, Dh = q.shape
        h = mx.zeros((B_b, H, Dh, Dh))
        outputs = []
        for t in range(L):
            g_t = gates[:, t, :, None, None]
            k_t = k[:, t, :, :, None]
            v_t = v[:, t, :, None, :]
            q_t = q[:, t, :, :, None]
            h   = g_t * h + k_t * v_t
            o_t = mx.sum(q_t * h, axis=-2)
            outputs.append(o_t)
        return mx.stack(outputs, axis=1)

    for _ in range(warmup):
        y_py = gla_python_loop(q, k, v, gates); mx.eval(y_py)

    times_py = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y_py = gla_python_loop(q, k, v, gates); mx.eval(y_py)
        times_py.append(time.perf_counter() - t0)
    avg_py = sum(times_py) / len(times_py)
    print(f"    Python loop:   {avg_py*1000:.1f} ms")

    # Metal kernel
    try:
        for _ in range(warmup):
            y_metal = gla_scan_metal(q, k, v, gates); mx.eval(y_metal)

        times_metal = []
        for _ in range(trials):
            t0 = time.perf_counter()
            y_metal = gla_scan_metal(q, k, v, gates); mx.eval(y_metal)
            times_metal.append(time.perf_counter() - t0)
        avg_metal = sum(times_metal) / len(times_metal)
        print(f"    Metal kernel:  {avg_metal*1000:.1f} ms  ({avg_py/avg_metal:.1f}x faster)")

        diff = mx.abs(y_py - y_metal); mx.eval(diff)
        print(f"    Max diff (Python vs Metal): {diff.max().item():.2e}")
    except Exception as e:
        print(f"    Metal kernel FAILED: {e}")

    # Chunked scan
    for _ in range(warmup):
        y_chunk = gla_scan_chunked(q, k, v, gates, chunk_size=32); mx.eval(y_chunk)

    times_chunk = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y_chunk = gla_scan_chunked(q, k, v, gates, chunk_size=32); mx.eval(y_chunk)
        times_chunk.append(time.perf_counter() - t0)
    avg_chunk = sum(times_chunk) / len(times_chunk)
    print(f"    Chunked scan:  {avg_chunk*1000:.1f} ms  ({avg_py/avg_chunk:.1f}x faster)")

    diff_c = mx.abs(y_py - y_chunk); mx.eval(diff_c)
    print(f"    Max diff (Python vs Chunked): {diff_c.max().item():.2e}")

    # ---- Gradient Flow Test ----
    print(f"\n--- Gradient Flow Test ---")
    try:
        def loss_ssm(u, delta, B_in, C_in, A_neg):
            return mx.sum(selective_scan_metal(u, delta, B_in, C_in, A_neg))

        grad_fn = mx.grad(loss_ssm, argnums=(0, 1, 2, 3, 4))
        grads   = grad_fn(u, delta, B_in, C_in, A_neg)
        mx.eval(grads)
        print(f"    SSM Metal grad shapes: {[g.shape for g in grads]}")
        norms = [f"{mx.sqrt(mx.sum(g*g)).item():.4f}" for g in grads]
        print(f"    SSM Metal grad norms:  {norms}")
        print("    SSM gradient: PASS")
    except Exception as e:
        print(f"    SSM gradient FAILED: {e}")

    try:
        def loss_gla(q, k, v, gates):
            return mx.sum(gla_scan_metal(q, k, v, gates))

        grad_fn = mx.grad(loss_gla, argnums=(0, 1, 2, 3))
        grads   = grad_fn(q, k, v, gates)
        mx.eval(grads)
        print(f"    GLA Metal grad shapes: {[g.shape for g in grads]}")
        norms = [f"{mx.sqrt(mx.sum(g*g)).item():.4f}" for g in grads]
        print(f"    GLA Metal grad norms:  {norms}")
        print("    GLA gradient: PASS")
    except Exception as e:
        print(f"    GLA gradient FAILED: {e}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_scan()
