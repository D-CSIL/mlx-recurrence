"""_utils.py — Shared kernel cache and scan utilities."""

from __future__ import annotations

import mlx.core as mx

# =============================================================================
# Metal Kernel Cache — compiled once per shape configuration
# =============================================================================

_kernel_cache: dict = {}


def _get_or_build_kernel(name: str, input_names: list, output_names: list,
                         source: str) -> object:
    """Cache compiled Metal kernels by name."""
    if name not in _kernel_cache:
        _kernel_cache[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )
    return _kernel_cache[name]


def _linear_scan_direct(decay, inp, chunk_size=32):
    """
    Chunked linear recurrence using direct decay values (not log-space).
    h[t] = decay[t] * h[t-1] + inp[t]

    Used by SSM and GLA backward passes where decay = dA (already exponentiated).

    Args:
        decay:      [B, L, D]  — decay coefficients in (0, 1)
        inp:        [B, L, D]  — input values
        chunk_size: timesteps per chunk

    Returns: [B, L, D]
    """
    B, L, D = decay.shape
    h_prev = mx.zeros((B, 1, D))
    chunks = []

    for t0 in range(0, L, chunk_size):
        t1        = min(t0 + chunk_size, L)
        dec_chunk = decay[:, t0:t1, :]
        inp_chunk = inp[:, t0:t1, :]

        log_dec  = mx.log(dec_chunk + 1e-38)
        log_P    = mx.cumsum(log_dec, axis=1)
        P        = mx.exp(log_P)

        inp_scaled = inp_chunk / (P + 1e-30)
        S          = mx.cumsum(inp_scaled, axis=1)
        h_chunk    = P * (h_prev + S)

        chunks.append(h_chunk)
        h_prev = h_chunk[:, -1:, :]

    return mx.concatenate(chunks, axis=1)
