"""rglru.py — RG-LRU diagonal recurrence (RecurrentGemma / Griffin).

NEW v2 kernel. This is the simplest plug-in in the framework and exists to
demonstrate that the chassis (segment checkpoint + recompute backward,
shape validation, parity helper) generalises to a new recurrence with
almost no new code: a diagonal (scalar-per-channel) linear scan.

RG-LRU correspondence
---------------------
The Griffin / RecurrentGemma "Real-Gated Linear Recurrent Unit" computes::

    a_t = exp(-c * softplus(Lambda) * sigmoid(r_t))           # recurrent gate
    h_t = a_t (.) h_{t-1} + sqrt(1 - a_t^2) (.) (i_t (.) x_t)  # diagonal scan

where ``(.)`` is elementwise over the channel dim ``D``, ``r_t`` is the
recurrence-gate pre-activation, ``i_t`` the input gate, and ``Lambda`` a
learned per-channel decay parameter.

This kernel deliberately handles ONLY the inner linear scan::

    h_t = a_t (.) h_{t-1} + b_t

taking the gate ``a`` ([B, L, D]) and the already-gated input
``b = sqrt(1 - a^2) (.) (i (.) x)`` ([B, L, D]) as inputs. Computing ``a``
and ``b`` from ``Lambda, r, i, x`` is cheap, elementwise, and fully
auto-differentiable in pure MLX, so it stays OUTSIDE the kernel; the kernel
is the part with the sequential dependency that MLX cannot fuse for free.

State is diagonal: one scalar ``h[d]`` per channel, conceptually
``[B, D]``. There is no inner state loop and no cross-lane reduction — each
channel's gradients are independent — so unlike the SSD/GLA kernels this one
needs no ``simd_sum``. It still follows the segment-checkpoint + recompute
backward pattern so it composes with the rest of the framework and supports
chunked prefill via a final-state variant.

VJP of ``h_t = a_t * h_{t-1} + b_t`` (output y_t = h_t, h_0 carry = 0):
    adj_t      = grad_y_t + a_{t+1} * adj_{t+1}      (reverse recurrence)
    grad_b[t]  = adj_t
    grad_a[t]  = adj_t * h_{t-1}     (h_{-1} = 0)

Numerics: fp32 state/accumulation regardless of input dtype.

Constraints:
    L % seg == 0
    D % 32  == 0       (threadgroup x is sized in multiples of 32)

Public API:
    rglru_scan(a, x, seg=32)             -> y      (note: x is the gated input b)
    rglru_scan_with_state(a, x, seg=32)  -> (y, final_state)
    rglru_scan_reference(a, x)           -> y      (pure MLX)
"""

from __future__ import annotations

import mlx.core as mx

from ._chassis import DEFAULT_SEG, get_or_build_kernel, check_segment_shape


# ---------------------------------------------------------------------------
# Metal forward: diagonal scan with segment checkpoints
# ---------------------------------------------------------------------------

def _rglru_forward_kernel(a, b, seg):
    """Forward diagonal scan writing only segment-boundary state.

    One thread per (batch, channel) owns the scalar ``h[d]`` across L steps.

    Args:
        a:   [B, L, D]   per-channel recurrent gate
        b:   [B, L, D]   per-channel (already-gated) input
        seg: segment length (L % seg == 0)

    Returns:
        y:      [B, L, D]            h at every step (output = post-update h)
        h_ckpt: [B, nSeg, D]         state at the END of each segment
    """
    B_batch, L, D = a.shape
    n_seg = L // seg

    source = f"""
        uint d = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        if (d >= {D}u || b >= {B_batch}u) return;

        float h = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            int idx = (b * {L} + t) * {D} + d;
            float a_val = (float)a_in[idx];
            float b_val = (float)b_in[idx];
            h = a_val * h + b_val;
            output[idx] = h;

            if (((t + 1) % {seg}) == 0) {{
                int s = t / {seg};
                int ck_idx = (b * {n_seg} + s) * {D} + d;
                h_ckpt[ck_idx] = h;
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"rglru_fwd_{B_batch}_{L}_{D}_{seg}",
        input_names=["a_in", "b_in"],
        output_names=["output", "h_ckpt"],
        source=source,
    )

    results = kernel(
        inputs=[a.reshape(-1), b.reshape(-1)],
        output_shapes=[
            (B_batch * L * D,),
            (B_batch * n_seg * D,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(D, B_batch, 1),
        threadgroup=(min(D, 256), 1, 1),
    )

    y = results[0].reshape(B_batch, L, D)
    h_ckpt = results[1].reshape(B_batch, n_seg, D)
    return y, h_ckpt


# ---------------------------------------------------------------------------
# Metal backward: segment recompute + reverse adjoint sweep
# ---------------------------------------------------------------------------

def _rglru_backward_kernel(grad_y, h_ckpt, a, b, seg):
    """Recompute states from checkpoints, then reverse adjoint sweep.

    Returns:
        grad_a: [B, L, D]
        grad_b: [B, L, D]
    """
    B_batch, L, D = a.shape
    n_seg = L // seg

    source = f"""
        uint d = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        if (d >= {D}u || b >= {B_batch}u) return;

        float adj = 0.0f;

        for (int s = {n_seg - 1}; s >= 0; s--) {{
            // ---- phase 1: recompute states for this segment ----
            float h_seg[{seg}];          // h at each step within the segment
            float h_start;               // state entering the segment (h_{{-1}})
            if (s > 0) {{
                int ck_idx = (b * {n_seg} + (s - 1)) * {D} + d;
                h_start = h_ckpt[ck_idx];
            }} else {{
                h_start = 0.0f;
            }}

            float h = h_start;
            for (int tl = 0; tl < {seg}; tl++) {{
                int t = s * {seg} + tl;
                int idx = (b * {L} + t) * {D} + d;
                h = (float)a_in[idx] * h + (float)b_in[idx];
                h_seg[tl] = h;
            }}

            // ---- phase 2: adjoint sweep, newest -> oldest ----
            for (int tl = {seg - 1}; tl >= 0; tl--) {{
                int t = s * {seg} + tl;
                int idx = (b * {L} + t) * {D} + d;

                float h_prev = (tl > 0) ? h_seg[tl - 1] : h_start;

                adj += (float)grad_y[idx];
                grad_b[idx] = adj;
                grad_a[idx] = adj * h_prev;

                // propagate to previous step: multiply by a_t
                adj *= (float)a_in[idx];
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"rglru_bwd_{B_batch}_{L}_{D}_{seg}",
        input_names=["grad_y", "h_ckpt", "a_in", "b_in"],
        output_names=["grad_a", "grad_b"],
        source=source,
    )

    results = kernel(
        inputs=[grad_y.reshape(-1), h_ckpt.reshape(-1), a.reshape(-1), b.reshape(-1)],
        output_shapes=[
            (B_batch * L * D,),
            (B_batch * L * D,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(D, B_batch, 1),
        threadgroup=(min(D, 256), 1, 1),
    )

    grad_a = results[0].reshape(B_batch, L, D)
    grad_b = results[1].reshape(B_batch, L, D)
    return grad_a, grad_b


# ---------------------------------------------------------------------------
# Custom function + VJP (one cached impl per seg, so seg can be a Python arg)
# ---------------------------------------------------------------------------

_impl_cache: dict = {}


def _make_impl(seg):
    """Build (and cache) an ``mx.custom_function`` RG-LRU impl bound to ``seg``."""
    if seg in _impl_cache:
        return _impl_cache[seg]

    @mx.custom_function
    def _impl(a, b):
        check_segment_shape(a.shape[1], seg, a.shape[2], "D")
        return _rglru_forward_kernel(a, b, seg)

    @_impl.vjp
    def _vjp(primals, cotangents, outputs):
        a, b = primals
        grad_y = cotangents[0]
        _y, h_ckpt = outputs
        return _rglru_backward_kernel(grad_y, h_ckpt, a, b, seg)

    _impl_cache[seg] = _impl
    return _impl


def rglru_scan(a, x, seg=DEFAULT_SEG):
    """RG-LRU diagonal linear scan, fused Metal forward + backward.

    Computes ``h_t = a_t (.) h_{t-1} + x_t`` and returns ``y_t = h_t`` for
    all ``t`` (carry starts at zero).

    Args:
        a:   [B, L, D]   per-channel recurrent gate
                         (e.g. exp(-c * softplus(Lambda) * sigmoid(r))).
        x:   [B, L, D]   the already-gated input ``b_t`` (e.g.
                         ``sqrt(1 - a^2) (.) (i (.) x)``). Named ``x`` to
                         match the requested public signature.
        seg: segment length for checkpointing (L % seg == 0; default 32)

    Returns:
        y:   [B, L, D]

    Note: ``D`` must be a multiple of 32. fp32 state/accumulation internally;
    bf16 inputs widen implicitly. Compute ``a`` and the gated input in pure
    MLX (cheap, auto-diff) and pass them here.
    """
    y, _h_ckpt = _make_impl(seg)(a, x)
    return y


def rglru_scan_with_state(a, x, seg=DEFAULT_SEG):
    """RG-LRU scan that also returns the final state for chunked prefill.

    Returns:
        y:           [B, L, D]
        final_state: [B, D]
    """
    y, h_ckpt = _make_impl(seg)(a, x)
    return y, h_ckpt[:, -1]


# ---------------------------------------------------------------------------
# Pure-MLX reference (slow, for parity testing only)
# ---------------------------------------------------------------------------

def rglru_scan_reference(a, x):
    """Pure-MLX token-loop reference for :func:`rglru_scan`. Differentiable."""
    B_batch, L, D = a.shape
    h = mx.zeros((B_batch, D))
    ys = []
    for t in range(L):
        h = a[:, t, :] * h + x[:, t, :]
        ys.append(h)
    return mx.stack(ys, axis=1)   # [B, L, D]
