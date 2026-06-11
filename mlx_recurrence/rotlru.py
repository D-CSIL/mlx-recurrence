"""rotlru.py — Rotational LRU diagonal recurrence (complex-diagonal scan).

NEW v2 plug-in. Generalizes :mod:`mlx_recurrence.rglru` from a real diagonal
gate to a complex one: each state channel pair ``(u, w)`` is treated as a
complex number that is scaled by a real magnitude gate ``a_t`` AND rotated by
an angle ``theta_t`` every step::

    [u]       [cos th  -sin th] [u]       [bu]
    [w]  = a *[sin th   cos th] [w]     + [bw]
       t            (R(theta))     t-1        t

i.e. ``h_t = a_t * e^{i*theta_t} * h_{t-1} + b_t`` in complex form. This is
the eigenvalue structure of the complex LRU (Orvieto et al., "Resurrecting
Recurrent Neural Networks") and of S4/Mamba-style complex poles: oscillatory
memory that is RELATIVE (depends on write-to-read distance), not tied to an
absolute clock.

The kernel takes ``cos(theta)`` and ``sin(theta)`` as separate per-step
inputs (``cs``, ``sn``) rather than computing trig in-kernel. Computing them
is cheap, elementwise, auto-differentiable MLX host code; gradients w.r.t.
the angle chain through ``d theta = -sn * grad_cs + cs * grad_sn``
automatically. The kernel itself is range-agnostic: ``a`` may be negative,
and ``(cs, sn)`` need not be normalized (it just applies the 2x2 matrix).

Layout: pairs are INTERLEAVED in the channel dim — ``x[..., 2p]`` is the
``u`` component and ``x[..., 2p+1]`` the ``w`` component of pair ``p``.
``a``/``cs``/``sn`` are indexed per pair (``Dp = D // 2``).

State is pair-diagonal: one ``(u, w)`` register pair per thread, no
cross-lane reductions (each pair's gradients are independent), so like
RG-LRU this plug-in needs no ``simd_sum``. It follows the chassis
segment-checkpoint + recompute backward pattern and supports chunked prefill
via a final-state variant.

VJP (per pair; ``M = R(theta)``, ``y_t = h_t``, zero initial carry):
    adj_t       = grad_y_t + a_{t+1} * M_{t+1}^T * adj_{t+1}
    grad_b[t]   = adj_t
    grad_a[t]   = adj_t . (M_t h_{t-1})
    grad_cs[t]  = a_t * (adjU*u_{t-1} + adjW*w_{t-1})
    grad_sn[t]  = a_t * (adjW*u_{t-1} - adjU*w_{t-1})

Numerics: fp32 state/accumulation regardless of input dtype.

Constraints:
    L % seg == 0
    D % 32  == 0   (D = 2*Dp interleaved channels; threadgroup sizing)

Public API:
    rotlru_scan(a, cs, sn, x, seg=32)             -> y
    rotlru_scan_with_state(a, cs, sn, x, seg=32)  -> (y, final_state)
    rotlru_scan_reference(a, cs, sn, x)           -> y   (pure MLX)
"""

from __future__ import annotations

import mlx.core as mx

from ._chassis import DEFAULT_SEG, get_or_build_kernel, check_segment_shape


def _validate(a, cs, sn, x):
    if a.ndim != 3:
        raise ValueError(f"a must be [B, L, Dp], got shape {a.shape}")
    if a.shape != cs.shape or a.shape != sn.shape:
        raise ValueError(
            f"a/cs/sn shapes must match, got {a.shape}, {cs.shape}, {sn.shape}"
        )
    B, L, Dp = a.shape
    if x.shape != (B, L, 2 * Dp):
        raise ValueError(
            f"x must be [B, L, 2*Dp] = [{B}, {L}, {2*Dp}], got {x.shape}"
        )
    return B, L, Dp


# ---------------------------------------------------------------------------
# Metal forward: pair-diagonal rotational scan with segment checkpoints
# ---------------------------------------------------------------------------

def _rotlru_forward_kernel(a, cs, sn, x, seg):
    """Forward scan writing only segment-boundary state.

    One thread per (batch, pair) owns the (u, w) register pair across L steps.

    Returns:
        y:      [B, L, D]        post-update state at every step (D = 2*Dp)
        h_ckpt: [B, nSeg, D]     state at the END of each segment
    """
    B_batch, L, Dp = a.shape
    D = 2 * Dp
    n_seg = L // seg

    source = f"""
        uint p = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        if (p >= {Dp}u || b >= {B_batch}u) return;

        float u = 0.0f;
        float w = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            int ip = (b * {L} + t) * {Dp} + p;        // a / cs / sn index
            int ix = (b * {L} + t) * {D} + 2 * p;     // interleaved x / y index
            float av = (float)a_in[ip];
            float c  = (float)cs_in[ip];
            float s  = (float)sn_in[ip];
            float bu = (float)x_in[ix];
            float bw = (float)x_in[ix + 1];

            float un = av * (c * u - s * w) + bu;
            float wn = av * (s * u + c * w) + bw;
            u = un;
            w = wn;
            output[ix]     = u;
            output[ix + 1] = w;

            if (((t + 1) % {seg}) == 0) {{
                int sgi = t / {seg};
                int ck = (b * {n_seg} + sgi) * {D} + 2 * p;
                h_ckpt[ck]     = u;
                h_ckpt[ck + 1] = w;
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"rotlru_fwd_{B_batch}_{L}_{Dp}_{seg}",
        input_names=["a_in", "cs_in", "sn_in", "x_in"],
        output_names=["output", "h_ckpt"],
        source=source,
    )

    results = kernel(
        inputs=[a.reshape(-1), cs.reshape(-1), sn.reshape(-1), x.reshape(-1)],
        output_shapes=[
            (B_batch * L * D,),
            (B_batch * n_seg * D,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(Dp, B_batch, 1),
        threadgroup=(min(Dp, 256), 1, 1),
    )

    y = results[0].reshape(B_batch, L, D)
    h_ckpt = results[1].reshape(B_batch, n_seg, D)
    return y, h_ckpt


# ---------------------------------------------------------------------------
# Metal backward: segment recompute + reverse adjoint sweep
# ---------------------------------------------------------------------------

def _rotlru_backward_kernel(grad_y, h_ckpt, a, cs, sn, x, seg):
    """Recompute states from checkpoints, then reverse adjoint sweep.

    Returns:
        grad_a:  [B, L, Dp]
        grad_cs: [B, L, Dp]
        grad_sn: [B, L, Dp]
        grad_x:  [B, L, D]
    """
    B_batch, L, Dp = a.shape
    D = 2 * Dp
    n_seg = L // seg

    source = f"""
        uint p = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        if (p >= {Dp}u || b >= {B_batch}u) return;

        float adjU = 0.0f;
        float adjW = 0.0f;

        for (int sgi = {n_seg - 1}; sgi >= 0; sgi--) {{
            // ---- phase 1: recompute this segment's states ----
            float u_seg[{seg}];
            float w_seg[{seg}];
            float u_start, w_start;
            if (sgi > 0) {{
                int ck = (b * {n_seg} + (sgi - 1)) * {D} + 2 * p;
                u_start = h_ckpt[ck];
                w_start = h_ckpt[ck + 1];
            }} else {{
                u_start = 0.0f;
                w_start = 0.0f;
            }}

            float u = u_start;
            float w = w_start;
            for (int tl = 0; tl < {seg}; tl++) {{
                int t = sgi * {seg} + tl;
                int ip = (b * {L} + t) * {Dp} + p;
                int ix = (b * {L} + t) * {D} + 2 * p;
                float av = (float)a_in[ip];
                float c  = (float)cs_in[ip];
                float s  = (float)sn_in[ip];
                float un = av * (c * u - s * w) + (float)x_in[ix];
                float wn = av * (s * u + c * w) + (float)x_in[ix + 1];
                u = un;
                w = wn;
                u_seg[tl] = u;
                w_seg[tl] = w;
            }}

            // ---- phase 2: adjoint sweep, newest -> oldest ----
            for (int tl = {seg - 1}; tl >= 0; tl--) {{
                int t = sgi * {seg} + tl;
                int ip = (b * {L} + t) * {Dp} + p;
                int ix = (b * {L} + t) * {D} + 2 * p;
                float av = (float)a_in[ip];
                float c  = (float)cs_in[ip];
                float s  = (float)sn_in[ip];

                float u_prev = (tl > 0) ? u_seg[tl - 1] : u_start;
                float w_prev = (tl > 0) ? w_seg[tl - 1] : w_start;

                adjU += (float)grad_y[ix];
                adjW += (float)grad_y[ix + 1];

                grad_x[ix]     = adjU;
                grad_x[ix + 1] = adjW;

                // rotated previous state: M_t h_{{t-1}}
                float ru = c * u_prev - s * w_prev;
                float rw = s * u_prev + c * w_prev;
                grad_a[ip]  = adjU * ru + adjW * rw;
                grad_cs[ip] = av * (adjU * u_prev + adjW * w_prev);
                grad_sn[ip] = av * (adjW * u_prev - adjU * w_prev);

                // pull back: adj_{{t-1}} = a_t * R(theta)^T adj_t
                float nU = av * ( c * adjU + s * adjW);
                float nW = av * (-s * adjU + c * adjW);
                adjU = nU;
                adjW = nW;
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"rotlru_bwd_{B_batch}_{L}_{Dp}_{seg}",
        input_names=["grad_y", "h_ckpt", "a_in", "cs_in", "sn_in", "x_in"],
        output_names=["grad_a", "grad_cs", "grad_sn", "grad_x"],
        source=source,
    )

    results = kernel(
        inputs=[
            grad_y.reshape(-1), h_ckpt.reshape(-1),
            a.reshape(-1), cs.reshape(-1), sn.reshape(-1), x.reshape(-1),
        ],
        output_shapes=[
            (B_batch * L * Dp,),
            (B_batch * L * Dp,),
            (B_batch * L * Dp,),
            (B_batch * L * D,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        grid=(Dp, B_batch, 1),
        threadgroup=(min(Dp, 256), 1, 1),
    )

    grad_a = results[0].reshape(B_batch, L, Dp)
    grad_cs = results[1].reshape(B_batch, L, Dp)
    grad_sn = results[2].reshape(B_batch, L, Dp)
    grad_x = results[3].reshape(B_batch, L, D)
    return grad_a, grad_cs, grad_sn, grad_x


# ---------------------------------------------------------------------------
# Custom function + VJP (one cached impl per seg)
# ---------------------------------------------------------------------------

_impl_cache: dict = {}


def _make_impl(seg):
    if seg in _impl_cache:
        return _impl_cache[seg]

    @mx.custom_function
    def _impl(a, cs, sn, x):
        _validate(a, cs, sn, x)
        check_segment_shape(a.shape[1], seg, x.shape[2], "D")
        return _rotlru_forward_kernel(a, cs, sn, x, seg)

    @_impl.vjp
    def _vjp(primals, cotangents, outputs):
        a, cs, sn, x = primals
        grad_y = cotangents[0]
        _y, h_ckpt = outputs
        return _rotlru_backward_kernel(grad_y, h_ckpt, a, cs, sn, x, seg)

    _impl_cache[seg] = _impl
    return _impl


def rotlru_scan(a, cs, sn, x, seg=DEFAULT_SEG):
    """Rotational LRU pair-diagonal scan, fused Metal forward + backward.

    Computes ``h_t = a_t * R(theta_t) h_{t-1} + x_t`` over interleaved
    ``(u, w)`` channel pairs and returns ``y_t = h_t`` for all ``t``
    (carry starts at zero).

    Args:
        a:   [B, L, Dp]  per-pair magnitude gate (any real value).
        cs:  [B, L, Dp]  cos(theta_t) per pair (host-computed).
        sn:  [B, L, Dp]  sin(theta_t) per pair (host-computed).
        x:   [B, L, D]   drive, D = 2*Dp, pairs interleaved (u, w).
        seg: segment length for checkpointing (L % seg == 0; default 32).

    Returns:
        y:   [B, L, D]

    ``D`` must be a multiple of 32. fp32 state/accumulation internally; bf16
    inputs widen implicitly. Gradients flow to all four inputs; an angle
    parameter chains through cs/sn automatically when they are computed as
    ``mx.cos(theta)`` / ``mx.sin(theta)`` in MLX host code.
    """
    y, _h_ckpt = _make_impl(seg)(a, cs, sn, x)
    return y


def rotlru_scan_with_state(a, cs, sn, x, seg=DEFAULT_SEG):
    """Rotational LRU scan that also returns the final state for prefill.

    Returns:
        y:           [B, L, D]
        final_state: [B, D]
    """
    y, h_ckpt = _make_impl(seg)(a, cs, sn, x)
    return y, h_ckpt[:, -1]


# ---------------------------------------------------------------------------
# Pure-MLX reference (slow, for parity testing only)
# ---------------------------------------------------------------------------

def rotlru_scan_reference(a, cs, sn, x):
    """Pure-MLX token-loop reference for :func:`rotlru_scan`. Differentiable."""
    B_batch, L, Dp = _validate(a, cs, sn, x)
    xp = x.reshape(B_batch, L, Dp, 2)
    u = mx.zeros((B_batch, Dp))
    w = mx.zeros((B_batch, Dp))
    ys = []
    for t in range(L):
        at, c, s = a[:, t, :], cs[:, t, :], sn[:, t, :]
        un = at * (c * u - s * w) + xp[:, t, :, 0]
        wn = at * (s * u + c * w) + xp[:, t, :, 1]
        u, w = un, wn
        ys.append(mx.stack([u, w], axis=-1).reshape(B_batch, 2 * Dp))
    return mx.stack(ys, axis=1)   # [B, L, D]
