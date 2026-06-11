"""gla.py — Gated Linear Attention recurrence (checkpoint + recompute).

v2 chassis port of the GLA recurrence kernel. Same segment-checkpoint +
recompute backward pattern as :mod:`mlx_recurrence.ssd`, applied to the
matrix-valued GLA state.

Recurrence
----------
Per head ``head`` the state is the ``Dh x Dh`` matrix ``h[b, head, i, j]``
with a single scalar forget gate per token::

    h[i, j]       = gate[b,t,head] * h[i, j] + k[b,t,head,i] * v[b,t,head,j]
    o[b,t,head,j] = sum_i q[b,t,head,i] * h[i, j]

i.e. ``h_t = gate_t * h_{t-1} + k_t (outer) v_t`` and ``o_t = q_t @ h_t``
(output uses the post-update state). ``gate`` is typically a sigmoid output
in ``(0, 1)``; ``q`` is assumed pre-scaled / post-RoPE.

The conceptual state tensor is ``[B, H, Dh, Dh]``. The checkpoint is laid
out ``[B, nSeg, H, Dh, Dh]`` with ``j`` fastest-moving so the 32 lanes of a
simdgroup own 32 contiguous ``j`` columns — coalesced reads/writes.

Numerics: fp32 state and accumulation regardless of input dtype; identical
update order (output uses post-update ``h``) and gradient formulas to the
validated D-CSIL-3 training kernel. grad_v is exact per-thread; grad_q,
grad_k, grad_gates are fused in-kernel via 32-lane ``simd_sum``.

Constraints:
    L  % seg == 0
    Dh % 32  == 0       (Dh is the simd-reduced lane dimension)

Public API:
    gla_scan(q, k, v, gates, seg=32)             -> y
    gla_scan_with_state(q, k, v, gates, seg=32)  -> (y, final_state)
    gla_scan_reference(q, k, v, gates)           -> y   (pure MLX)
"""

from __future__ import annotations

import mlx.core as mx

from ._chassis import DEFAULT_SEG, get_or_build_kernel, check_segment_shape


# ---------------------------------------------------------------------------
# Metal forward: GLA scan with segment checkpoints (no full state history)
# ---------------------------------------------------------------------------

def _gla_forward_kernel(q, k, v, gates, seg):
    """Forward GLA scan writing only segment-boundary state.

    One thread per (batch*head, j) owns the ``h[:, j]`` state column in
    registers across all L timesteps.

    Args:
        q, k, v: [B, L, H, Dh]
        gates:   [B, L, H]
        seg:     segment length (L % seg == 0)

    Returns:
        y:      [B, L, H, Dh]
        h_ckpt: [B, nSeg, H, Dh, Dh]  state at the END of each segment
    """
    B_batch, L, H, Dh = q.shape
    n_seg = L // seg

    source = f"""
        uint j  = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;
        if (j >= {Dh}u || bh >= {B_batch * H}u) return;

        uint b    = bh / {H}u;
        uint head = bh % {H}u;

        float h[{Dh}];
        for (int i = 0; i < {Dh}; i++) h[i] = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            int g_idx   = (b * {L} + t) * {H} + head;
            int kv_base = ((b * {L} + t) * {H} + head) * {Dh};

            float gate = (float)gates[g_idx];
            float v_j  = (float)v[kv_base + j];
            float o_j  = 0.0f;

            for (int i = 0; i < {Dh}; i++) {{
                h[i] = gate * h[i] + (float)k[kv_base + i] * v_j;
                o_j += (float)q[kv_base + i] * h[i];
            }}
            output[kv_base + j] = o_j;

            // checkpoint at end of each segment (j fastest -> coalesced)
            if (((t + 1) % {seg}) == 0) {{
                int s = t / {seg};
                for (int i = 0; i < {Dh}; i++) {{
                    int ck_idx = ((((b * {n_seg} + s) * {H} + head) * {Dh} + i) * {Dh} + j);
                    h_ckpt[ck_idx] = h[i];
                }}
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"gla_fwd_{B_batch}_{L}_{H}_{Dh}_{seg}",
        input_names=["q", "k", "v", "gates"],
        output_names=["output", "h_ckpt"],
        source=source,
    )

    results = kernel(
        inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1), gates.reshape(-1)],
        output_shapes=[
            (B_batch * L * H * Dh,),
            (B_batch * n_seg * H * Dh * Dh,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(Dh, B_batch * H, 1),
        threadgroup=(min(Dh, 256), 1, 1),
    )

    y = results[0].reshape(B_batch, L, H, Dh)
    h_ckpt = results[1].reshape(B_batch, n_seg, H, Dh, Dh)
    return y, h_ckpt


# ---------------------------------------------------------------------------
# Metal backward: segment recompute + fused simd-reduced gradient partials
# ---------------------------------------------------------------------------

def _gla_backward_kernel(grad_y, h_ckpt, q, k, v, gates, seg):
    """Recompute states from checkpoints, then adjoint sweep with fused reductions.

    Returns:
        grad_v:   [B, L, H, Dh]       (exact, per-thread)
        grad_q_p: [B, L, H, nW, Dh]   (sum over nW -> grad_q)
        grad_k_p: [B, L, H, nW, Dh]   (sum over nW -> grad_k)
        grad_g_p: [B, L, H, nW]       (sum over nW -> grad_gates)
    """
    B_batch, L, H, Dh = q.shape
    n_seg = L // seg
    n_w = Dh // 32  # simdgroups (j-lane groups) per head

    source = f"""
        uint j  = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;
        if (j >= {Dh}u || bh >= {B_batch * H}u) return;

        uint b    = bh / {H}u;
        uint head = bh % {H}u;
        // threadgroup x is a multiple of 32 and x-major, so lanes of a
        // simdgroup are 32 consecutive j values within one head.
        uint lane = j % 32u;
        uint w    = j / 32u;

        float adj[{Dh}];
        for (int i = 0; i < {Dh}; i++) adj[i] = 0.0f;

        for (int s = {n_seg - 1}; s >= 0; s--) {{
            // ---- phase 1: recompute states for this segment ----
            float h[{Dh}];
            if (s > 0) {{
                for (int i = 0; i < {Dh}; i++) {{
                    int ck_idx = ((((b * {n_seg} + (s - 1)) * {H} + head) * {Dh} + i) * {Dh} + j);
                    h[i] = h_ckpt[ck_idx];
                }}
            }} else {{
                for (int i = 0; i < {Dh}; i++) h[i] = 0.0f;
            }}

            for (int tl = 0; tl < {seg}; tl++) {{
                int t = s * {seg} + tl;
                int g_idx   = (b * {L} + t) * {H} + head;
                int kv_base = ((b * {L} + t) * {H} + head) * {Dh};
                float gate = (float)gates[g_idx];
                float v_j  = (float)v[kv_base + j];

                for (int i = 0; i < {Dh}; i++) {{
                    h[i] = gate * h[i] + (float)k[kv_base + i] * v_j;
                    int sc_idx = ((((b * {H} + head) * {seg} + tl) * {Dh} + i) * {Dh} + j);
                    scratch[sc_idx] = h[i];
                }}
            }}

            // ---- phase 2: adjoint sweep, newest -> oldest ----
            for (int tl = {seg - 1}; tl >= 0; tl--) {{
                int t = s * {seg} + tl;
                int g_idx   = (b * {L} + t) * {H} + head;
                int kv_base = ((b * {L} + t) * {H} + head) * {Dh};

                float gate = (float)gates[g_idx];
                float v_j  = (float)v[kv_base + j];
                float go_j = (float)grad_y[kv_base + j];

                float gv_j = 0.0f;
                float gg   = 0.0f;

                for (int i = 0; i < {Dh}; i++) {{
                    float ki = (float)k[kv_base + i];

                    int sc_idx = ((((b * {H} + head) * {seg} + tl) * {Dh} + i) * {Dh} + j);
                    float h_cur = scratch[sc_idx];
                    float h_prev;
                    if (tl > 0) {{
                        h_prev = scratch[sc_idx - {Dh * Dh}];
                    }} else if (s > 0) {{
                        int ck_idx = ((((b * {n_seg} + (s - 1)) * {H} + head) * {Dh} + i) * {Dh} + j);
                        h_prev = h_ckpt[ck_idx];
                    }} else {{
                        h_prev = 0.0f;
                    }}

                    // driving term (adj sampled after this, before gate multiply)
                    adj[i] += (float)q[kv_base + i] * go_j;

                    gv_j += adj[i] * ki;
                    gg   += adj[i] * h_prev;

                    // fused reductions over the 32 j-lanes of this simdgroup
                    float gq_l = simd_sum(go_j * h_cur);
                    float gk_l = simd_sum(adj[i] * v_j);
                    if (lane == 0u) {{
                        int p_idx = ((((b * {L} + t) * {H} + head) * {n_w} + w) * {Dh} + i);
                        grad_q_p[p_idx] = gq_l;
                        grad_k_p[p_idx] = gk_l;
                    }}

                    adj[i] *= gate;
                }}

                grad_v[kv_base + j] = gv_j;
                float gg_l = simd_sum(gg);
                if (lane == 0u) {{
                    grad_g_p[(((b * {L} + t) * {H} + head) * {n_w} + w)] = gg_l;
                }}
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"gla_bwd_{B_batch}_{L}_{H}_{Dh}_{seg}",
        input_names=["grad_y", "h_ckpt", "q", "k", "v", "gates"],
        output_names=["grad_v", "grad_q_p", "grad_k_p", "grad_g_p", "scratch"],
        source=source,
    )

    results = kernel(
        inputs=[grad_y.reshape(-1), h_ckpt.reshape(-1), q.reshape(-1),
                k.reshape(-1), v.reshape(-1), gates.reshape(-1)],
        output_shapes=[
            (B_batch * L * H * Dh,),
            (B_batch * L * H * n_w * Dh,),
            (B_batch * L * H * n_w * Dh,),
            (B_batch * L * H * n_w,),
            (B_batch * H * seg * Dh * Dh,),   # scratch, discarded
        ],
        output_dtypes=[mx.float32] * 5,
        grid=(Dh, B_batch * H, 1),
        threadgroup=(min(Dh, 256), 1, 1),
    )

    grad_v   = results[0].reshape(B_batch, L, H, Dh)
    grad_q_p = results[1].reshape(B_batch, L, H, n_w, Dh)
    grad_k_p = results[2].reshape(B_batch, L, H, n_w, Dh)
    grad_g_p = results[3].reshape(B_batch, L, H, n_w)

    grad_q     = mx.sum(grad_q_p, axis=3)   # [B, L, H, Dh]
    grad_k     = mx.sum(grad_k_p, axis=3)   # [B, L, H, Dh]
    grad_gates = mx.sum(grad_g_p, axis=3)   # [B, L, H]

    return grad_q, grad_k, grad_v, grad_gates


# ---------------------------------------------------------------------------
# Custom function + VJP (one cached impl per seg, so seg can be a Python arg)
# ---------------------------------------------------------------------------

_impl_cache: dict = {}


def _make_impl(seg):
    """Build (and cache) an ``mx.custom_function`` GLA impl bound to ``seg``."""
    if seg in _impl_cache:
        return _impl_cache[seg]

    @mx.custom_function
    def _impl(q, k, v, gates):
        check_segment_shape(q.shape[1], seg, q.shape[3], "Dh")
        return _gla_forward_kernel(q, k, v, gates, seg)

    @_impl.vjp
    def _vjp(primals, cotangents, outputs):
        q, k, v, gates = primals
        grad_y = cotangents[0]
        _y, h_ckpt = outputs
        return _gla_backward_kernel(grad_y, h_ckpt, q, k, v, gates, seg)

    _impl_cache[seg] = _impl
    return _impl


def gla_scan(q, k, v, gates, seg=DEFAULT_SEG):
    """Gated Linear Attention recurrence, fused Metal forward + backward.

    Args:
        q:      [B, L, H, Dh]   queries (pre-scaled, post-RoPE)
        k:      [B, L, H, Dh]   keys
        v:      [B, L, H, Dh]   values
        gates:  [B, L, H]       scalar forget gate per head, typically in (0, 1)
        seg:    segment length for checkpointing (L % seg == 0; default 32)

    Returns:
        y:      [B, L, H, Dh]

    Note: ``Dh`` must be a multiple of 32 (it is the simd-reduced lane dim).
    fp32 state/accumulation internally; bf16 inputs widen implicitly.
    """
    y, _h_ckpt = _make_impl(seg)(q, k, v, gates)
    return y


def gla_scan_with_state(q, k, v, gates, seg=DEFAULT_SEG):
    """GLA scan that also returns the final state for chunked prefill.

    Returns:
        y:           [B, L, H, Dh]
        final_state: [B, H, Dh, Dh]   (matches the GLA conceptual state)
    """
    y, h_ckpt = _make_impl(seg)(q, k, v, gates)
    return y, h_ckpt[:, -1]


# ---------------------------------------------------------------------------
# Pure-MLX reference (slow, for parity testing only)
# ---------------------------------------------------------------------------

def gla_scan_reference(q, k, v, gates):
    """Pure-MLX token-loop reference for :func:`gla_scan`. Differentiable."""
    B_batch, L, H, Dh = q.shape
    h = mx.zeros((B_batch, H, Dh, Dh))
    ys = []
    for t in range(L):
        g = gates[:, t, :, None, None]                       # [B,H,1,1]
        kv = k[:, t, :, :, None] * v[:, t, :, None, :]       # [B,H,Dh,Dh]
        h = g * h + kv
        ys.append(mx.sum(q[:, t, :, :, None] * h, axis=-2))  # [B,H,Dh]
    return mx.stack(ys, axis=1)                               # [B,L,H,Dh]
