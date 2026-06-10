"""ssd.py — Mamba-2-style head-wise SSD selective scan (checkpoint + recompute).

This is the v2 chassis port of the head-wise State-Space Duality (SSD) scan.
It uses the segment-checkpoint + recompute backward pattern documented in
:mod:`mlx_recurrence._chassis`: the forward writes only segment-boundary
state, and the backward recomputes each segment into a small scratch buffer
before running the adjoint sweep, with grad_B / grad_C / grad_delta
reductions fused in-kernel via 32-lane ``simd_sum``.

Mamba-2 correspondence
----------------------
For each head ``head`` and channel ``d`` the per-head SSM state is
``h[b, head, d, n]`` with ``N`` state components. The recurrence is::

    decay      = exp(delta[b,t,head] * A_neg[head,n])       # scalar per (head,n)
    h[n]       = decay * h[n] + delta[b,t,head] * B[b,t,head,n] * u[b,t,head,d]
    y[b,t,head,d] = sum_n h[n] * C[b,t,head,n]

This is the diagonal (per-head, scalar-A) SSD form used by Mamba-2: ``A`` is
a per-(head, state) decay rate (``A_neg = -exp(A_log)``, strictly negative),
``B`` and ``C`` are the input/output projections shared across the ``Dh``
channels of a head, and ``delta`` is the per-token, per-head step size.

The conceptual state tensor is ``[B, H, Dh, N]`` (Dh channels x N state
components per head). Internally the checkpoint is laid out
``[B, nSeg, H, N, Dh]`` so the fastest-moving axis (``Dh``) is contiguous
across the 32 lanes of a simdgroup — coalesced reads/writes.

Numerics: fp32 state and accumulation regardless of input dtype; bf16
inputs are read with implicit widening. Identical update order and gradient
formulas to the validated D-CSIL-3 training kernel.

Constraints (see :func:`mlx_recurrence._chassis.check_segment_shape`):
    L  % seg == 0
    Dh % 32  == 0       (Dh is the simd-reduced lane dimension)

Public API:
    ssd_scan(u, delta, B, C, A_neg, seg=32)              -> y
    ssd_scan_with_state(u, delta, B, C, A_neg, seg=32)   -> (y, final_state)
    ssd_scan_reference(u, delta, B, C, A_neg)            -> y   (pure MLX)
"""

from __future__ import annotations

import mlx.core as mx

from ._chassis import DEFAULT_SEG, get_or_build_kernel, check_segment_shape


# ---------------------------------------------------------------------------
# Metal forward: scan with segment checkpoints (no full state history)
# ---------------------------------------------------------------------------

def _ssd_forward_kernel(u, delta, B_in, C_in, A_neg, seg):
    """Forward scan writing only segment-boundary state.

    Args:
        u:      [B, L, H, Dh]
        delta:  [B, L, H]
        B_in:   [B, L, H, N]
        C_in:   [B, L, H, N]
        A_neg:  [H, N]   (negative; A = -exp(A_log))
        seg:    segment length (L % seg == 0)

    Returns:
        y:      [B, L, H, Dh]
        h_ckpt: [B, nSeg, H, N, Dh]  state at the END of each segment
    """
    B_batch, L, H, Dh = u.shape
    N = A_neg.shape[-1]
    inner_dim = H * Dh
    n_seg = L // seg

    source = f"""
        uint hd = thread_position_in_grid.x;
        uint b  = thread_position_in_grid.y;
        if (hd >= {inner_dim}u || b >= {B_batch}u) return;

        uint head = hd / {Dh}u;
        uint d    = hd - head * {Dh}u;

        float h[{N}];
        for (int n = 0; n < {N}; n++) h[n] = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            int u_idx  = (((b * {L} + t) * {H} + head) * {Dh} + d);
            int dt_idx = ((b * {L} + t) * {H} + head);

            float dt_val = (float)delta[dt_idx];
            float u_val  = (float)u[u_idx];
            float y_val  = 0.0f;

            for (int n = 0; n < {N}; n++) {{
                int hn_idx = head * {N} + n;
                int bc_idx = (((b * {L} + t) * {H} + head) * {N} + n);

                float decay = exp(dt_val * (float)A_neg[hn_idx]);
                h[n] = decay * h[n] + dt_val * (float)B_in[bc_idx] * u_val;
                y_val += h[n] * (float)C_in[bc_idx];
            }}
            output[u_idx] = y_val;

            // checkpoint at end of each segment (coalesced: d fastest)
            if (((t + 1) % {seg}) == 0) {{
                int s = t / {seg};
                for (int n = 0; n < {N}; n++) {{
                    int ck_idx = ((((b * {n_seg} + s) * {H} + head) * {N} + n) * {Dh} + d);
                    h_ckpt[ck_idx] = h[n];
                }}
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"ssd_fwd_{B_batch}_{L}_{H}_{Dh}_{N}_{seg}",
        input_names=["u", "delta", "B_in", "C_in", "A_neg"],
        output_names=["output", "h_ckpt"],
        source=source,
    )

    results = kernel(
        inputs=[u.reshape(-1), delta.reshape(-1), B_in.reshape(-1),
                C_in.reshape(-1), A_neg.reshape(-1)],
        output_shapes=[
            (B_batch * L * H * Dh,),
            (B_batch * n_seg * H * N * Dh,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(inner_dim, B_batch, 1),
        threadgroup=(min(inner_dim, 256), 1, 1),
    )

    y = results[0].reshape(B_batch, L, H, Dh)
    h_ckpt = results[1].reshape(B_batch, n_seg, H, N, Dh)
    return y, h_ckpt


# ---------------------------------------------------------------------------
# Metal backward: segment recompute + fused simd-reduced gradient partials
# ---------------------------------------------------------------------------

def _ssd_backward_kernel(grad_y, h_ckpt, u, delta, B_in, C_in, A_neg, seg):
    """Recompute states from checkpoints, then adjoint sweep with fused reductions.

    Returns:
        grad_u:    [B, L, H, Dh]
        grad_dt_p: [B, L, H, nW]        (sum over nW -> grad_delta)
        grad_B_p:  [B, L, H, nW, N]     (sum over nW -> grad_B)
        grad_C_p:  [B, L, H, nW, N]     (sum over nW -> grad_C)
        grad_A_p:  [B, H, nW, N]        (sum over B, nW -> grad_A_neg)
    """
    B_batch, L, H, Dh = u.shape
    N = A_neg.shape[-1]
    inner_dim = H * Dh
    n_seg = L // seg
    n_w = Dh // 32  # simdgroups per head

    source = f"""
        uint hd = thread_position_in_grid.x;
        uint b  = thread_position_in_grid.y;
        if (hd >= {inner_dim}u || b >= {B_batch}u) return;

        uint head = hd / {Dh}u;
        uint d    = hd - head * {Dh}u;
        // threadgroup x is a multiple of 32 and x-major, so lanes of a
        // simdgroup are 32 consecutive d values within one head.
        uint lane = hd % 32u;
        uint w    = d / 32u;

        float adj[{N}];
        float gA[{N}];
        for (int n = 0; n < {N}; n++) {{ adj[n] = 0.0f; gA[n] = 0.0f; }}

        for (int s = {n_seg - 1}; s >= 0; s--) {{
            // ---- phase 1: recompute states for this segment ----
            float h[{N}];
            if (s > 0) {{
                for (int n = 0; n < {N}; n++) {{
                    int ck_idx = ((((b * {n_seg} + (s - 1)) * {H} + head) * {N} + n) * {Dh} + d);
                    h[n] = h_ckpt[ck_idx];
                }}
            }} else {{
                for (int n = 0; n < {N}; n++) h[n] = 0.0f;
            }}

            for (int tl = 0; tl < {seg}; tl++) {{
                int t = s * {seg} + tl;
                int u_idx  = (((b * {L} + t) * {H} + head) * {Dh} + d);
                int dt_idx = ((b * {L} + t) * {H} + head);
                float dt_val = (float)delta[dt_idx];
                float u_val  = (float)u[u_idx];

                for (int n = 0; n < {N}; n++) {{
                    int hn_idx = head * {N} + n;
                    int bc_idx = (((b * {L} + t) * {H} + head) * {N} + n);
                    float decay = exp(dt_val * (float)A_neg[hn_idx]);
                    h[n] = decay * h[n] + dt_val * (float)B_in[bc_idx] * u_val;
                    int sc_idx = ((((b * {H} + head) * {seg} + tl) * {N} + n) * {Dh} + d);
                    scratch[sc_idx] = h[n];
                }}
            }}

            // ---- phase 2: adjoint sweep, newest -> oldest ----
            for (int tl = {seg - 1}; tl >= 0; tl--) {{
                int t = s * {seg} + tl;
                int u_idx  = (((b * {L} + t) * {H} + head) * {Dh} + d);
                int dt_idx = ((b * {L} + t) * {H} + head);

                float dt_val = (float)delta[dt_idx];
                float u_val  = (float)u[u_idx];
                float gy     = (float)grad_y[u_idx];

                float gu  = 0.0f;
                float gdt = 0.0f;

                for (int n = 0; n < {N}; n++) {{
                    int hn_idx = head * {N} + n;
                    int bc_idx = (((b * {L} + t) * {H} + head) * {N} + n);

                    float a  = (float)A_neg[hn_idx];
                    float bv = (float)B_in[bc_idx];
                    float cv = (float)C_in[bc_idx];
                    float decay = exp(dt_val * a);

                    int sc_idx = ((((b * {H} + head) * {seg} + tl) * {N} + n) * {Dh} + d);
                    float h_cur = scratch[sc_idx];
                    float h_prev;
                    if (tl > 0) {{
                        h_prev = scratch[sc_idx - {N * Dh}];
                    }} else if (s > 0) {{
                        int ck_idx = ((((b * {n_seg} + (s - 1)) * {H} + head) * {N} + n) * {Dh} + d);
                        h_prev = h_ckpt[ck_idx];
                    }} else {{
                        h_prev = 0.0f;
                    }}

                    adj[n] += gy * cv;

                    gu    += adj[n] * dt_val * bv;
                    gdt   += adj[n] * (a * decay * h_prev + bv * u_val);
                    gA[n] += adj[n] * dt_val * decay * h_prev;

                    // fused reductions over the 32 d-lanes of this simdgroup
                    float gB_l = simd_sum(adj[n] * dt_val * u_val);
                    float gC_l = simd_sum(h_cur * gy);
                    if (lane == 0u) {{
                        int p_idx = ((((b * {L} + t) * {H} + head) * {n_w} + w) * {N} + n);
                        grad_B_p[p_idx] = gB_l;
                        grad_C_p[p_idx] = gC_l;
                    }}

                    adj[n] *= decay;
                }}

                grad_u[u_idx] = gu;
                float gdt_l = simd_sum(gdt);
                if (lane == 0u) {{
                    grad_dt_p[(((b * {L} + t) * {H} + head) * {n_w} + w)] = gdt_l;
                }}
            }}
        }}

        for (int n = 0; n < {N}; n++) {{
            float gA_l = simd_sum(gA[n]);
            if (lane == 0u) {{
                grad_A_p[(((b * {H} + head) * {n_w} + w) * {N} + n)] = gA_l;
            }}
        }}
    """

    kernel = get_or_build_kernel(
        f"ssd_bwd_{B_batch}_{L}_{H}_{Dh}_{N}_{seg}",
        input_names=["grad_y", "h_ckpt", "u", "delta", "B_in", "C_in", "A_neg"],
        output_names=["grad_u", "grad_dt_p", "grad_B_p", "grad_C_p",
                      "grad_A_p", "scratch"],
        source=source,
    )

    results = kernel(
        inputs=[grad_y.reshape(-1), h_ckpt.reshape(-1), u.reshape(-1),
                delta.reshape(-1), B_in.reshape(-1), C_in.reshape(-1),
                A_neg.reshape(-1)],
        output_shapes=[
            (B_batch * L * H * Dh,),
            (B_batch * L * H * n_w,),
            (B_batch * L * H * n_w * N,),
            (B_batch * L * H * n_w * N,),
            (B_batch * H * n_w * N,),
            (B_batch * H * seg * N * Dh,),   # scratch, discarded
        ],
        output_dtypes=[mx.float32] * 6,
        grid=(inner_dim, B_batch, 1),
        threadgroup=(min(inner_dim, 256), 1, 1),
    )

    grad_u    = results[0].reshape(B_batch, L, H, Dh)
    grad_dt_p = results[1].reshape(B_batch, L, H, n_w)
    grad_B_p  = results[2].reshape(B_batch, L, H, n_w, N)
    grad_C_p  = results[3].reshape(B_batch, L, H, n_w, N)
    grad_A_p  = results[4].reshape(B_batch, H, n_w, N)

    grad_delta = mx.sum(grad_dt_p, axis=3)        # [B, L, H]
    grad_B     = mx.sum(grad_B_p, axis=3)         # [B, L, H, N]
    grad_C     = mx.sum(grad_C_p, axis=3)         # [B, L, H, N]
    grad_A     = mx.sum(grad_A_p, axis=(0, 2))    # [H, N]

    return grad_u, grad_delta, grad_B, grad_C, grad_A


# ---------------------------------------------------------------------------
# Custom function + VJP (one cached impl per seg, so seg can be a Python arg)
# ---------------------------------------------------------------------------

_impl_cache: dict = {}


def _make_impl(seg):
    """Build (and cache) an ``mx.custom_function`` SSD impl bound to ``seg``.

    ``mx.custom_function`` only forwards array primals to the VJP, so ``seg``
    is closed over here rather than passed as an argument.
    """
    if seg in _impl_cache:
        return _impl_cache[seg]

    @mx.custom_function
    def _impl(u, delta, B_in, C_in, A_neg):
        check_segment_shape(u.shape[1], seg, u.shape[3], "Dh")
        return _ssd_forward_kernel(u, delta, B_in, C_in, A_neg, seg)

    @_impl.vjp
    def _vjp(primals, cotangents, outputs):
        u, delta, B_in, C_in, A_neg = primals
        grad_y = cotangents[0]
        _y, h_ckpt = outputs
        return _ssd_backward_kernel(
            grad_y, h_ckpt, u, delta, B_in, C_in, A_neg, seg
        )

    _impl_cache[seg] = _impl
    return _impl


def ssd_scan(u, delta, B, C, A_neg, seg=DEFAULT_SEG):
    """Head-wise SSD (Mamba-2) selective scan, fused Metal forward + backward.

    Args:
        u:      [B, L, H, Dh]   input projected to heads x channels
        delta:  [B, L, H]       per-token, per-head step size (>= 0)
        B:      [B, L, H, N]    input projection (state-mixing)
        C:      [B, L, H, N]    output projection
        A_neg:  [H, N]          decay rates, A = -exp(A_log) (strictly negative)
        seg:    segment length for checkpointing (L % seg == 0; default 32)

    Returns:
        y:      [B, L, H, Dh]

    Note: ``Dh`` must be a multiple of 32 (it is the simd-reduced lane dim).
    fp32 state/accumulation internally; bf16 inputs widen implicitly.
    """
    y, _h_ckpt = _make_impl(seg)(u, delta, B, C, A_neg)
    return y


def ssd_scan_with_state(u, delta, B, C, A_neg, seg=DEFAULT_SEG):
    """SSD scan that also returns the final per-head state for chunked prefill.

    Returns:
        y:           [B, L, H, Dh]
        final_state: [B, H, Dh, N]   (the Mamba-2 conceptual state layout)
    """
    y, h_ckpt = _make_impl(seg)(u, delta, B, C, A_neg)
    # h_ckpt last boundary is [B, H, N, Dh]; conceptual state is [B, H, Dh, N].
    final_state = mx.swapaxes(h_ckpt[:, -1], -1, -2)
    return y, final_state


# ---------------------------------------------------------------------------
# Pure-MLX reference (slow, for parity testing only)
# ---------------------------------------------------------------------------

def ssd_scan_reference(u, delta, B, C, A_neg):
    """Pure-MLX token-loop reference for :func:`ssd_scan`. Differentiable."""
    B_batch, L, H, Dh = u.shape
    N = A_neg.shape[-1]
    h = mx.zeros((B_batch, H, Dh, N))
    ys = []
    for t in range(L):
        dt = delta[:, t, :, None, None]                       # [B,H,1,1]
        decay = mx.exp(dt * A_neg[None, :, None, :])          # [B,H,1,N]
        inp = dt * B[:, t, :, None, :] * u[:, t, :, :, None]
        h = decay * h + inp
        ys.append(mx.sum(h * C[:, t, :, None, :], axis=-1))
    return mx.stack(ys, axis=1)                                # [B,L,H,Dh]
