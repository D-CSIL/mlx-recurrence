"""ssm_scan.py — SSM selective scan: Metal kernel forward, chunked MLX backward."""

from __future__ import annotations

import mlx.core as mx

from ._utils import _get_or_build_kernel, _linear_scan_direct


# =============================================================================
# SSM Selective Scan — Metal Forward Kernel
# =============================================================================

def _ssm_forward_kernel(u, delta, B_in, C_in, A_neg):
    """
    Fused Metal kernel for SSM selective scan forward pass.

    Each GPU thread handles one (batch, inner_dim) pair and loops over
    L timesteps and state_dim elements sequentially on-GPU.

    Args:
        u:      [B, L, inner_dim]
        delta:  [B, L, inner_dim]
        B_in:   [B, L, state_dim]
        C_in:   [B, L, state_dim]
        A_neg:  [inner_dim, state_dim]  — must be -exp(A_log)

    Returns:
        y:      [B, L, inner_dim]
        h_all:  [B, L, inner_dim, state_dim]  — saved states for backward
    """
    B_batch, L, inner_dim = u.shape
    state_dim = A_neg.shape[-1]

    u_flat     = u.reshape(-1)
    delta_flat = delta.reshape(-1)
    B_flat     = B_in.reshape(-1)
    C_flat     = C_in.reshape(-1)
    A_flat     = A_neg.reshape(-1)

    source = f"""
        uint d = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;

        if (d >= {inner_dim}u || b >= {B_batch}u) return;

        // Thread-local SSM state
        float h[{state_dim}];
        for (int n = 0; n < {state_dim}; n++) h[n] = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            int u_idx = b * {L * inner_dim} + t * {inner_dim} + d;
            float dt_val = delta[u_idx];
            float u_val = u[u_idx];

            float y_val = 0.0f;
            for (int n = 0; n < {state_dim}; n++) {{
                float a_val = A_neg[d * {state_dim} + n];
                int bc_idx = b * {L * state_dim} + t * {state_dim} + n;
                float b_val = B_in[bc_idx];
                float c_val = C_in[bc_idx];

                float decay = exp(dt_val * a_val);
                h[n] = decay * h[n] + dt_val * b_val * u_val;

                // Save h for backward pass
                int h_idx = ((b * {L} + t) * {inner_dim} + d) * {state_dim} + n;
                h_all[h_idx] = h[n];

                y_val += h[n] * c_val;
            }}

            output[u_idx] = y_val;
        }}
    """

    kernel_name = f"ssm_fwd_{B_batch}_{L}_{inner_dim}_{state_dim}"
    kernel = _get_or_build_kernel(
        kernel_name,
        input_names=["u", "delta", "B_in", "C_in", "A_neg"],
        output_names=["output", "h_all"],
        source=source,
    )

    grid = (inner_dim, B_batch, 1)
    tg_x = min(inner_dim, 256)
    threadgroup = (tg_x, 1, 1)

    results = kernel(
        inputs=[u_flat, delta_flat, B_flat, C_flat, A_flat],
        output_shapes=[
            (B_batch * L * inner_dim,),
            (B_batch * L * inner_dim * state_dim,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )

    y     = results[0].reshape(B_batch, L, inner_dim)
    h_all = results[1].reshape(B_batch, L, inner_dim, state_dim)
    return y, h_all


def _ssm_backward_chunked(grad_y, h_all, u, delta, B_in, C_in, A_neg):
    """
    Backward pass for SSM selective scan using chunked vectorized ops.

    Pure MLX operations — auto-differentiable, no custom Metal needed.
    Used as fallback when Metal backward is unavailable.

    Adjoint recurrence:
        grad_h[t] = grad_y[t] * C[t] + dA[t+1] * grad_h[t+1]
    runs backward from t = L-1 to 0.
    """
    B_batch, L, inner_dim = u.shape
    state_dim = A_neg.shape[-1]

    dt_exp  = mx.expand_dims(delta, -1)       # [B, L, inner_dim, 1]
    A_exp   = mx.expand_dims(A_neg, (0, 1))   # [1, 1, inner_dim, state_dim]
    dA_all  = mx.exp(dt_exp * A_exp)           # [B, L, inner_dim, state_dim]

    grad_y_exp      = mx.expand_dims(grad_y, -1)  # [B, L, inner_dim, 1]
    C_exp           = mx.expand_dims(C_in, 2)     # [B, L, 1, state_dim]
    grad_h_from_y   = grad_y_exp * C_exp           # [B, L, inner_dim, state_dim]

    dA_shifted = mx.concatenate([
        dA_all[:, 1:, :, :],
        mx.ones((B_batch, 1, inner_dim, state_dim))
    ], axis=1)
    dA_shifted_rev       = dA_shifted[:, ::-1, :, :]
    grad_h_from_y_rev    = grad_h_from_y[:, ::-1, :, :]

    D             = inner_dim * state_dim
    decay_rev_flat = dA_shifted_rev.reshape(B_batch, L, D)
    input_rev_flat = grad_h_from_y_rev.reshape(B_batch, L, D)

    grad_h_rev_flat = _linear_scan_direct(decay_rev_flat, input_rev_flat, chunk_size=32)
    grad_h = grad_h_rev_flat.reshape(B_batch, L, inner_dim, state_dim)[:, ::-1, :, :]

    h_prev = mx.concatenate([
        mx.zeros((B_batch, 1, inner_dim, state_dim)),
        h_all[:, :-1, :, :]
    ], axis=1)

    # grad_u
    grad_u = mx.sum(
        grad_h * dt_exp * mx.expand_dims(B_in, 2),
        axis=-1
    )

    # grad_B
    u_exp  = mx.expand_dims(u, -1)
    grad_B = mx.sum(grad_h * dt_exp * u_exp, axis=2)

    # grad_C
    grad_C = mx.sum(grad_y_exp * h_all, axis=2)

    # grad_delta
    term1      = A_exp * dA_all * h_prev
    term2      = mx.expand_dims(B_in, 2) * u_exp
    grad_delta = mx.sum(grad_h * (term1 + term2), axis=-1)

    # grad_A_neg
    grad_A_neg = mx.sum(grad_h * dt_exp * dA_all * h_prev, axis=(0, 1))

    return grad_u, grad_delta, grad_B, grad_C, grad_A_neg


# =============================================================================
# SSM Backward — Fused Metal Kernel
# =============================================================================

def _ssm_backward_metal(grad_y, h_all, u, delta, B_in, C_in, A_neg):
    """
    Fused Metal kernel backward pass for SSM selective scan.

    Runs the full adjoint recurrence on-GPU in a single pass per thread,
    computing grad_u, grad_delta, and per-thread grad_A_neg contributions
    directly. Uses MLX for cross-dimension reductions (grad_B, grad_C).

    Each GPU thread handles one (batch, inner_dim) pair and sweeps
    backward through L timesteps, maintaining adjoint state in registers.
    """
    B_batch, L, inner_dim = u.shape
    state_dim = A_neg.shape[-1]

    grad_y_flat = grad_y.reshape(-1)
    h_all_flat  = h_all.reshape(-1)
    u_flat      = u.reshape(-1)
    delta_flat  = delta.reshape(-1)
    B_flat      = B_in.reshape(-1)
    C_flat      = C_in.reshape(-1)
    A_flat      = A_neg.reshape(-1)

    source = f"""
        uint d = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;

        if (d >= {inner_dim}u || b >= {B_batch}u) return;

        // Thread-local adjoint state
        float adj[{state_dim}];
        for (int n = 0; n < {state_dim}; n++) adj[n] = 0.0f;

        // Thread-local accumulator for grad_A_neg (sum over t)
        float gA_local[{state_dim}];
        for (int n = 0; n < {state_dim}; n++) gA_local[n] = 0.0f;

        for (int t = {L - 1}; t >= 0; t--) {{
            int u_idx = b * {L * inner_dim} + t * {inner_dim} + d;
            float dt_val = delta[u_idx];
            float u_val  = u[u_idx];
            float gy     = grad_y[u_idx];

            float gu  = 0.0f;
            float gdt = 0.0f;

            for (int n = 0; n < {state_dim}; n++) {{
                float a  = A_neg[d * {state_dim} + n];
                int bc_idx = b * {L * state_dim} + t * {state_dim} + n;
                float bv = B_in[bc_idx];
                float cv = C_in[bc_idx];

                float decay = exp(dt_val * a);

                int h_idx  = ((b * {L} + t) * {inner_dim} + d) * {state_dim} + n;
                int hp_idx = ((b * {L} + (t - 1)) * {inner_dim} + d) * {state_dim} + n;
                float hp = (t > 0) ? h_all[hp_idx] : 0.0f;

                // Add driving term from output gradient
                adj[n] += gy * cv;

                // Store adjoint for cross-dim reductions
                adj_out[h_idx] = adj[n];

                // Per-thread output gradients
                gu  += adj[n] * dt_val * bv;
                gdt += adj[n] * (a * decay * hp + bv * u_val);

                // Accumulate grad_A_neg contribution (sum over t)
                gA_local[n] += adj[n] * dt_val * decay * hp;

                // Propagate adjoint backward
                adj[n] *= decay;
            }}

            grad_u[u_idx]     = gu;
            grad_delta[u_idx] = gdt;
        }}

        // Write grad_A_neg partial sums (per b,d — needs sum over b in MLX)
        for (int n = 0; n < {state_dim}; n++) {{
            int gA_idx = (b * {inner_dim} + d) * {state_dim} + n;
            grad_A_partial[gA_idx] = gA_local[n];
        }}
    """

    kernel_name = f"ssm_bwd_v2_{B_batch}_{L}_{inner_dim}_{state_dim}"
    kernel = _get_or_build_kernel(
        kernel_name,
        input_names=["grad_y", "h_all", "u", "delta", "B_in", "C_in", "A_neg"],
        output_names=["grad_u", "grad_delta", "adj_out", "grad_A_partial"],
        source=source,
    )

    grid = (inner_dim, B_batch, 1)
    tg_x = min(inner_dim, 256)
    threadgroup = (tg_x, 1, 1)

    results = kernel(
        inputs=[grad_y_flat, h_all_flat, u_flat, delta_flat, B_flat, C_flat, A_flat],
        output_shapes=[
            (B_batch * L * inner_dim,),                  # grad_u
            (B_batch * L * inner_dim,),                  # grad_delta
            (B_batch * L * inner_dim * state_dim,),      # adj_out
            (B_batch * inner_dim * state_dim,),           # grad_A_partial
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )

    grad_u     = results[0].reshape(B_batch, L, inner_dim)
    grad_delta = results[1].reshape(B_batch, L, inner_dim)
    adj_all    = results[2].reshape(B_batch, L, inner_dim, state_dim)
    gA_partial = results[3].reshape(B_batch, inner_dim, state_dim)

    # ── MLX parallel reductions ──────────────────────────────────────────

    # grad_B[b,t,n] = sum_d( adj[b,t,d,n] * delta[b,t,d] * u[b,t,d] )
    dt_u = mx.expand_dims(delta * u, -1)
    grad_B = mx.sum(adj_all * dt_u, axis=2)

    # grad_C[b,t,n] = sum_d( h[b,t,d,n] * grad_y[b,t,d] )
    grad_y_exp = mx.expand_dims(grad_y, -1)
    grad_C = mx.sum(h_all * grad_y_exp, axis=2)

    # grad_A_neg[d,n] = sum_b( gA_partial[b,d,n] )
    grad_A_neg = mx.sum(gA_partial, axis=0)

    return grad_u, grad_delta, grad_B, grad_C, grad_A_neg


# =============================================================================
# SSM Custom Function — Metal forward + MLX backward
# =============================================================================

@mx.custom_function
def _selective_scan_metal_impl(u, delta, B_in, C_in, A_neg):
    """
    Internal: returns (y, h_all) so the VJP can access saved states
    from the forward outputs without re-running the forward kernel.

    Re-running the forward inside the VJP causes Metal buffer aliasing
    at model scale, producing corrupt gradients.
    """
    y, h_all = _ssm_forward_kernel(u, delta, B_in, C_in, A_neg)
    return y, h_all


@_selective_scan_metal_impl.vjp
def _selective_scan_metal_vjp(primals, cotangents, outputs):
    u, delta, B_in, C_in, A_neg = primals
    grad_y = cotangents[0]
    _y, h_all = outputs  # Use saved h_all from forward — no re-run

    grad_u, grad_delta, grad_B, grad_C, grad_A = _ssm_backward_metal(
        grad_y, h_all, u, delta, B_in, C_in, A_neg
    )

    return grad_u, grad_delta, grad_B, grad_C, grad_A


def selective_scan_metal(u, delta, B_in, C_in, A_neg):
    """
    Fused selective scan: Metal kernel forward + Metal kernel backward.

    Drop-in replacement for a Python for-loop SSM selective scan.

    Args:
        u:      [B, L, inner_dim]
        delta:  [B, L, inner_dim]
        B_in:   [B, L, state_dim]
        C_in:   [B, L, state_dim]
        A_neg:  [inner_dim, state_dim]  — must be -exp(A_log)

    Returns: [B, L, inner_dim]
    """
    y, _h_all = _selective_scan_metal_impl(u, delta, B_in, C_in, A_neg)
    return y


# =============================================================================
# SSM Pure MLX Fallback — Chunked Vectorized Scan
# =============================================================================

def _linear_scan_logdecay(decay_log, inp, chunk_size=32):
    """
    Chunked linear recurrence in log-decay space.
    h[t] = exp(decay_log[t]) * h[t-1] + inp[t]

    Uses closed-form: h[t] = P[t] * (h_prev + cumsum(inp / P))
    where P[t] = exp(cumsum(decay_log)).

    Args:
        decay_log: [B, L, D]  — log of decay (negative values)
        inp:       [B, L, D]  — input values

    Returns: [B, L, D]
    """
    B, L, D = decay_log.shape
    h_prev = mx.zeros((B, 1, D))
    chunks = []

    for t0 in range(0, L, chunk_size):
        t1  = min(t0 + chunk_size, L)
        dec = decay_log[:, t0:t1, :]
        x   = inp[:, t0:t1, :]

        log_P    = mx.cumsum(dec, axis=1)
        P        = mx.exp(log_P)
        x_scaled = x / (P + 1e-30)
        S        = mx.cumsum(x_scaled, axis=1)
        h_chunk  = P * (h_prev + S)

        chunks.append(h_chunk)
        h_prev = h_chunk[:, -1:, :]

    return mx.concatenate(chunks, axis=1)


def selective_scan_chunked(u, delta, B_in, C_in, A_neg, chunk_size=32):
    """
    Vectorized selective scan using chunked parallel reduction.
    Pure MLX ops — fully auto-differentiable, no custom VJP needed.

    Reduces Python loop from L iterations to L/chunk_size iterations.
    Each iteration uses vectorized cumsum/cumprod within the chunk.

    Args:
        u:          [B, L, inner_dim]
        delta:      [B, L, inner_dim]
        B_in:       [B, L, state_dim]
        C_in:       [B, L, state_dim]
        A_neg:      [inner_dim, state_dim]
        chunk_size: timesteps per chunk

    Returns: [B, L, inner_dim]
    """
    B_batch, L, inner_dim = u.shape
    state_dim = A_neg.shape[-1]

    dt_exp    = mx.expand_dims(delta, -1)     # [B, L, D, 1]
    A_exp     = mx.expand_dims(A_neg, (0, 1)) # [1, 1, D, N]
    decay_log = dt_exp * A_exp                 # [B, L, D, N]

    inp = dt_exp * mx.expand_dims(B_in, 2) * mx.expand_dims(u, -1)

    DN        = inner_dim * state_dim
    decay_flat = decay_log.reshape(B_batch, L, DN)
    inp_flat   = inp.reshape(B_batch, L, DN)

    h_flat = _linear_scan_logdecay(decay_flat, inp_flat, chunk_size)

    h     = h_flat.reshape(B_batch, L, inner_dim, state_dim)
    C_exp = mx.expand_dims(C_in, 2)
    y     = mx.sum(h * C_exp, axis=-1)

    return y
