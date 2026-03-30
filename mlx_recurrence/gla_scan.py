"""gla_scan.py — GLA recurrence: Metal kernel forward, chunked MLX backward."""

from __future__ import annotations

import mlx.core as mx

from ._utils import _get_or_build_kernel, _linear_scan_direct


# =============================================================================
# GLA Recurrence — Metal Forward Kernel
# =============================================================================

def _gla_forward_kernel(q, k, v, gates):
    """
    Fused Metal kernel for GLA recurrence forward pass.

    Each GPU thread handles one (batch, head, j) triple, maintaining the
    h[:, j] column (Dh floats) in thread-local memory across L timesteps.

    Args:
        q:      [B, L, H, Dh]  — queries (already scaled)
        k:      [B, L, H, Dh]  — keys
        v:      [B, L, H, Dh]  — values
        gates:  [B, L, H]      — forget gates (sigmoid output)

    Returns:
        output: [B, L, H, Dh]
        h_all:  [B, L, H, Dh, Dh]  — saved states for backward
    """
    B_batch, L, H, Dh = q.shape

    q_flat = q.reshape(-1)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    g_flat = gates.reshape(-1)

    source = f"""
        uint j = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;

        if (j >= {Dh}u || bh >= {B_batch * H}u) return;

        uint b = bh / {H}u;
        uint head = bh % {H}u;

        // Thread-local state: one column of the Dh x Dh state matrix
        float h[{Dh}];
        for (int i = 0; i < {Dh}; i++) h[i] = 0.0f;

        for (int t = 0; t < {L}; t++) {{
            // Gate for this (batch, timestep, head)
            int g_idx = b * {L * H} + t * {H} + head;
            float gate = gates[g_idx];

            // k and v for this timestep
            int kv_base = (b * {L} + t) * {H * Dh} + head * {Dh};
            float v_j = v[kv_base + j];

            // Update h column: h[i] = gate * h[i] + k[i] * v[j]
            float o_j = 0.0f;
            for (int i = 0; i < {Dh}; i++) {{
                float k_i = k[kv_base + i];
                float q_i = q[kv_base + i];

                h[i] = gate * h[i] + k_i * v_j;

                // Save h for backward
                int h_idx = ((((b * {L} + t) * {H} + head) * {Dh} + i) * {Dh} + j);
                h_all[h_idx] = h[i];

                // Accumulate output: o[j] = sum_i(q[i] * h[i])
                o_j += q_i * h[i];
            }}

            // Write output
            int out_idx = kv_base + j;
            output[out_idx] = o_j;
        }}
    """

    kernel_name = f"gla_fwd_{B_batch}_{L}_{H}_{Dh}"
    kernel = _get_or_build_kernel(
        kernel_name,
        input_names=["q", "k", "v", "gates"],
        output_names=["output", "h_all"],
        source=source,
    )

    grid = (Dh, B_batch * H, 1)
    tg_x = min(Dh, 64)
    threadgroup = (tg_x, 1, 1)

    results = kernel(
        inputs=[q_flat, k_flat, v_flat, g_flat],
        output_shapes=[
            (B_batch * L * H * Dh,),
            (B_batch * L * H * Dh * Dh,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )

    output = results[0].reshape(B_batch, L, H, Dh)
    h_all  = results[1].reshape(B_batch, L, H, Dh, Dh)
    return output, h_all


def _gla_backward_chunked(grad_out, h_all, q, k, v, gates):
    """
    Backward pass for GLA recurrence using chunked vectorized ops.

    Adjoint recurrence:
        grad_h[t] = q[t] outer grad_out[t] + gate[t+1] * grad_h[t+1]
    runs backward from t = L-1 to 0.
    """
    B_batch, L, H, Dh = q.shape

    q_exp          = mx.expand_dims(q, -1)           # [B, L, H, Dh, 1]
    grad_out_exp   = mx.expand_dims(grad_out, -2)    # [B, L, H, 1, Dh]
    grad_h_from_out = q_exp * grad_out_exp            # [B, L, H, Dh, Dh]

    gates_exp = gates[:, :, :, None, None]            # [B, L, H, 1, 1]
    gates_shifted = mx.concatenate([
        gates_exp[:, 1:, :, :, :],
        mx.ones((B_batch, 1, H, 1, 1))
    ], axis=1)

    grad_h_rev = grad_h_from_out[:, ::-1, :, :, :]
    gates_rev  = gates_shifted[:, ::-1, :, :, :]

    D       = H * Dh * Dh
    dec_flat = mx.broadcast_to(
        gates_rev, (B_batch, L, H, Dh, Dh)
    ).reshape(B_batch, L, D)
    inp_flat = grad_h_rev.reshape(B_batch, L, D)

    grad_h_flat = _linear_scan_direct(dec_flat, inp_flat, chunk_size=32)
    grad_h = grad_h_flat.reshape(B_batch, L, H, Dh, Dh)[:, ::-1, :, :, :]

    h_prev = mx.concatenate([
        mx.zeros((B_batch, 1, H, Dh, Dh)),
        h_all[:, :-1, :, :, :]
    ], axis=1)

    # grad_q[b,t,h,i] = sum_j(grad_out[b,t,h,j] * h[b,t,h,i,j])
    grad_q = mx.sum(
        mx.expand_dims(grad_out, -2) * h_all,
        axis=-1
    )

    # grad_k[b,t,h,i] = sum_j(grad_h[b,t,h,i,j] * v[b,t,h,j])
    v_exp  = mx.expand_dims(v, -2)
    grad_k = mx.sum(grad_h * v_exp, axis=-1)

    # grad_v[b,t,h,j] = sum_i(grad_h[b,t,h,i,j] * k[b,t,h,i])
    k_exp  = mx.expand_dims(k, -1)
    grad_v = mx.sum(grad_h * k_exp, axis=-2)

    # grad_gate[b,t,h] = sum_{i,j}(grad_h[b,t,h,i,j] * h_prev[b,t,h,i,j])
    grad_gates = mx.sum(grad_h * h_prev, axis=(-2, -1))

    return grad_q, grad_k, grad_v, grad_gates


# =============================================================================
# GLA Custom Function — Metal forward + MLX backward
# =============================================================================

@mx.custom_function
def gla_scan_metal(q, k, v, gates):
    """
    Fused GLA recurrence: Metal kernel forward, chunked MLX backward.

    Drop-in replacement for a Python for-loop GLA recurrence.

    Args:
        q:      [B, L, H, Dh]  — queries (pre-scaled, post-RoPE)
        k:      [B, L, H, Dh]  — keys (post-RoPE)
        v:      [B, L, H, Dh]  — values
        gates:  [B, L, H]      — forget gates (after sigmoid)

    Returns: [B, L, H, Dh]
    """
    output, _h_all = _gla_forward_kernel(q, k, v, gates)
    return output


@gla_scan_metal.vjp
def gla_scan_metal_vjp(primals, cotangents, outputs):
    q, k, v, gates = primals
    grad_out = cotangents[0]

    _, h_all = _gla_forward_kernel(q, k, v, gates)

    grad_q, grad_k, grad_v, grad_gates = _gla_backward_chunked(
        grad_out, h_all, q, k, v, gates
    )

    return grad_q, grad_k, grad_v, grad_gates


# =============================================================================
# GLA Pure MLX Fallback — Chunked Vectorized Scan
# =============================================================================

def gla_scan_chunked(q, k, v, gates, chunk_size=32):
    """
    Vectorized GLA recurrence using chunked parallel reduction.
    Pure MLX ops — fully auto-differentiable.

    Args:
        q:          [B, L, H, Dh]
        k:          [B, L, H, Dh]
        v:          [B, L, H, Dh]
        gates:      [B, L, H]
        chunk_size: timesteps per chunk

    Returns: [B, L, H, Dh]
    """
    B_batch, L, H, Dh = q.shape

    gates_log = mx.log(gates + 1e-38)  # [B, L, H]

    k_exp    = mx.expand_dims(k, -1)   # [B, L, H, Dh, 1]
    v_exp    = mx.expand_dims(v, -2)   # [B, L, H, 1, Dh]
    kv_outer = k_exp * v_exp            # [B, L, H, Dh, Dh]

    D       = H * Dh * Dh
    inp_flat = kv_outer.reshape(B_batch, L, D)

    gates_log_exp = mx.expand_dims(mx.expand_dims(gates_log, -1), -1)  # [B, L, H, 1, 1]
    gates_log_bc  = mx.broadcast_to(
        gates_log_exp, (B_batch, L, H, Dh, Dh)
    ).reshape(B_batch, L, D)

    from .ssm_scan import _linear_scan_logdecay
    h_flat = _linear_scan_logdecay(gates_log_bc, inp_flat, chunk_size)
    h      = h_flat.reshape(B_batch, L, H, Dh, Dh)

    q_exp  = mx.expand_dims(q, -1)
    output = mx.sum(q_exp * h, axis=-2)

    return output
