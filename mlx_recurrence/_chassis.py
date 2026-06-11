"""_chassis.py — Shared infrastructure for chassis-based recurrence kernels.

This module factors out the machinery common to every v2 plug-in kernel
(SSD, GLA, RG-LRU) so each kernel file only has to supply its own Metal
source strings and gradient wiring. It deliberately does NOT abstract the
Metal source itself: every recurrence has a different state shape and
update rule, and the kernel bodies are meant to stay readable per-kernel.

What lives here
---------------
1. Kernel cache / builder around ``mx.fast.metal_kernel`` so each unique
   shape configuration is JIT-compiled exactly once per process.
2. Shape validation shared by the segment-checkpoint + recompute pattern:
   the sequence length must tile evenly into segments (``L % seg == 0``)
   and the simd-reduced lane dimension must be a multiple of 32.
3. A reusable parity-test helper that compares forward output plus every
   gradient against a pure-MLX reference and reports max abs / rel diffs.

The segment-checkpoint + recompute pattern (shared design)
----------------------------------------------------------
Every kernel here follows the same playbook, tuned to the Apple Silicon
unified-memory hierarchy:

  Forward:  run the recurrence once, write only the state at each
            segment boundary -> ``h_ckpt`` (SEG=32 => ~1/32 the writes of
            saving every timestep). The last checkpoint doubles as the
            chunk's final state, enabling chunked prefill.

  Backward: walk segments newest -> oldest. For each segment, recompute
            its per-timestep states from the preceding checkpoint into a
            small scratch buffer (one segment's worth, stays resident in
            the system-level cache instead of streaming the full state
            history through DRAM), then run the adjoint sweep over that
            segment. Cross-lane gradient reductions are fused in-kernel
            with ``simd_sum`` over 32-lane simdgroups; the remaining
            sum-over-simdgroups is a single cheap MLX reduction.

All kernels keep fp32 state and accumulation regardless of input dtype,
and reproduce the forward states bit-exactly on recompute (same fp32 ops,
same order, from the same checkpoint).
"""

from __future__ import annotations

import mlx.core as mx

# Default segment length for the checkpoint+recompute pattern. SEG=32 is a
# sweet spot on M3 Max: matches the 32-lane simdgroup width used by the
# fused reductions and keeps the per-segment scratch buffer small enough to
# stay SLC-resident at training shapes.
DEFAULT_SEG = 32

# Simd lane width on Apple GPUs. Lane dimensions reduced with simd_sum must
# be a multiple of this.
SIMD_WIDTH = 32


# ---------------------------------------------------------------------------
# Kernel cache / builder
# ---------------------------------------------------------------------------

_kernel_cache: dict = {}


def get_or_build_kernel(name, input_names, output_names, source, header=""):
    """Compile a Metal kernel once per unique ``name`` and cache it.

    ``name`` should encode every shape/template constant baked into
    ``source`` (e.g. ``f"ssd_fwd_{B}_{L}_{H}_{Dh}_{N}_{seg}"``) so that
    distinct shapes get distinct compiled kernels and identical shapes
    reuse the cached one.

    ``source`` is the kernel BODY only (no ``kernel void`` signature) — MLX
    generates the signature from ``input_names`` / ``output_names``. Helper
    functions / includes go in ``header``.
    """
    if name not in _kernel_cache:
        _kernel_cache[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
        )
    return _kernel_cache[name]


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def check_segment_shape(L, seg, lane_dim, lane_name="lane dimension"):
    """Validate the constraints of the segment-checkpoint + simd-reduce pattern.

    Args:
        L:         sequence length.
        seg:       segment length for checkpointing.
        lane_dim:  the dimension mapped to 32-lane simdgroups (must tile by 32).
        lane_name: human-readable name of ``lane_dim`` for the error message.

    Raises:
        ValueError: if either constraint is violated.
    """
    if seg <= 0:
        raise ValueError(f"seg must be positive, got {seg}")
    if L % seg != 0:
        raise ValueError(
            f"sequence length L={L} must be divisible by seg={seg} "
            f"(segment-checkpoint pattern tiles L into L/seg segments)"
        )
    if lane_dim % SIMD_WIDTH != 0:
        raise ValueError(
            f"{lane_name}={lane_dim} must be a multiple of {SIMD_WIDTH} "
            f"(fused gradient reductions use {SIMD_WIDTH}-lane simdgroups)"
        )


# ---------------------------------------------------------------------------
# Parity-test helper
# ---------------------------------------------------------------------------

def parity_check(
    kernel_fn,
    reference_fn,
    inputs,
    arg_names,
    grad_argnums,
    *,
    w_out=None,
    y_tol=1e-3,
    grad_rtol=1e-3,
    label="",
    verbose=True,
):
    """Compare a kernel against a pure-MLX reference: forward + all grads.

    Both ``kernel_fn`` and ``reference_fn`` take the positional ``inputs``
    and return the forward output ``y``. This helper builds a scalar loss
    ``sum(y * w_out)`` and compares ``mx.grad`` of that loss w.r.t. every
    argument in ``grad_argnums``.

    Args:
        kernel_fn:    the kernel under test, ``fn(*inputs) -> y``.
        reference_fn: pure-MLX reference, ``fn(*inputs) -> y``.
        inputs:       tuple/list of input arrays (positional).
        arg_names:    names for each input (for readable output), same length
                      as ``inputs``.
        grad_argnums: tuple of argument indices to differentiate.
        w_out:        output weighting for the scalar loss. Defaults to a
                      fixed-seed random tensor shaped like ``y``.
        y_tol:        absolute tolerance on the forward output diff.
        grad_rtol:    relative tolerance on each gradient diff.
        label:        prefix printed before results.
        verbose:      print per-tensor diffs.

    Returns:
        (ok: bool, report: dict) where ``report`` maps each compared tensor
        name to ``{"abs": max_abs_diff, "rel": max_rel_diff}``.
    """
    inputs = list(inputs)

    y_k = kernel_fn(*inputs)
    y_r = reference_fn(*inputs)
    mx.eval(y_k, y_r)

    if w_out is None:
        mx.random.seed(1234)
        w_out = mx.random.normal(y_r.shape)
        mx.eval(w_out)

    def loss_kernel(*args):
        return mx.sum(kernel_fn(*args) * w_out)

    def loss_ref(*args):
        return mx.sum(reference_fn(*args) * w_out)

    report = {}
    ok = True

    y_abs = float(mx.max(mx.abs(y_k - y_r)))
    y_scale = float(mx.max(mx.abs(y_r))) + 1e-8
    y_rel = y_abs / y_scale
    report["y"] = {"abs": y_abs, "rel": y_rel}
    ok = ok and (y_abs < y_tol)

    g_k = mx.grad(loss_kernel, argnums=grad_argnums)(*inputs)
    g_r = mx.grad(loss_ref, argnums=grad_argnums)(*inputs)
    mx.eval(g_k, g_r)

    if not isinstance(g_k, (tuple, list)):
        g_k = (g_k,)
        g_r = (g_r,)

    grad_names = [arg_names[i] for i in grad_argnums]
    for name, gk, gr in zip(grad_names, g_k, g_r):
        abs_diff = float(mx.max(mx.abs(gk - gr)))
        scale = float(mx.max(mx.abs(gr))) + 1e-8
        rel = abs_diff / scale
        report[f"grad_{name}"] = {"abs": abs_diff, "rel": rel}
        ok = ok and (rel < grad_rtol)

    if verbose:
        prefix = f"{label}  " if label else ""
        print(f"{prefix}y          max|diff| = {y_abs:.2e}  (rel {y_rel:.2e})")
        for name in grad_names:
            r = report[f"grad_{name}"]
            print(
                f"{prefix}grad_{name:<8} max|diff| = {r['abs']:.2e}"
                f"  (rel {r['rel']:.2e})"
            )
        print(f"{prefix}-> {'PASS' if ok else 'FAIL'}")

    return ok, report
