# D-CSIL-3 — Architecture & Kernel Bottleneck Analysis

**Date:** 2026-06-09
**Scope:** `d_csil_3.py`, `kernels/v1`, `kernels/v2_evolve`,
`experiments/v1_bf16/` (the pipeline of the currently active run)
**Hardware:** M3 Max, 36 GB unified memory, ~28 GB Metal ceiling

**Live run snapshot at time of analysis:** phase 3
(grounding_reasoning_mix_1000M), step ~1.454M / 3.33M, batch 3 × seq 512
bf16, ~1,074 tok/s, active 7.73 GB, **peak 23.88 GB**, `--no-compile-step`.

No existing file was modified. New code lives in `optimizations/`.

---

## Ranked bottlenecks — training (live v1_bf16 path)

### 1. Full per-timestep state materialization in both scan kernels (dominant)

The two kernels the live run actually executes both write every
timestep's recurrent state to unified memory so the backward pass can
read it back:

| Kernel | Saved tensor | Size per layer (fp32, B=3, L=512) | Layers | Total |
|---|---|---|---|---|
| `experiments/v1_bf16/ssm_head_scan.py` | `h_all [B,L,H,Dh,N]` = [3,512,12,128,64] | 604 MB | 12 | 7.25 GB |
| `kernels/v1/gla_scan.py` | `h_all [B,L,H,Dh,Dh]` = [3,512,12,64,64] | 302 MB | 12 | 3.62 GB |

That is **~10.9 GB of the 23.88 GB peak (~45%)** held from forward until
each layer's backward. On top of that:

- The SSM backward writes an equally sized `adj_out` (another 604 MB
  transient per layer).
- The MLX gradient epilogues (`grad_B = sum(adj_all * delta * u)`,
  `grad_C = sum(h_all * grad_y)`) materialize several more full-size
  broadcast temporaries per layer — and with `--no-compile-step` there
  is no fusion, so every one of them round-trips DRAM.

Net effect: several GB of DRAM traffic per layer per step on a part with
~400 GB/s of bandwidth. This is the single largest cost in the step and
the reason batch size is capped at 3.

**Fix (delivered):** `optimizations/ssm_head_scan_v3.py` and
`optimizations/gla_scan_v3.py` — checkpoint + recompute scans.
See "Delivered kernels" below.

### 2. Batch size capped at 3 (consequence of #1)

At 235M params, weights + AdamW state are ~3.3 GB in bf16/fp32-mixed —
the model itself is small. Activations (overwhelmingly the saved scan
states) consume the rest. Freeing ~10.9 GB of saved scan state (both
v3 kernels delivered) should allow **batch 6–8**,
which on a bandwidth-bound workload translates to a near-proportional
tokens/s improvement on top of the per-step savings.

### 3. fp32-only kernel I/O during bf16 training

All scan kernels read/write fp32 buffers. State precision should stay
fp32 (correctness of the recurrence), but y/grad outputs and the saved
checkpoints could be bf16 with fp32 accumulation — a further ~2× cut on
the remaining scan traffic. Secondary once #1 lands.

### 4. Redundant global reads and exp() recomputation across the head dim

In both scan kernels every one of the `Dh` threads of a head re-reads
the same `B[t,n]`, `C[t,n]`, `delta[t]` values and recomputes
`exp(dt*A)` — 128× redundancy for the SSM. In practice the simdgroup
same-address broadcast and L1 absorb most of it, but threadgroup-memory
staging of B/C/delta per timestep is the natural next refinement of the
v3 kernel if profiling shows it matters.

### 5. `--no-compile-step`

Without `mx.compile`, every elementwise op in the projections, norms,
gates, and loss is a separate dispatch with a materialized output.
Worth re-testing compile compatibility once the custom-VJP kernels are
stable, on a fresh run — do not change the live run.

### 6. Python-loop causal conv1d (minor)

`_causal_conv1d` does K=4 shifted slice-multiply-adds plus a pad concat
per SSM layer. A grouped/depthwise `mx.conv1d` would be one fused op.
Small win; listed for completeness.

---

## Ranked bottlenecks — future edge inference

### 7. `prefill()` is a token-by-token Python loop (largest inference gap)

`DCSIL3.prefill` steps **one token at a time through all 24 layers**:
512 tokens × 24 layers ≈ 12K+ `step()` calls, each ~10–20 small
dispatches → hundreds of thousands of kernel launches to process one
prompt. The chunk-parallel kernels can't help because they don't return
final states.

**Fix path (plumbed):** `selective_scan_heads_v3_with_final_state()`
and `gla_scan_v3_with_final_state()` return `(y, final_state)` with the
states in exactly the `step()` cache layouts (`[B,H,Dh,N]` SSM,
`[B,H,Dh,Dh]` GLA). A chunked prefill then becomes: run the parallel
forward over the prompt, seed every layer's cache from the final
states, and decode from there. Remaining work is the model-level
`prefill_chunked()` wiring.

### 8. `generate_text()` doesn't use the cache path at all

It calls `model(x)` on the full context for every generated token —
O(L) work per token, O(L²) per generation — even though `prefill`/`step`
exist. The "O(1) per token" claim is only realized by the step path.
Any benchmark or demo using `generate_text` is leaving ~an order of
magnitude on the table.

### 9. Decode-step dispatch overhead

Each generated token runs 24 layers of many tiny MLX ops in Python.
For edge SOTA: fuse the per-layer step into one or two Metal kernels
per layer type (the recurrent update is tiny — [H,Dh,N] per layer), and
wrap the whole token step in `mx.compile`. State caches could be bf16.

### 10. v2 main-path GLA `chunk_size=512` (affects `d_csil_3.py`, not the live run)

With seq_len=512 and chunk_size=512, `gla_chunk_metal` degenerates to a
single chunk: the Metal boundary scan does nothing, and the intra-chunk
path computes full [C,C] = [512,512] attention matrices — i.e.
**quadratic attention at training length**, plus [B,1,512,512,H] decay
and QK tensors (~38 MB each ×3 ×12 layers retained for backward).
Chunk 64–128 would engage the scan and cut intra-chunk cost ~4–8×.
Similarly, the v2 SSD intra-chunk path materializes a 6-D
`decay_ratio [B,nC,C,C,H,N]` (~302 MB/layer at training shapes) —
the same class of problem as #1, fixable with the same fused-kernel
approach if the v2 path is ever promoted.

---

## Delivered kernels — `ssm_head_scan_v3.py` + `gla_scan_v3.py`

Drop-in replacements for `selective_scan_heads_metal` and
`gla_scan_metal` (same signatures, same fp32 numerics, same gradient
formulas). Three changes, all aimed at the Apple Silicon memory
hierarchy (described for the SSM kernel; the GLA kernel applies the
identical pattern to its Dh×Dh state with j-lane simd reductions):

1. **Segment checkpointing (forward).** Saves the SSM state only every
   SEG=32 steps: `h_ckpt [B,nSeg,H,N,Dh]` = **18.9 MB vs 604 MB** per
   layer at training shapes. Layout is d-fastest for coalesced stores.

2. **Recompute into SLC-resident scratch (backward).** Each segment's
   states are recomputed from its checkpoint into a reused
   `[B,H,SEG,N,Dh]` scratch (~37.7 MB). Because the buffer is small and
   rewritten per segment, it lives in the M3 Max system-level cache
   instead of streaming 604 MB through DRAM. Recompute is bit-exact
   (same fp32 ops, same order, same starting state).

3. **Fused simd-reduced gradient epilogue.** `grad_B`, `grad_C`,
   `grad_delta` reductions over the head dimension happen in-kernel via
   `simd_sum` across the 32 d-lanes of each simdgroup. Partial outputs
   shrink 32× (`[B,L,H,4,N]` instead of `[B,L,H,128,N]`-scale tensors)
   and the entire MLX broadcast-temporary epilogue disappears.

GLA-specific numbers (B=3, L=512, H=12, Dh=64, SEG=32):
- checkpoints `[B,nSeg,H,Dh,Dh]` = **9.4 MB vs 302 MB** h_all per layer
- backward scratch `[B,H,SEG,Dh,Dh]` ≈ 18.9 MB, SLC-resident
- `grad_q`/`grad_k`/`grad_gates` simd-reduced in-kernel
  (partials `[B,L,H,2,Dh]` ≈ 9.4 MB vs the 302 MB `adj_out` + h_all
  broadcast epilogue); `grad_v` stays per-thread exact as in v1
- `gla_scan_v3_with_final_state()` returns the final `[B,H,Dh,Dh]`
  state in exactly the `GatedLinearAttention.step()` cache layout —
  the GLA half of chunked prefill (bottleneck #7)

**Status (both kernels):** parity verified 2026-06-09 against pure-MLX
references — outputs, all gradients (5 for SSM, 4 for GLA), and the GLA
final state match to ~1e-7 relative (fp32 round-off) across two shape
configurations each. Benchmark mode (`--bench`) exists in both files but
has NOT been run: it allocates >1 GB and must wait until the active
training run finishes.

**Expected impact (to be validated with `--bench` after the run):**
- ~10.5 GB lower peak memory (SSM 7.25 GB + GLA 3.6 GB of h_all, plus
  adj buffers and epilogue temps, replaced by ~340 MB of checkpoints)
- Forward: removes a 604 MB (SSM) / 302 MB (GLA) write per layer
- Backward: trades a second in-register recurrence pass (cheap, the
  kernels are bandwidth-bound) for ~10–30× less DRAM traffic
- Unlocks batch 6–8 → estimated 1.5–2× end-to-end tokens/s combined

**Adoption checklist (after the current run completes):**
1. Parity (safe anytime):
   `python optimizations/ssm_head_scan_v3.py`
   `python optimizations/gla_scan_v3.py`
2. Bench v1 vs v3 (training shapes, GPU-heavy):
   `python optimizations/ssm_head_scan_v3.py --bench`
   `python optimizations/gla_scan_v3.py --bench`
3. Copy `experiments/v1_bf16/d_csil_3_v1_bf16.py` to a new experiment
   dir, swap the two imports to `selective_scan_heads_metal_v3` and
   `gla_scan_metal_v3`, run a short fixed-seed loss-parity probe vs the
   v1 model (e.g. 200 steps, same data order), then increase batch size
   while watching peak memory.
4. Wire chunked prefill using both `*_with_final_state()` variants
   (bottleneck #7) and route `generate_text` through prefill/step (#8).
