# mlx-recurrence

A plug-in framework of **fused Metal GPU kernels for linear recurrence on Apple
Silicon** — think *flash-linear-attention for [MLX](https://github.com/ml-explore/mlx)*.

Sequential recurrences (SSMs, gated linear attention, diagonal RNNs) are the one
thing MLX cannot fuse for free: a Python loop over `L` timesteps costs `L`
Python→Metal dispatches. These kernels collapse the entire recurrence into a
single dispatch, with a **segment-checkpoint + recompute backward** that cuts
training memory by 12–18× over storing the full state history.

| Kernel | Recurrence | Used by | State |
|---|---|---|---|
| `ssd_scan` | Mamba-2-style head-wise SSD selective scan | Mamba-2 / SSM hybrids | `[B, H, Dh, N]` |
| `gla_scan` | Gated Linear Attention (scalar forget gate, outer-product write) | GLA / linear-attention hybrids | `[B, H, Dh, Dh]` |
| `rglru_scan` | RG-LRU diagonal scan | Griffin / RecurrentGemma-style | `[B, D]` |
| `rotlru_scan` | Rotational LRU: complex-diagonal scan (magnitude gate + per-step rotation of 2D channel pairs) | Complex-LRU / S4-style oscillatory memory | `[B, D]` (interleaved pairs) |

Each kernel is a self-contained plug-in on a shared chassis
(`mlx_recurrence._chassis`) that provides the checkpoint+recompute pattern,
shape validation, and a parity-test helper — adding a new recurrence means
writing one Metal source pair and its VJP wiring, not rebuilding the
infrastructure. The original v0.1 kernels remain available under
`mlx_recurrence.legacy` (and re-exported at top level) for backwards
compatibility.

## Validated in production

These are not microbenchmark-only kernels. The v2 SSD and GLA kernels were
dropped into a live multi-week D-CSIL SSM+GLA hybrid training run mid-flight
(checkpoint pause → parity gates → resume), on an M3 Max, bf16, batch 3,
L=512:

| Gate | v1 (full state history) | v2 (checkpoint + recompute) |
|---|---|---|
| Kernel parity (fwd + every gradient) | — | ~1e-7 rel, all gates PASS |
| Peak training memory | 23.88 GB | **10.34 GB** |
| Sustained tokens/sec | ~1,074 | **~1,481–1,540 (≈1.4×)** |
| Loss continuity across the swap | — | clean (no NaN/inf, same loss band) |

Full report: [`docs/validation/V3_VALIDATION_REPORT_20260610.md`](docs/validation/V3_VALIDATION_REPORT_20260610.md)
(the consuming training repo names these kernels "v3" in its shim — same code).
Note the baseline above is a run *already using fused v0.1-style kernels* —
the gains over having no custom kernels at all are far larger (next section).

## Benchmarks

Two baselines matter, and they answer different questions:

1. **vs. no custom kernels at all** — the Python per-step loop or chunked-MLX
   fallback a user would otherwise write. This is the speedup you get by
   adopting the package.
2. **v2 vs. v0.1-style full-history kernels** — what the
   checkpoint+recompute redesign adds on top, mainly for training memory.

### 1. Fused kernels vs. no custom kernels (M3 Max, seq_len=2048)

| Pass | SSM | GLA |
|---|---|---|
| Forward — fused kernel vs Python per-step loop | **7.3×** | **9.1×** |
| Forward + backward — fused VJP vs chunked-MLX autograd | **19.0×** | **31.8×** |

(Measured for the v0.1 release; charts in `benchmarks/`. Without fused
kernels, training these recurrences on Apple Silicon is impractical above
seq_len ≈ 512 — the backward pass is the killer.)

The v2 kernels measured faster still at training shapes (next table), so
vs-no-kernels speedups for v2 are expected to be at least this large. A
direct single-shape v2-vs-no-kernels measurement is planned once the current
production run frees the GPU, and will replace this estimate.

### 2. v2 vs. v0.1-style full-history kernels (training shapes: B=3, L=512, H=12, Dh=64)

| Kernel | fwd | fwd + bwd | peak memory |
|---|---|---|---|
| SSD, full-history baseline | 3.14 ms | 32.22 ms | 1,792 MB |
| **SSD v2** | **2.30 ms** | **17.34 ms (1.86×)** | **145 MB (12×)** |
| GLA, full-history baseline | 2.10 ms | 17.92 ms | 1,477 MB |
| **GLA v2** | **1.41 ms** | **12.06 ms (1.49×)** | **81 MB (18×)** |

The memory column is the one that matters for training: the baseline stores
every per-timestep state for the backward pass; the v2 kernels store only
segment-boundary checkpoints (1/32 of the writes) and recompute each segment
into a small scratch buffer that stays cache-resident during the adjoint sweep.

## Installation

```bash
pip install mlx-recurrence
```

(Or from source: `pip install git+https://github.com/D-CSIL/mlx-recurrence.git`.
The legacy v0.1-era kernels need no separate install — they ship inside this
package under `mlx_recurrence.legacy` with top-level re-exports.)

Requires: Python >= 3.10, MLX >= 0.22.0, Apple Silicon Mac (Metal GPU).

## Usage (v2 kernels)

All v2 kernels are fully differentiable (`mx.grad` / `mx.value_and_grad`
work through them via custom VJPs), keep **fp32 state and accumulation
regardless of input dtype** (bf16 inputs widen implicitly), and share two
shape constraints from the checkpoint + simd-reduction pattern:

```
L  % seg == 0        # sequence tiles into segments (seg defaults to 32)
lane_dim % 32 == 0   # Dh for ssd/gla, D for rglru (32-lane simdgroups)
```

### SSD selective scan (Mamba-2 style)

```python
import mlx.core as mx
from mlx_recurrence import ssd_scan, ssd_scan_with_state

B, L, H, Dh, N = 3, 512, 12, 64, 16

u     = mx.random.normal((B, L, H, Dh))                  # input
delta = mx.abs(mx.random.normal((B, L, H))) * 0.1 + 0.01 # per-token step size
B_in  = mx.random.normal((B, L, H, N))                   # input projection
C_in  = mx.random.normal((B, L, H, N))                   # output projection
A_neg = -mx.exp(mx.random.normal((H, N)))                # decay rates, < 0

y = ssd_scan(u, delta, B_in, C_in, A_neg)                # -> [B, L, H, Dh]
y, final_state = ssd_scan_with_state(u, delta, B_in, C_in, A_neg)  # chunked prefill
```

### GLA recurrence

```python
from mlx_recurrence import gla_scan, gla_scan_with_state

B, L, H, Dh = 3, 512, 12, 64

q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)   # pre-scaled / post-RoPE
k     = mx.random.normal((B, L, H, Dh))
v     = mx.random.normal((B, L, H, Dh))
gates = mx.sigmoid(mx.random.normal((B, L, H)))          # scalar forget gate, (0,1)

o = gla_scan(q, k, v, gates)                             # -> [B, L, H, Dh]
o, final_state = gla_scan_with_state(q, k, v, gates)     # state: [B, H, Dh, Dh]
```

### RG-LRU diagonal scan (Griffin / RecurrentGemma)

The kernel handles the inner linear scan `h_t = a_t ⊙ h_{t-1} + b_t`; compute
the gate `a` and the already-gated input `b` in pure MLX (cheap, elementwise,
auto-differentiable) and pass them in. The kernel only multiplies — `a` may be
any real value, not just `(0, 1)` (negative / oscillating gates are covered by
the test suite).

```python
from mlx_recurrence import rglru_scan, rglru_scan_with_state

B, L, D = 3, 512, 1536

a = mx.sigmoid(mx.random.normal((B, L, D)))              # per-channel gate
b = mx.random.normal((B, L, D))                          # gated input

y = rglru_scan(a, b)                                     # -> [B, L, D]
y, final_state = rglru_scan_with_state(a, b)             # state: [B, D]
```

### Rotational LRU (complex-diagonal scan)

Generalizes `rglru_scan` from a real diagonal gate to a complex one: each
interleaved channel pair `(u, w)` is scaled by a magnitude gate AND rotated
by an angle every step — `h_t = a_t · e^{iθ_t} · h_{t-1} + b_t` in complex
form, the eigenvalue structure of the complex LRU and S4-style oscillatory
memory. Pass `cos(θ)`/`sin(θ)` computed in MLX host code; gradients w.r.t.
the angle chain through them automatically.

```python
from mlx_recurrence import rotlru_scan, rotlru_scan_with_state

B, L, D = 3, 512, 1536
Dp = D // 2                                              # channel pairs

a     = mx.sigmoid(mx.random.normal((B, L, Dp)))         # magnitude gate per pair
theta = mx.random.uniform(0.0, 3.14, (B, L, Dp))         # rotation per step
b     = mx.random.normal((B, L, D))                      # drive, pairs interleaved

y = rotlru_scan(a, mx.cos(theta), mx.sin(theta), b)      # -> [B, L, D]
y, final_state = rotlru_scan_with_state(a, mx.cos(theta), mx.sin(theta), b)
```

Validated by the parity suite (forward + every gradient vs reference,
negative gates, θ=0 reduces exactly to `rglru_scan`, isometry check) and
exercised by a 10k-step training run; microbenchmarks pending.

Every kernel ships a pure-MLX reference (`*_scan_reference`) for parity
testing and as a fallback on shapes that violate the constraints.

## Testing

```bash
pytest tests/        # 45 tests, ~4 s, tiny shapes
```

- `tests/test_v2_ssd.py`, `test_v2_gla.py`, `test_v2_rglru.py`,
  `test_v2_rotlru.py` — framework parity suites: forward output **and every
  gradient** compared against the pure-MLX reference (two shape configs per
  kernel, multi-segment, plus final-state checks). Negative-gate coverage
  for `rglru`/`rotlru`; θ=0→rglru reduction and isometry checks for `rotlru`.
- `tests/test_v2_legacy_compat.py` — the legacy top-level re-exports keep
  working.
- `tests/test_kernels.py`, `test_backward_metal.py` — original v0.1 suites,
  unchanged.

## Implementation details

### The chassis pattern (shared by all v2 kernels)

**Forward:** run the recurrence once; write only the state at each segment
boundary (`seg=32` → 1/32 the state writes). The last checkpoint doubles as
the chunk's final state, enabling chunked prefill via the `*_with_state`
variants.

**Backward:** walk segments newest → oldest. For each segment, recompute its
per-timestep states from the preceding checkpoint into a small scratch buffer
(one segment's worth — stays resident in the system-level cache instead of
streaming the full history through DRAM), then run the adjoint sweep.
Cross-lane gradient reductions are fused in-kernel with `simd_sum` over
32-lane simdgroups; the remaining sum over simdgroups is one cheap MLX
reduction. Recompute runs the same fp32 ops in the same order from the same
checkpoint, so it reproduces the forward states bit-exactly.

### Per-kernel thread mapping

- **SSD** — one thread per `(batch, head, channel)`; the `N`-element state
  lives in registers across all `L` steps. Checkpoints laid out `[B, nSeg, H,
  N, Dh]` with `Dh` fastest so simdgroup lanes read/write coalesced.
- **GLA** — one thread per `(batch·head, j)`; each thread owns one column of
  the `Dh×Dh` state matrix in registers. `grad_v` is exact per-thread;
  `grad_q`/`grad_k`/`grad_gates` are j-lane `simd_sum` partials.
- **RG-LRU** — one thread per `(batch, channel)` owning the scalar `h[d]`.
  Diagonal state means no cross-lane reductions at all — the simplest plug-in,
  and the template to copy when adding a new diagonal recurrence.
- **Rotational LRU** — one thread per `(batch, pair)` owning the `(u, w)`
  register pair; the 2×2 rotation is applied in-register. Pair-diagonal, so
  like RG-LRU it needs no cross-lane reductions.

### Legacy v0.1 kernels

The original token-loop kernels (`selective_scan_metal`, `gla_scan_metal`,
and the chunked pure-MLX fallbacks) are unchanged under
`mlx_recurrence.legacy` and re-exported at top level. They store the full
state history for the backward pass (fine for inference and short-sequence
training) and have no shape constraints. Original benchmarks (M3 Max,
seq_len=2048): 7.3×/9.1× forward speedup over the Python loop and 19×/31.8×
fwd+bwd over chunked-MLX autograd for SSM/GLA respectively; charts in
`benchmarks/`.

## Citation

If you use mlx-recurrence in your work, please credit:

> Paul O. Derrington, Jr. — Derrington Collaborative Synthetic Intelligence Labs (D-CSIL)

## License

MIT License — Copyright (c) 2026 Paul O. Derrington, Jr.

Matches the MLX license. See [LICENSE](LICENSE).
