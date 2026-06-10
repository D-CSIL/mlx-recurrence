# Design: Gated DeltaNet Metal Kernel (MLX, Apple Silicon)

**Status:** Research + design only. No code, no GPU work. A multi-day training run is
live on this machine; nothing in this document is to be executed.
`/Volumes/SuperDock WD Black 4TB/D-CSIL-3` is strictly read-only and was consulted
only to extract the v3 kernel chassis pattern.

**Audience:** the engineer who will implement this, already fluent in
`ssm_head_scan_v3.py` and `gla_scan_v3.py` (the checkpoint+recompute chassis).

**Epistemic status of claims:**
- ✅ **Confirmed** = taken from a published paper / official implementation, cited inline.
- 🔬 **Hypothesis** = my reasoning about how the chassis maps to this recurrence; not yet validated by parity tests or benchmarks.
- 🚧 **Future work** = out of scope for the first landing.

The recurrence math, conventions, and WY derivation are ✅ confirmed against the papers.
Every statement about *kernel layout, thread mapping, scratch sizing, and performance*
is 🔬 hypothesis until Phase 1/2 parity gates pass — Gated DeltaNet's recurrence is
structurally different from GLA/SSD (it is **not** elementwise decay), so the chassis
mapping is the part most likely to need revision.

---

## 0. Sources (verified this session)

- DeltaNet / WY form: Yang, Wang, Zhang, Kim, Cui, Kasai, others — *"Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length"*, NeurIPS 2024, arXiv:2406.06484.
  (WY representation, generalized Householder, chunkwise parallel.)
- Gated DeltaNet: Yang, Kautz, Hatamizadeh — *"Gated Delta Networks: Improving Mamba2
  with Delta Rule"*, ICLR 2025, arXiv:2412.06464. Official code: github.com/NVlabs/GatedDeltaNet.
- Conventions confirmed from the ICLR camera-ready + official code via web search this
  session: `g_t = exp(α_t)` per-head scalar decay (Mamba-2 parameterization),
  `β_t = σ(b_t)` per-head scalar in (0,1), **q and k L2-normalized** for stability,
  SiLU+short-conv on q/k/v projections, output passes through norm+gating before the
  output projection.

⚠️ **Where formulations conflict** (resolve before implementing — see §9):
1. **Output side convention** (`o_t = S_tᵀ q_t` vs `o_t = q_tᵀ S_t`) differs by a
   transpose of the state across papers/codebases. The prompt's recurrence writes the
   write term as `β_t k_t v_tᵀ` (rank-1 outer, k on the left). We pin the
   **"k-on-left, value-read-by-q" convention** below (§1) and make it internally
   consistent. The implementer MUST re-pin this against whatever attention block in
   *this* repo will consume the kernel, because the GLA v3 kernel here uses
   `kv = k ⊗ v` then `o = q·h` over the `i` (key) axis — that is the opposite outer-product
   orientation from the prompt's `β k vᵀ`. **This is the single highest-risk ambiguity.**
2. **β range:** DeltaNet (2406.06484) experiments include β ∈ (0,2) (`2·σ`) for an
   "anti-Hebbian"/over-write capability; Gated DeltaNet (2412.06464) official code uses
   β = σ(·) ∈ (0,1). We design for **β ∈ (0,1)** (Gated DeltaNet default) but the kernel
   math is range-agnostic — only the host-side activation changes.
3. **Where the gate multiplies:** `α_t` scales the *retained* state only, not the write
   term. Confirmed: `S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ`. The α does
   **not** multiply `β_t k_t v_tᵀ`. (Contrast with a naive "α·everything" reading.)

---

## 1. Forward recurrence + output convention (pinned)

Per batch `b`, head `h`, the state is a matrix `S ∈ R^{Dh×Dh}`. We index rows by
`i` (the **key/feature** axis) and columns by `j` (the **value** axis), so
`S[i, j]`. With this layout the recurrence is:

```
  k_t, q_t  : L2-normalized vectors in R^{Dh}   (key/query, indexed by i)
  v_t       : vector in R^{Dh}                   (value, indexed by j)
  α_t       : scalar in (0,1), α_t = exp(a_t),  a_t ≤ 0   (Mamba-2 decay param)
  β_t       : scalar in (0,1), β_t = σ(b_t)              (write strength)

  S_t = α_t · (I − β_t k_t k_tᵀ) · S_{t-1}  +  β_t · k_t v_tᵀ           (R1)
  o_t = S_tᵀ q_t                                                        (R2)
```

Expanding (R1) into the form used for the adjoint derivation (prompt's eq.):

```
  S_t = α_t S_{t-1} − α_t β_t k_t (k_tᵀ S_{t-1}) + β_t k_t v_tᵀ          (R1')
```

**Index-level reading of (R1') for a single entry `S_t[i,j]`:**

```
  Let  p_t[j] = Σ_i k_t[i] · S_{t-1}[i,j]      # k_tᵀ S_{t-1}, a row vector over j  (Dh)
  S_t[i,j] = α_t · S_{t-1}[i,j]
             − α_t · β_t · k_t[i] · p_t[j]      # rank-1 erase
             + β_t · k_t[i] · v_t[j]            # rank-1 write                       (R1'')
```

**Output (R2) at index `j`:**

```
  o_t[j] = Σ_i q_t[i] · S_t[i,j]               # column-j of S read by q over i      (R2')
```

This is the **"k/q on the row (i) axis, v/o on the column (j) axis"** convention. It is
self-consistent and matches the prompt's `β k vᵀ` write term and `o = Sᵀq`.

🔬 **Critical chassis difference vs GLA.** In `gla_scan_v3.py` the per-thread state column
is `h[:, j]` owned by thread `j`, and the update `h[i] = gate*h[i] + k[i]*v_j` is
**fully elementwise in (i,j)** — thread `j` only ever needs its own `v_j` and the shared
`k[i]`. There is **no coupling across columns**. Gated DeltaNet's erase term needs
`p_t[j] = Σ_i k_t[i] S_{t-1}[i,j]`, which is a **dot product down the i-axis of column j**.
With the GLA layout (thread = column `j`, owns `S[:,j]` = `h[0..Dh-1]` in registers),
`p_t[j]` is a *within-thread* reduction over that thread's own `h[i]` register array —
**no cross-thread communication needed for p**. The cross-thread coupling instead appears
in (a) computing `o_t[j]` (within-thread, fine) and (b) the **backward** pass, where the
adjoint of the erase term couples columns through `k`. See §3 and §4.

This is the key insight: **the GLA per-column thread layout survives for the forward pass**
because `k_tᵀ S_{t-1}` decomposes per-column. 🔬 Hypothesis — to be confirmed by the
Phase-2 forward parity gate.

### Forward, sequential, per-thread (column `j`) pseudocode

```
thread owns column j of S  ->  registers h[i], i=0..Dh-1   (Dh fp32 regs, =64)
for t in 0..L-1:
    load scalars α_t, β_t                         # per (b,h,t)
    load v_j = v[b,t,h,j]                          # this thread's value entry
    # p_j = (k_tᵀ S_{t-1})[j] = Σ_i k[i]*h[i]      # WITHIN-THREAD reduction over i
    p_j = 0
    for i in 0..Dh-1: p_j += k[b,t,h,i] * h[i]
    # update column j
    for i in 0..Dh-1:
        h[i] = α_t*h[i] - α_t*β_t*k[i]*p_j + β_t*k[i]*v_j
    # output o_t[j] = Σ_i q[i]*h[i]                # WITHIN-THREAD reduction over i
    o_j = 0
    for i in 0..Dh-1: o_j += q[b,t,h,i] * h[i]
    output[b,t,h,j] = o_j
    if (t+1) % SEG == 0: checkpoint h[i] -> h_ckpt (i-major, j fastest, coalesced)
```

`k`, `q` for timestep `t` are read redundantly by all `Dh` threads of the head; they
should be staged in threadgroup memory once per `t` (one `Dh`-vector each) to cut DRAM
reads by `Dh×`. `v_j`, `o_j` are per-thread. This is identical in spirit to how GLA v3
reads `k[kv_base+i]` per-thread — but here the redundancy is worth a threadgroup stage
because both `p_j` and `o_j` sweep the full `k`/`q` vectors. 🔬

---

## 2. Chunkwise parallel form (WY representation)

This is the path to GPU-efficient training; it is the form FLA/CUDA uses. We derive it
here at index precision so the Phase-3 kernel can be written without re-reading the paper.
✅ math; 🔬 kernel realizability on MLX.

### 2.1 Setup for one chunk

Split the sequence into chunks of length `C` (use `C=64`; `C=32` for tiny tests). Index
within a chunk by `r,c ∈ {0..C-1}`. Let the chunk start at global time `t0`, and let
`S_in = S_{t0-1}` be the state entering the chunk (`Dh×Dh`).

Per token in the chunk, fold the scalar decay into a cumulative product. Define the
**cumulative gate** from the chunk start:

```
  γ_r = Π_{m=0}^{r} α_{t0+m}        (γ_{-1} = 1)            # ✅ confirmed Mamba-2/GDN fold
```

Work in log space for γ to avoid underflow: `log γ_r = Σ_{m≤r} a_{t0+m}` (a ≤ 0). 🔬

### 2.2 Householder / delta-rule chunk recurrence (un-gated core, then gate-folded)

DeltaNet's core insight (arXiv:2406.06484): the per-token transition
`(I − β_t k_t k_tᵀ)` is a **generalized Householder** matrix, and a product of such
matrices over a chunk admits the **WY representation** — i.e. the product equals
`I − W_chunk K_chunkᵀ` for matrices `W, K ∈ R^{C×Dh}`, computable with one lower-triangular
solve, *without* forming any `Dh×Dh` matrix inside the chunk.

Stack the chunk's vectors as rows:

```
  K = [k_{t0} ; … ; k_{t0+C-1}]    ∈ R^{C×Dh}    # L2-normalized rows
  V = [v_{t0} ; … ]                 ∈ R^{C×Dh}
  Q = [q_{t0} ; … ]                 ∈ R^{C×Dh}
  β = diag(β_{t0..t0+C-1})          ∈ R^{C×C}
```

**Gate folding.** Gated DeltaNet multiplies each retained state by α_t. Following the GDN
chunk algorithm (arXiv:2412.06464 §3, "chunkwise form"), absorb the cumulative decay into
*decayed* key/query/value rows so the inner algebra is the plain DeltaNet WY form:

```
  k̃_r = k_r                       # keys are NOT decayed (they enter the write at time r)
  q̃_r = γ_{r-1} · q_r ??          # see §9 risk: exact placement of γ on q vs k differs
```

🔬→⚠️ **The exact assignment of the γ factors to Q/K/V rows is the most error-prone part
of the WY derivation** and the published chunk algorithm states it as a set of decay
matrices `Λ` applied to specific terms. **Do not hand-transcribe from memory.** Phase 1
(§7) builds a pure-MLX reference *directly from the sequential recurrence (R1'')* and the
Phase-3 chunk kernel is validated *against that reference*, so a γ-placement error is
caught numerically rather than reasoned about. The derivation below gives the *structure*;
the exact γ weights are to be fixed by matching the sequential reference. ✅ structure / 🔬 weights.

### 2.3 The WY auxiliary matrices

Define the strictly-lower-triangular matrix `A ∈ R^{C×C}`:

```
  A[r,c] = β_r · (k_rᵀ k_c)        for c < r,   else 0       # intra-chunk key interactions
```

Solve for `T` (the WY "T-matrix"), a lower-triangular `C×C`:

```
  T = (I + tril(A, -1))^{-1}        # forward substitution, C×C, cheap        (W1)
```

Then the per-token "pseudo-values" / W-matrix:

```
  W = T · (β ⊙ K)                   ∈ R^{C×Dh}     # rows w_r           (W2)
  U = T · (β ⊙ V) − (W) (Sᵀ_in?)    ... see below                       (W3)
```

The chunk produces two output contributions:

**(a) Inter-chunk (contribution of the incoming state `S_in`):**
```
  O_inter[r, :] = (decayed q_r)ᵀ S_in            # each token reads the entering state  (O1)
```
read along the i-axis: `O_inter[r,j] = Σ_i q̃_r[i] · S_in[i,j]`.

**(b) Intra-chunk (contribution of writes inside the chunk):** a causal, strictly-lower
attention-like term:
```
  P = tril( Q̃ K̃ᵀ , -1 ) ∈ R^{C×C}              # token r reads earlier tokens c<r
  O_intra = P · (corrected values)                                              (O2)
```
where the "corrected values" are the WY-corrected value rows `U` (eq. W3) that already
account for the delta-rule erase among intra-chunk tokens.

**(c) Chunk state update (for the next chunk):**
```
  S_out = (Π_chunk α) · S_in   −  (decay-weighted) Kᵀ (erase)  +  Kᵀ U_write    (S1)
```
i.e. `S_out` is `S_in` scaled by the full-chunk decay, with the chunk's net rank-`C`
update applied via `Kᵀ · U` (a `Dh×C · C×Dh → Dh×Dh` matmul). This is the only place a
`Dh×Dh` object is formed, and only **once per chunk** rather than once per token.

### 2.4 Concrete shapes (B=3, L=512, H=12, Dh=64, C=64)

`nChunks = L / C = 8`.

| Object        | Shape                       | Notes                                 |
|---------------|-----------------------------|---------------------------------------|
| q,k,v         | [3, 512, 12, 64]            | fp(in); k,q L2-normed; v raw          |
| α (a_t)       | [3, 512, 12]                | per-head scalar, log-space            |
| β (b_t)       | [3, 512, 12]                | per-head scalar, σ                    |
| per-chunk K,V,Q | [3, 8, 12, 64, 64]        | (chunk, C, Dh) reshaped               |
| A, T          | [3, 8, 12, 64, 64]          | C×C lower-tri, fp32                    |
| W, U          | [3, 8, 12, 64, 64]          | C×Dh                                  |
| S (chunk states) | [3, 8+1, 12, 64, 64]     | inter-chunk carry, Dh×Dh, **fp32**    |
| P (intra)     | [3, 8, 12, 64, 64]          | C×C lower-tri                         |
| o             | [3, 512, 12, 64]            | = O_inter + O_intra                   |

The inter-chunk recurrence over the 8 chunk-states `S` is itself sequential but only 8
steps deep — this is where checkpoint+recompute (§4) applies, exactly as the v3 kernels
checkpoint segment boundaries. Here the natural "segment" = one chunk.

🔬 **MLX realizability note.** The chunk algorithm is matmul-heavy (T-solve, `Q̃K̃ᵀ`,
`KᵀU`). On Apple Silicon these are well served by MLX's own GEMM; a *fully fused* Metal
kernel for the whole chunk is probably **not** worth it for the first landing. The likely
Phase-3 shape is: do the small C×C solves and the C×Dh / Dh×Dh matmuls as MLX ops, and use
a **custom Metal kernel only for the 8-step inter-chunk state recurrence** (the sequential
part), which is the GLA-style scan over chunk-states. That keeps the novel kernel small and
reuses MLX GEMM for the dense parts. See §7 Phase 3.

---

## 3. Sequential fallback design (Metal, v3 chassis)

This is the **first kernel to build** (Phase 2) and the correctness oracle for the chunk
kernel. It mirrors `gla_scan_v3.py` as closely as possible.

### 3.1 Thread mapping (pinned)

- Grid: `x = Dh` (column `j`), `y = B*H` (batch×head). Identical to GLA v3.
- Thread `(j, bh)` owns state **column** `S[:, j]` in registers as `h[i]`, `i=0..Dh-1`.
- `threadgroup = (min(Dh,256), 1, 1)`; lanes of a simdgroup are 32 consecutive `j`.

### 3.2 What changes vs GLA (the rank-1 multiplicative term)

GLA update touches `h[i]` using only `k[i]`, `v_j`, `gate` — no reduction. Gated DeltaNet
needs, per `t`, **two within-thread reductions over `i`** before/with the update:

```
  p_j = Σ_i k[i] · h[i]          # = (k_tᵀ S_{t-1})[j]   — needed for the erase term
  o_j = Σ_i q[i] · h_new[i]      # = o_t[j]              — the output (post-update)
```

Both are reductions down **this thread's own `h[i]` register array** → **no cross-thread
traffic for the forward state update.** This is the crucial finding: the per-column layout
**does** survive, because `k_tᵀ S` decomposes columnwise and each column lives entirely in
one thread. ✅ (algebra) / 🔬 (kernel parity).

**Data movement per `t`:**
- `k[0..Dh-1]`, `q[0..Dh-1]`: shared across all `Dh` threads of the head → stage once into
  threadgroup memory (`2·Dh` floats), then every thread reads from threadgroup. This is the
  one addition vs GLA's pattern and removes `~Dh×` redundant DRAM reads.
- `v_j`, `α_t`, `β_t`: per-thread / per-head scalars, read directly.
- `o_j`: one write per `(t,j)`.

```
# per thread (j, bh):
stage k_t[0..Dh-1], q_t[0..Dh-1] into threadgroup mem (cooperatively, threadgroup_barrier)
p_j = 0
for i: p_j += sk[i] * h[i]                       # sk = staged k
c_erase = α_t * β_t * p_j                          # scalar
c_write = β_t * v_j                                # scalar (v_j per-thread)
o_j = 0
for i:
    h[i] = α_t*h[i] - sk[i]*c_erase + sk[i]*c_write   # = α h − αβ k pⱼ + β k vⱼ
    o_j += sq[i] * h[i]                            # output read AFTER update (post-update o = Sᵀq)
output[b,t,h,j] = o_j
checkpoint at segment boundary (i-major, j fastest)
```

Note `o_t = S_tᵀ q_t` uses the **post-update** state, mirroring GLA v3's "output uses
post-update h". Confirm against the chosen output convention (§9 risk 1).

### 3.3 Does threadgroup cooperation / simd reductions become necessary?

🔬 **Forward: NO.** `p_j` and `o_j` are within-thread. The only shared data is `k_t`,`q_t`,
handled by threadgroup staging (a broadcast, not a reduction). No `simd_sum` needed in fwd.

**Backward: YES, in two places** (see §4): the adjoint of the erase term produces a
gradient contribution that is a **dot product across `i` of quantities held by different
threads' columns**, and the `grad_q/grad_k` reductions are over the `j`-lanes exactly as in
GLA v3 (`simd_sum` over 32 j-lanes). So the backward kernel keeps the GLA v3 `simd_sum`
machinery and **adds** one cross-`j` reduction for the erase adjoint.

---

## 4. Backward pass (adjoint recurrence)

Differentiate (R1''): `S_t = α_t S_{t-1} − α_t β_t k_t (k_tᵀ S_{t-1}) + β_t k_t v_tᵀ`,
`o_t = S_tᵀ q_t`. Let `dS_t = ∂L/∂S_t` (a `Dh×Dh` matrix, the adjoint state), accumulated
newest→oldest as in the v3 kernels.

### 4.1 Output-side seed

```
  o_t[j] = Σ_i q_t[i] S_t[i,j]
  ⇒  ∂o_t[j]/∂S_t[i,j] = q_t[i]
  Seed into the adjoint:  dS_t[i,j] += q_t[i] · go_t[j]          # go = ∂L/∂o_t   (B0)
  grad_q_t[i]            = Σ_j go_t[j] · S_t[i,j]    (= S_t · go_t)               (Bq)
```
`(Bq)` is a reduction over `j` → in the per-column-`j` thread layout this is the
**`simd_sum` over j-lanes** pattern from GLA v3 (`grad_q_p` partials, `[B,L,H,nW,Dh]`).

### 4.2 Pull dS through the recurrence (the new, non-GLA part)

Write the transition compactly: `S_t = M_t S_{t-1} + β_t k_t v_tᵀ`, with
`M_t = α_t (I − β_t k_t k_tᵀ)`. Then:

```
  dS_{t-1} += M_tᵀ dS_t                                                  # (Bp) state pullback
            = α_t dS_t − α_t β_t k_t (k_tᵀ dS_t)
```
Index form (per entry, what a column-`j` thread computes):
```
  Let  d_j = (k_tᵀ dS_t)[j] = Σ_i k_t[i] · dS_t[i,j]      # WITHIN-THREAD reduction over i
  dS_{t-1}[i,j] = α_t · dS_t[i,j] − α_t β_t · k_t[i] · d_j                # (Bp')
```
`d_j` is within-thread (like the forward `p_j`). ✅ So **state pullback needs no
cross-thread comms either** — symmetric to the forward. Good.

### 4.3 Parameter gradients

Need `S_{t-1}` (recomputed from checkpoint, §4.5) and `p_t = k_tᵀ S_{t-1}` (recompute
within-thread). Let `r_t = k_tᵀ S_{t-1}` (row over j) and note the write adds `β k vᵀ`.

```
grad_v_t[j]  = β_t · (k_tᵀ dS_t)[j]                = β_t · d_j           # (Bv) within-thread ✓
```
(because `∂/∂v_t[j]` of `β k_t v_tᵀ` hits `dS_t[i,j]` weighted by `β k_t[i]`, summed over i =
`β·(k_tᵀ dS_t)[j]`). `grad_v` is exact per-thread, like GLA v3's `grad_v`.

```
grad_k_t[i]:  k appears in BOTH erase (−αβ k(kᵀS_{t-1})) and write (+β k vᵀ).
  ∂L/∂k_t[i] = Σ_j dS_t[i,j]·( −α_t β_t p_t[j] + β_t v_t[j] )            # term A (k as left factor)
             + Σ_j ( −α_t β_t S_{t-1}[i,j] · (Σ_{i'} dS_t[i',j] k_t[i']) ) # term B (k inside the dot)
```
- **Term A** is a reduction over `j` → `simd_sum` over j-lanes (GLA-v3 style), per `i`.
- **Term B** contains `e_j := Σ_{i'} k_t[i'] dS_t[i',j] = d_j` (already have it) **times**
  `S_{t-1}[i,j]`, summed over `j`: `Σ_j S_{t-1}[i,j] · d_j`. This is **also** a reduction
  over `j` (within the column-thread we have `S_{t-1}[i,j]` for our `j`, and `d_j` is our
  thread's scalar) → again `simd_sum` over j-lanes. ✅ So `grad_k` = (Term A + Term B),
  both realized as j-lane `simd_sum` partials `[B,L,H,nW,Dh]`. **No extra cross-`i`
  reduction is actually required** once you observe `d_j` is per-thread. 🔬 (verify in test)

```
grad_β_t  = Σ_{i,j} dS_t[i,j] · ( −α_t k_t[i] p_t[j] + k_t[i] v_t[j] )
          = Σ_i k_t[i] · [ Σ_j dS_t[i,j]·( −α_t p_t[j] + v_t[j] ) ]      # (Bβ) scalar per (b,h,t)
grad_α_t  = Σ_{i,j} dS_t[i,j] · ( S_{t-1}[i,j] − β_t k_t[i] p_t[j] )
          = Σ_{i,j} dS_t[i,j] · S_{t-1}[i,j]  −  β_t Σ_i k_t[i] (Σ_j dS_t[i,j] p_t[j])  # (Bα)
```
Both `grad_α`, `grad_β` are full `(i,j)` contractions → compute per-thread partial over `j`
(`simd_sum` j-lanes) then accumulate over `i` (within-thread) → one scalar partial per
simdgroup, reduced over `nW` on the host (`grad_*_p [B,L,H,nW]`, summed → `[B,L,H]`), exactly
like GLA v3's `grad_g_p` for gates. ✅ pattern reuse.

### 4.4 Adjoint loop order (per thread, column j)

```
adj[i] = 0   (i=0..Dh-1)         # this is dS_t[:,j] for our column j, carried across t
for s in nSeg-1 .. 0:
    recompute S states for segment s from checkpoint -> scratch (phase 1, §4.5)
    for tl in SEG-1 .. 0:        # newest -> oldest within segment
        t = s*SEG + tl
        stage k_t, q_t into threadgroup
        go_j = grad_o[b,t,h,j]
        # (B0) seed output grad into adj
        for i: adj[i] += sq[i] * go_j
        # grad_q: reduction over j-lanes of (go_j * S_t[i,j]); S_t = scratch[t]
        for i: gq_l = simd_sum( go_j * Scur[i,j] ); if lane0: grad_q_p[...] = gq_l
        # d_j = kᵀ adj  (within-thread over i)
        d_j = 0; for i: d_j += sk[i]*adj[i]
        # p_j = kᵀ S_{t-1}  (within-thread over i, from scratch[t-1])
        p_j = 0; for i: p_j += sk[i]*Sprev[i,j]
        # grad_v (within-thread exact)
        grad_v[b,t,h,j] = β_t * d_j
        # grad_k term A + B  (both j-lane simd_sum, per i)
        for i:
            gkA = adj[i]*(-α_t*β_t*p_j + β_t*v_j)
            gkB = -α_t*β_t*Sprev[i,j]*d_j
            gk_l = simd_sum(gkA + gkB); if lane0: grad_k_p[...] = gk_l
        # grad_β, grad_α : within-thread partial over i, then simd_sum over j-lanes -> scalar
        gβ_part = Σ_i sk[i]*adj[i]*(-α_t*p_j + v_j)        # scalar (this thread/j)
        gα_part = Σ_i adj[i]*Sprev[i,j] - β_t*sk[i]*adj[i]*p_j   # scalar (this thread/j)
        gβ_l = simd_sum(gβ_part); gα_l = simd_sum(gα_part)
        if lane0: grad_β_p[...] = gβ_l; grad_α_p[...] = gα_l
        # (Bp') pull adj back to t-1
        for i: adj[i] = α_t*adj[i] - α_t*β_t*sk[i]*d_j
```

`grad_v` exact per-thread; `grad_q,grad_k,grad_α,grad_β` via j-lane `simd_sum` partials,
host-reduced over `nW=Dh/32`. **This is exactly GLA v3's epilogue shape**, plus one extra
scalar gate (`α`) and the Term-B addend in `grad_k`.

### 4.5 Checkpoint + recompute (identical chassis)

- Forward saves `h_ckpt[B, nSeg, H, Dh, Dh]` (`i`-major, `j` fastest → coalesced), SEG=32.
  Same as GLA v3 (9.4 MB at training shapes for `Dh=64`).
- Backward, per segment: recompute `S_t` for `t` in the segment into reused scratch
  `[B, H, SEG, Dh, Dh]` (18.9 MB, SLC-resident), running the **forward** recurrence (R1'')
  from the segment's entry checkpoint. **Difference vs GLA recompute:** the recompute itself
  must compute `p_j = kᵀ S_{t-1}` each step (one extra within-thread reduction) — ALU cost,
  not bandwidth. Scratch holds `S_t` (post-update) and the entry checkpoint gives
  `S_{t-1}` for the first step; `S_{t-1}` for later steps = previous scratch slot (the
  `sc_idx - Dh*Dh` trick GLA v3 uses).
- Both `S_t` (scratch[t]) **and** `S_{t-1}` (scratch[t-1] or checkpoint) are needed in the
  adjoint sweep — GLA v3 already loads both `h_cur` and `h_prev`, so no new scratch traffic.

---

## 5. Numerical stability

✅ confirmed from papers/official code unless marked.

- **k, q L2-normalized** (per token, per head, over the `Dh`/`i` axis) before the kernel.
  This bounds `k_tᵀ k_t = 1`, so the Householder factor `(I − β k kᵀ)` has eigenvalues in
  `{1−β (mult. 1), 1 (mult. Dh−1)}` ⇒ with β∈(0,1) it is a **contraction** (spectral norm 1).
  Combined with α∈(0,1), the state recurrence is non-expansive → stable. Do the L2 norm on
  the host (MLX), not in the kernel, so the kernel input contract matches GLA v3 (q,k
  arrive pre-normalized, like GLA's pre-scaled/post-RoPE q).
- **β ∈ (0,1)** via `σ(b_t)` (Gated DeltaNet default). If the implementer wants DeltaNet's
  β∈(0,2) (`2σ`), the kernel is unchanged; only the host activation differs, and stability
  needs re-checking (β>1 can make `1−β<0`, an "anti-Hebbian" over-write — still bounded
  since `|1−β|<1` for β∈(0,2)). ⚠️ see §9 risk 2.
- **α in log space:** keep `a_t = log α_t ≤ 0`; for the chunk cumulative `γ_r = exp(Σ a)`,
  accumulate in log space and exponentiate late. Sequential kernel can multiply α directly
  (no chunk product) since α∈(0,1) per step doesn't underflow over SEG=32 in fp32. 🔬
- **fp32 state and accumulation, always**, regardless of bf16 inputs — identical to v3
  kernels (read bf16 with implicit widening, accumulate fp32). The `p_j`/`d_j` dot products
  are the new fp32-critical reductions: a bf16 accumulation of `Σ_i k[i] h[i]` over Dh=64
  terms would lose ~3 bits and break parity. Accumulate in `float`.
- **FTZ / small activations:** after many decay steps `h[i]` can fall into bf16-subnormal
  range, but since state is fp32 in-register this is not hit during the scan; only the final
  bf16 *output cast* (if any) sees it, same as GLA. No special FTZ handling beyond keeping
  state fp32. 🔬
- **Recompute bit-exactness:** as in v3, recompute runs the *same* fp32 ops in the *same*
  order from the same checkpoint ⇒ reproduces forward states exactly. The added `p_j`
  reduction must use the identical loop order in forward and recompute.

---

## 6. Memory / bandwidth budget (training shapes B=3, L=512, H=12, Dh=64, SEG=32)

Same accounting style as the v3 kernel headers. One Gated DeltaNet layer:

| Quantity                         | Naive `h_all`                         | This design (ckpt+recompute)            |
|----------------------------------|---------------------------------------|-----------------------------------------|
| Per-step state retained          | `h_all [B,L,H,Dh,Dh]` fp32 = **302 MB** | none                                    |
| Forward checkpoint write         | —                                     | `h_ckpt [B,nSeg,H,Dh,Dh]` = **9.4 MB**  |
| Backward scratch (reused, SLC)   | `adj_out [B,L,H,Dh,Dh]` = **302 MB**  | `scratch [B,H,SEG,Dh,Dh]` = **18.9 MB** |
| grad_q partials                  | `[B,L,H,Dh,Dh]`→broadcast temp        | `grad_q_p [B,L,H,nW,Dh]` = **9.4 MB**   |
| grad_k partials                  | full-size temp                        | `grad_k_p [B,L,H,nW,Dh]` = **9.4 MB**   |
| grad_v                           | `[B,L,H,Dh]` = 4.7 MB                 | `[B,L,H,Dh]` = **4.7 MB** (exact)       |
| grad_α partials                  | full-size temp                        | `grad_α_p [B,L,H,nW]` = **0.15 MB**     |
| grad_β partials                  | full-size temp                        | `grad_β_p [B,L,H,nW]` = **0.15 MB**     |
| **DRAM traffic / layer (approx)**| **~1.2–2.4 GB** (h_all + adj + temps) | **~60 MB** (ckpt + partials + grad_v)   |

`nW = Dh/32 = 2`. Numbers for `Dh=64` mirror GLA v3 exactly (GDN state is also `Dh×Dh`),
plus two negligible scalar-gate partials. Across 12 layers the naive design retains
**~3.6 GB** forward→backward; this design retains **~113 MB** of checkpoints, with the
18.9 MB scratch reused per layer. 🔬 the chunkwise (Phase 3) path changes this: it stores
only `nChunks+1=9` chunk-states `[B,9,H,Dh,Dh]=10.6 MB` plus the dense MLX intermediates
(T,W,U,P) which are `[B,nC,H,C×{C or Dh}]` ≈ tens of MB but transient.

**ALU cost vs GLA:** forward adds one `Dh`-length dot (`p_j`) per token; backward adds
`d_j`, `p_j` dots + the Term-B `grad_k` addend. Roughly **+50% FLOPs vs GLA**, but the scan
is bandwidth-bound on M3 Max, so wall-clock impact is expected small. 🔬 measure (§7).

---

## 7. Implementation plan (phased, with gates)

### Phase 1 — Pure-MLX reference + tests (no Metal, safe anytime)
- Implement `gated_deltanet_reference(q,k,v,a,b)` as a literal `for t` loop over (R1'')+(R2'),
  fp32, mirroring `gla_scan_reference`. q,k pre-L2-normalized by caller.
- Also implement an independent **chunkwise** reference (WY form, §2) in pure MLX. Validate
  chunk-ref vs sequential-ref to lock the γ-placement (§2.2) numerically.
- **Gate 1:** chunk-ref vs seq-ref `max|diff| < 1e-4` on tiny shapes; finite-difference grad
  check of seq-ref `rel < 1e-3`. Tiny configs: `B=2,L=64,H=2,Dh=64,C=32`;
  `B=1,L=128,H=3,Dh=32,C=32`.

### Phase 2 — Sequential Metal kernel (fwd+bwd), v3 chassis
- Forward kernel (§3.2): per-column thread, threadgroup-staged k,q, SEG=32 checkpoints.
- Backward kernel (§4.4): recompute + adjoint, j-lane `simd_sum` partials for
  grad_q/k/α/β, exact grad_v. `@mx.custom_function` + `.vjp`, identical wiring to GLA v3.
- **Gate 2:** `_parity_test` vs Phase-1 seq-ref: `y < 1e-3`, all grads `rel < 1e-3` on the
  two tiny configs **plus** a final-state parity check (like GLA v3's). Constraints asserted:
  `L%SEG==0`, `Dh%32==0`.
- **Gate 2b (off-train only):** `--bench` peak-mem ≈ GLA v3 + scalar partials; confirm the
  ~60 MB/layer figure. **Do not run while training is live.**

### Phase 3 — Chunkwise WY (training-speed path)
- Host-side MLX for: per-chunk `A,T` (tri-solve), `W,U`, `P=tril(Q̃K̃ᵀ)`, intra/inter outputs,
  `KᵀU` state update. **Custom Metal kernel only for the 8-step inter-chunk state scan**
  (GLA-style, checkpoint per chunk). This minimizes new kernel surface.
- **Gate 3:** chunk-kernel fwd+bwd vs **Phase-2 sequential kernel** (now the oracle):
  `rel < 1e-3` all grads, tiny + one medium config (`B=2,L=256,H=4,Dh=64,C=64`).
- **Gate 3b:** end-to-end fwd+bwd wall-clock vs Phase-2 sequential at training shapes
  (off-train); expect chunk path faster for long L. If not faster, ship Phase 2 and defer
  Phase 3.

### Wiring / compatibility (do NOT touch training infra)
- Deliver as a standalone module `optimizations/gated_deltanet_v3.py` (mirrors the v3 file
  layout: kernel cache, custom_function+vjp, reference, `_parity_test`, `_bench`). Does not
  import or modify the active pipeline. Adopt only after the live run finishes.
- Input contract: `q,k,v [B,L,H,Dh]` (q,k pre-L2-normed by the block), `a,b [B,L,H]` raw
  pre-activation logits — kernel applies `α=exp(a_clamped≤0)`, `β=σ(b)` internally, OR
  caller passes pre-activated α,β. **Pin this at implementation time** to match the repo's
  GDN block; default to caller-pre-activates (matches GLA v3 passing gates in (0,1)).

---

## 8. Tiny-shape test configs (summary)

| Config | B | L | H | Dh | C/SEG | Purpose                         |
|--------|---|-----|---|----|-------|---------------------------------|
| T1     | 2 | 64  | 2 | 64 | 32    | primary parity (Dh=64, nW=2)    |
| T2     | 1 | 128 | 3 | 32 | 32    | Dh=32 (nW=1) edge, multi-seg    |
| T3     | 2 | 256 | 4 | 64 | 64    | chunk path, nChunks=4           |

All satisfy `L%SEG==0`, `Dh%32==0`. Keep allocations < a few MB so they run beside a live
training job without GPU pressure spikes (these are CPU-fast, tiny-grid kernels).

---

## 9. Open questions / risks

1. ⚠️ **Output convention (highest risk).** `o_t = S_tᵀ q_t` (this doc) vs `q_tᵀ S_t`, and
   the outer-product orientation of the write (`k vᵀ` here vs GLA v3's `k⊗v` with `o=q·h`
   over the key axis). These differ by a transpose of `S`. **Action:** before writing the
   kernel, read the exact GDN attention block that will consume this in *this* repo and
   pin the convention so `grad_q/k/v` map to the right projections. The Phase-1 reference
   must match that block's expected I/O, not just an abstract paper equation.

2. ⚠️ **β range.** (0,1) [Gated DeltaNet] vs (0,2) [DeltaNet anti-Hebbian]. Affects only the
   host activation, but (0,2) changes stability margins and should be re-tested if chosen.

3. 🔬 **γ-factor placement in the WY chunk form (§2.2).** The exact assignment of cumulative
   decays to Q/K/V rows and the intra-chunk `P`/`U` terms is the most transcription-error-
   prone step. Mitigated by validating the chunk reference against the sequential reference
   in Phase 1 — but budget time for getting it right. Do not trust a from-memory transcription.

4. 🔬 **Is the per-column thread layout actually optimal?** It survives correctness (forward
   `p_j`, backward `d_j`/`p_j` are within-thread). But the C×C tri-solve and `KᵀU` of the
   chunk path want a different (tile) layout. The Phase-2 sequential kernel uses the column
   layout; the Phase-3 chunk path mostly uses MLX GEMM + a small GLA-style chunk-state scan.
   Don't force one layout across both.

5. 🔬 **Threadgroup staging of k,q.** Assumed beneficial (cuts `Dh×` redundant reads). On
   `Dh=64`, that's two 64-float stages per `t`; confirm it doesn't blow threadgroup memory
   or hurt occupancy vs just reading from device (GLA v3 reads device directly and is fine).
   Measure both; the simplest correct version (device reads, like GLA v3) is the fallback.

6. 🔬 **bf16 accumulation in the new dot products.** `p_j`,`d_j` MUST accumulate in fp32.
   Flagged here so it isn't lost; the parity gate at `rel<1e-3` will catch a regression.

7. 🚧 **Chunked prefill / inference state export** (like the v3 `*_with_final_state`
   helpers) — the chunk path naturally produces `S_out` per chunk; expose
   `final_state [B,H,Dh,Dh]` for KV-cache-style decoding. Defer to after Phase 3 lands.

8. 🚧 **Gated DeltaNet-2 (NVIDIA, May 2026, arXiv:2605.22791)** decouples erase and write
   strengths (separate β_erase, β_write). Out of scope; note that this design's adjoint
   already separates the erase (`−αβ k pⱼ`) and write (`+β k vⱼ`) terms, so extending to two
   βs is a small, contained change later.

---

## Sources
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length (arXiv:2406.06484)](https://arxiv.org/abs/2406.06484)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (arXiv:2412.06464)](https://arxiv.org/abs/2412.06464)
- [Gated DeltaNet ICLR 2025 camera-ready (openreview r8H7xhYPwz)](https://openreview.net/pdf?id=r8H7xhYPwz)
- [NVlabs/GatedDeltaNet official PyTorch implementation](https://github.com/NVlabs/GatedDeltaNet)
- Repo chassis read (read-only): `/Volumes/SuperDock WD Black 4TB/D-CSIL-3/optimizations/ssm_head_scan_v3.py`, `gla_scan_v3.py`
