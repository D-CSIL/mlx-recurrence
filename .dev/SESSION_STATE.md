# mlx-recurrence-v2 — Session State

**Last updated:** 2026-06-10
**This folder:** standalone clone of `D-CSIL-1/mlx-recurrence` at branch
`v2-framework` (+ validation docs). Created so (a) the GitHub update can be
prepped here and (b) the HELIX Claude instance can consume the kernels
without touching the original repo or the live D-CSIL-3 training tree.

## What's here
- `mlx_recurrence/_chassis.py` — shared checkpoint+recompute kernel infra
- `mlx_recurrence/ssd.py` — Mamba-2-style SSD scan (`ssd_scan`, `ssd_scan_with_state`)
- `mlx_recurrence/gla.py` — GLA scan (`gla_scan`, `gla_scan_with_state`)
- `mlx_recurrence/rglru.py` — RG-LRU diagonal scan (RecurrentGemma/Griffin)
- `mlx_recurrence/legacy/` — published v1 kernels (back-compat re-exports at top level)
- `tests/` — 20 v2 parity tests (verified passing in this clone 2026-06-10)
- `docs/DESIGN_gated_deltanet.md` — full implementation design for the next kernel
- `docs/validation/` — D-CSIL-3 real-run validation report + bottleneck analysis
  (the benchmark provenance: SSD 1.86x/12x mem, GLA 1.49x/18x mem,
  end-to-end 1,074→~1,500 tok/s sustained on a 259M model, M3 Max)

## Constraints for ALL sessions (incl. HELIX instance)
- Tiny-shape parity tests only while D-CSIL-3 training is live (multi-day
  run, ends ~2026-06-15). NO benchmarks/big GPU runs until then.
- Numerics rules: fp32 state + accumulation; L % seg == 0; lane dim % 32 == 0.
- Usage: `from mlx_recurrence import ssd_scan, gla_scan, rglru_scan`
  (run pytest first in any new environment).

## GitHub status (updated 2026-06-11)
**MERGED TO MAIN (Paul's call, same day):** v2 framework is the public face of
github.com/D-CSIL/mlx-recurrence (main @ 6a587663+; remote name `github` in
this clone). NOTE: the repo is **PUBLIC**, not private as previously recorded.
README fully refreshed: benchmarks lead with vs-NO-kernels numbers (19x/31.8x
backward — Paul's framing correction), v2-vs-v1 table second with combined
single-shape measurement deferred to post-run; v2 API docs; legacy section
preserved; agent-hook logs/ untracked. Branch v2-framework synced with main.
The committed .dev/SESSION_STATE.md is now public (benign: paths/plans, no
secrets) — prune if Paul prefers.

## NEW PLUG-IN: rotlru (2026-06-11 overnight, Paul-authorized autonomous build)
`mlx_recurrence/rotlru.py` — rotational LRU: pair-diagonal complex scan
h_t = a_t·R(θ_t)·h_{t-1} + b_t (interleaved (u,w) pairs; cs/sn host-computed,
angle grads chain through them). Chassis pattern, no simd_sum needed (like
rglru). 7/7 parity tests (`tests/test_v2_rotlru.py`): fwd+all-grads vs
reference ×2 configs, negative gates, θ=0 ≡ rglru reduction, norm
preservation, final-state, constraint raises. Built for HELIX's HSL rotation
variant (third bake-off arm). NOT yet exported from `__init__.py` — deferred
because a live bake-off process freshly imports the package via HELIX's
bridge; add the export + README section after runs complete.

## Remaining
0. Export rotlru in `__init__.py` + README section (after overnight bake-off)
1. Multi-day stability note in README once the live run completes (~2026-06-15)
2. Single-shape v2-vs-no-kernels benchmark (GPU-gated, replaces README estimate)
3. Gated DeltaNet implementation (3 phases, design doc ready)
4. PyPI release 0.2.0 (README points from-source until then)
