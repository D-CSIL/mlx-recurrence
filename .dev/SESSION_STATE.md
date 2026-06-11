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
**PUSHED:** `v2-framework` branch is live on github.com/D-CSIL/mlx-recurrence
(commit 6e586be7, remote name `github` in this clone). NOTE: the repo is
**PUBLIC**, not private as previously recorded. README was refreshed before
push (v2 API docs, real-run validation numbers from docs/validation/, legacy
section preserved); local agent-hook logs/ untracked. `main` is untouched —
merge decision still Paul's.

## Remaining
1. Merge v2-framework -> main (Paul's call; suggest after the live run ends
   ~2026-06-15 so the multi-day stability note can go in the same merge)
2. Gated DeltaNet implementation (3 phases, design doc ready)
3. PyPI release of 0.2.0 once merged (README install section says
   from-source for v2 until then)
