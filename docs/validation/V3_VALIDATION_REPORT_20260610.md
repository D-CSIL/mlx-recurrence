# v3 Kernel Validation Report — 2026-06-10

**Verdict: ALL GATES PASS → Phase 3 resumed on v3 kernels, batch 3.**

## Timeline
| Event | Time | Detail |
|---|---|---|
| Scheduled checkpoint | 01:56 | `epoch1_step1465673.npz` + all sidecars |
| Pause (SIGINT) | 06:28 | Graceful — trainer wrote final save and exited cleanly* |
| Final checkpoint | 06:29 | `03_grounding_reasoning_mix_1000M_final.npz` @ step **1,477,245** |
| Validation suite | 06:29–06:38 | parity → bench → 300-step b3 probe → 100-step b6 probe |
| Resume on v3 | ~06:40 | From step 1,477,245, target 1,953,123 (475,878 new steps) |

*The 01:57→06:28 gap: the checkpoint-watcher notification arrived late
(machine sleep), so training simply ran ~4.5 h further before the pause.
No data lost — the resume point is the exact stop step.

## Gate results (Paul's criteria)

| Gate | Requirement | Result | |
|---|---|---|---|
| Kernel parity | PASS both kernels | ~1e-7 rel, all grads, 2 configs each | PASS |
| Loss continuity | 2–4.5, no NaN/inf | 2.07–4.43, normal grad norms 4.3–9.2 | PASS |
| Peak memory | ≤ 23.88 GB (v1 baseline) | **10.34 GB** (b3) — 13.5 GB freed | PASS |
| **tok/s @ batch 3** | **≥ ~1,074 (v1 pace)** | **1,481–1,540 steady-state (~1.4×)** | **PASS** |

## Kernel microbenchmarks (training shapes, B=3 L=512)

| Kernel | fwd | fwd+bwd | peak mem |
|---|---|---|---|
| SSM v1 (h_all) | 3.14 ms | 32.22 ms | 1,792 MB |
| **SSM v3** | **2.30 ms** | **17.34 ms (1.86×)** | **145 MB (12×)** |
| GLA v1 (h_all) | 2.10 ms | 17.92 ms | 1,477 MB |
| **GLA v3** | **1.41 ms** | **12.06 ms (1.49×)** | **81 MB (18×)** |

## Probe details
- **300-step batch-3 resume probe** (apples-to-apples vs live run flags):
  resumed step 1,477,245 → 1,477,545. tok/s ramped 1,464 → 1,540 peak,
  ~1,481–1,540 steady. Active 5.80 GB, peak 10.34 GB. Loss bounced
  2.07–4.43 — same band as the phase 3 log tail.
- **100-step batch-6 memory probe**: tok/s 1,356 → 1,470 and still
  climbing when the 100-step cap hit (too short for steady state);
  peak **14.66 GB** — comfortably inside the 28 GB ceiling. Confirms
  batch 6 is viable for the between-phases plan; expect better than
  1,470 tok/s over longer runs.

## Resume configuration (running now)
- Shim: `optimizations/train_v3.py` (v3 kernels patched into the
  UNMODIFIED v1_bf16 trainer — no pipeline files changed)
- Checkpoint: `.../run_20260609_000905/checkpoints/03_grounding_reasoning_mix_1000M_final.npz`
- Flags: identical to the crash-resume wrapper (batch 3, bf16, chunked CE,
  lr 3e-4 cosine, total-training-steps 3,332,734, save-every 32,552,
  save-final) with `--max-new-steps 475878` → phase target step 1,953,123
- ETA at ~1,500 tok/s: **~5.4–5.6 days** (vs ~7.9 days on v1 — ~2.3 days saved)

## Full logs
- Console (parity + bench + suite): `optimizations/validation_runs/v3_test_console_20260610_062918.log`
- b3 probe: `optimizations/validation_runs/v3_test_b3_20260610_062928/run_*/training.log`
- b6 probe: `optimizations/validation_runs/v3_test_b6_20260610_062928/run_*/training.log`
