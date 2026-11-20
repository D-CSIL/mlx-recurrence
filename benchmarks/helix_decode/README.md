# HELIX: where the prefill and decode gains come from

A rotational scan and a routed feed-forward layer accelerate different parts of a request.
The **existing** [rotational scan](../../mlx_recurrence/rotlru.py) handles the input sequence
(prefill). This example adds **two Metal shaders** for one-token cached decode: one computes
gate/up projections and SwiGLU for the two selected experts; the other computes their down
projections and weighted sum. Packing the four experts' weights once, before inference,
is a separate optimization and accounts for most of the measured decode gain.

## Run the weight-free example

From the repository root, using native arm64 Python on an Apple Silicon Mac:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "mlx==0.30.0"
python benchmarks/helix_decode/bench.py --rounds 48 --seed 800 --csv fresh.csv
python benchmarks/helix_decode/report.py
```

The first script creates synthetic FP32 inputs from the seed: no downloads of models,
tokens or corpora. The optional CSV path must not exist. The second script recomputes
statistics from the **included** receipts (works on other machines with Python alone).
These are examples, not supported package entry points. No new training/backward path.

**Measured synthetic routed-layer latency** on one 36 GB M3 Max, macOS 27.2, Python 3.12.9,
MLX 0.30.0: batch 1, one token, width 1,024, FFN width 2,048, 4 experts, top 2.

| Implementation | What is evaluated | Median ms |
|---|---|---:|
| Stack + dense | Restack all frozen matrices each call; compute all four experts | 1.8374 |
| Packed dense | Reuse resident packs; compute all four | 0.9529 |
| Packed gather | MLX `gather_mm`, selected two | 0.6446 |
| Packed Metal | Same selection, two custom dispatches | 0.5726 |

48 timed observations per path, balanced mode order, same input and routing logits
within each round, 12 ordered expert pairs warmed and checked before timing. Medians
include host graph construction, GPU execution and synchronous `mx.eval(out)`, not
GPU counters. Max absolute output difference vs stack+dense was 6.56e-7 (limit 1e-4).
One pack was 96 MiB and took 7.8 ms in this run; five such layers retain **480 MiB**
besides the original weights. Rebuild packs after changing weights. Do not pass arbitrary
expert IDs to a raw shader: `selected_decode` derives in-range IDs from four routing
logits on device. The example's router is illustrative, not the HELIX learned router.
The dispatch and Metal source match the measured prototype; the standalone harness does
not import or patch an external model.

## Historical whole-model measurements (not the synthetic run)

One frozen 209.05M HELIX step-800 checkpoint, FP32, one M3 Max, MLX 0.30.0.
Inputs were tokenized before timing. Each condition used identical input tokens within
a group. The example above reproduces the **operators**, not the checkpoint-specific
model, prompts, rates or cache behavior. Only timing, shape, dispatch and parity scalars
from these earlier runs are included, not weights, text or token IDs.

**Same request, four conditions.** Reference = nonfused rotational recurrence +
training-safe dense router. Prefill only = existing scan when input length is at least
128 and divisible by 32. Decode only = prepacked top-two Metal after prefill; both =
the same cache handed from fused prefill to the cached decode path. All modes retained
the frozen packs; the timer excludes packing, loading, tokenization, compilation and
network. This is a warm, single-stream comparison, not a throughput or quality claim
for another model or system.

| Input + output tokens | Reference | Prefill only | Decode only | Both | Paired ref/both (range; rounds) |
|---|---:|---:|---:|---:|---|
| 44 + 17 | 188.5 | 189.7 | 111.8 | 110.5 | 1.71× (1.62–1.78; 8) |
| 128 + 17 | 222.6 | 185.6 | 146.1 | 111.2 | 2.04× (1.94–2.07; 8) |
| 160 + 17 | 239.0 | 189.5 | 159.6 | 112.3 | 2.12× (2.03–2.19; 8) |
| 192 + 17 | 247.9 | 189.0 | 170.4 | 116.0 | 2.18× (2.12–2.25; 8) |
| 512 + 64 | 901.3 | 739.8 | 581.7 | 406.2 | 2.20× (2.14–2.43; 4) |
| 960 + 64 | 1,137.8 | 787.1 | 794.2 | 454.4 | 2.51× (2.42–2.54; 4) |
| 4,032 + 64 | 3,278.9 | 1,509.8 | 2,987.1 | 1,189.6 | 2.76× (2.62–2.80; 4) |

Times are median full-request **milliseconds**. Speedup is the median of paired ratios,
not a ratio of medians. Compare to **prefill only** if that scan is already enabled:
at 4,032 + 64, adding decode reduced median latency from 1,510 to 1,190 ms rather
than from 3,279. At 44 input tokens the prefill scan never ran, as intended.
Prefill latency ends at materialized logits and cache; time to first token ends at
the first host-visible greedy selection. Subsequent cached forwards synchronize at each
greedy selection; total time ends with the final caches materialized. Fixed-budget
generation continued over EOS. All 44 matched four-way groups yielded identical token
sequences. Independent full-vocabulary and boundary-cache checks at all seven lengths
used 1e-3 absolute tolerance. At eligible lengths, both used 5 rotational dispatches
in prefill, zero in decode, then 5 gate/up and 5 down/mix calls per **cached** token.
The first output token comes from prefill. Lengths test execution, not recall quality.

**Decode controls.** Excluding prefill, the fixed 16-forward loop (48/128/192 input tokens,
8 balanced rounds each) measured 95.80 cached steps/s baseline, 163.75 packed dense,
177.62 packed MLX gather, and 180.81 packed Metal. Custom Metal vs stock was 1.886×
paired; vs selected-only MLX it was just 1.023× (17/24 wins). In the separate application
loop (3 prompts, 6 rounds, up to 22 output tokens with stopping, counting first token
from prefill logits) baseline/gather/Metal measured 93.29/170.31/179.66 output tokens/s.
All trajectories matched within each comparison. Whole-model packing took 47 ms in the
fixed-loop run. A separate real-activation timing of each shader alone yielded
0.291 and 0.216 ms (50 warmed synchronized dispatches each, not additive whole-model
latencies). These repeated prompts are not independent samples of a population.
An earlier dedicated prefill-only test found 2.788× paired vs token-loop reference
(9 pairs across 128/160/192; 8 equal greedy outputs per pair). It materialized logits at
the prefill boundary rather than fencing all cache tensors. Do not pool its timings with
the four-way test, or multiply separately measured prefill and decode speedups.

**Receipts.** [Synthetic raw timings](data/synthetic_m3max.csv);
[historical 2×2 raw trials](data/integration_2x2.csv) (durations in seconds, custom calls
counted by stage); [parity maxima](data/integration_parity.csv) (per shape/condition);
[decode controls](data/decode_controls.csv) (counted tokens divided by seconds);
[individual dispatches](data/dispatch.csv) (ms);
[earlier prefill pairs](data/prefill_only_pairs.csv) (seconds).
The digit in `same_tokens` is a comparison result, not a token ID.
The repository's [MIT license](../../LICENSE) applies to this example. The rotational
scan and chassis already in this repository are unchanged; no new recurrence API,
package release, weights, model integration or production deployment is implied.
