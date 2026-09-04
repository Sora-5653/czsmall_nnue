# ADR 0014: Treat network capacity and search as a joint compute-budget problem

Status: accepted (2026-08-16), experimental evidence provisional

## Context

The TetraFormer roadmap has considered larger Transformer variants and, eventually, MoE-style specialization. That direction implicitly assumes that additional network capacity is a scarce and useful resource: if a larger network produces substantially better policy/value estimates, conditional computation could buy more capacity without paying the full dense inference cost.

That assumption had not been tested against the competing use of the same inference budget: running a smaller network more often inside search.

This matters especially for Tetris. The simulator is exact and relatively cheap, many immediate consequences of an action are known rather than predicted, and previous architecture ablations already showed that static held-out losses can disagree with Arena strength once search is enabled. It is therefore unsafe to use parameter count, held-out policy loss, or raw forward throughput as a proxy for the final engine optimum.

The question for this ADR is not "are parameters useful?" or "is search useful?" in isolation. It is:

> Under the current data, value quality, search implementation, and RX 9070 XT runtime, where should the next unit of inference compute go: a larger network or more search?

This experiment is deliberately local. It does not claim to identify the final model size for a mature self-play system.

## Experiment

### Fixed data and training protocol

`trainer/run_size_ablation.py` trains three plain TetraFormer variants while holding constant:

- the same 47,693-sample Gen-4 corpus;
- the same leakage-safe hashed game split: 37,231 train / 10,462 validation samples;
- the exact sampled minibatch schedule within a training seed;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- batch 256;
- 400 optimizer updates;
- policy/value/aux weights `1.0 / 1.0 / 0.1`;
- timing losses off;
- all input, action, target, and search contracts.

Two independent training seeds, 42 and 1337, were run.

The size ladder is approximately logarithmic:

| size | width | layers | heads | FFN | parameters |
|---|---:|---:|---:|---:|---:|
| XS | 64 | 2 | 4 | 192 | 132,744 |
| M | 128 | 4 | 4 | 384 | 946,920 |
| S | 256 | 8 | 8 | 768 | 7,175,592 |

S is the existing TetraFormer-S architecture. XS -> S is about a 54x parameter increase.

### Held-out result

| training seed | XS policy | M policy | S policy | XS value | M value | S value |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 3.308810 | 3.312117 | **3.302256** | 0.694144 | 0.693571 | 0.693175 |
| 1337 | 3.315064 | 3.305652 | **3.301711** | 0.694858 | 0.692887 | 0.692998 |

The larger S model is the most consistent policy imitator, but the gain is small relative to the parameter increase. On seed 42 the XS-to-S policy-loss difference is about 0.0066; on seed 1337 it is about 0.0134.

The value path is a more serious warning. Validation value accuracy is around 0.51-0.52 and scalar MSE is about 1.0 across these runs. These checkpoints therefore do not provide evidence for a well-calibrated search leaf value. Deeper or broader search cannot be assumed to improve monotonically when it consumes this value estimate.

Training cost also scales much faster than the held-out gain:

| training seed | XS seconds | M seconds | S seconds |
|---|---:|---:|---:|
| 42 | 21.5 | 36.5 | 103.8 |
| 1337 | 16.4 | 27.1 | 88.0 |

Peak allocated VRAM was approximately 0.68 / 1.70 / 6.12 GiB for XS / M / S.

### Forward latency

On the RX 9070 XT, fp16, maximum production tensor shapes, batch 16, representative forward latency was approximately:

| size | ms / batch | positions / second |
|---|---:|---:|
| XS | 2.97 | 5.4k |
| M | 4.4 | 3.6k |
| S | 7.8 | 2.0k |

This is an important negative result for naive parameter accounting: 54x fewer parameters does not make XS 54x faster. Kernel launch cost, memory movement, tensor preparation, batching, and other fixed costs dominate much of the runtime.

It is also not valid to turn the batch-16 throughput ratio directly into a search simulation ratio. Doing so initially mapped S/32 simulations to XS/about-83 simulations. In Arena, that XS setting consumed about 74.9 seconds of GPU inference versus about 21.7 seconds in the matched 32-vs-32 trial. The raw forward benchmark was therefore rejected as the equal-compute calibration.

### Search-runtime calibration

The LC3-style streaming queue was measured under actual four-game, 32-simulation search. Gumbel search showed typical per-game request batches around five positions before cross-game aggregation; PUCT produced larger effective batches. The actual search server therefore operates in a materially different regime from a synthetic full batch-16 forward benchmark.

For Gumbel at 32 simulations, four games and seed 4242, approximate GPU-inference cost per played piece was:

| size | GPU inference / played piece | relative simulations for S=1.0 |
|---|---:|---:|
| XS | 0.042 s | about 1.30x |
| M | 0.042 s | about 1.32x |
| S | 0.055 s | 1.00x |

For PUCT at 32 simulations, the analogous empirical ratios were approximately XS 1.49x and M 1.32x relative to S.

These ratios are still only an approximation to equal neural-inference cost, not an exact equal wall-clock contract. Game length is policy-dependent and tree/engine overhead remains outside this ratio. A future time-budgeted search mode would provide a cleaner final comparison.

## Arena results

All serious Gumbel 32-simulation comparisons keep the current production-style settings: timing off, one determinization, policy temperature 1.0, Gumbel `c_scale=0.01`, noise scale `0.05`, paired mirrored games.

### Policy-only

Across four game-seed blocks (42, 1337, 777, 5150), one paired block per seed:

| training seed | comparison | record | score rate |
|---|---|---:|---:|
| 42 | XS vs S | 8-8 | 50.0% |
| 42 | M vs S | 8-8 | 50.0% |
| 1337 | XS vs S | 9-7 | 56.2% |
| 1337 | M vs S | 4-12 | 25.0% |

This is too small for a definitive ranking, but it shows that the held-out policy-loss advantage of S is not automatically a comparable raw-play advantage. Initialization sensitivity is substantial, especially for M.

### Gumbel, S at 32 simulations

For XS versus S, using four independent game-seed blocks for each of the two training seeds:

| training seed | XS sims | S sims | record | score rate |
|---|---:|---:|---:|---:|
| 42 | 32 | 32 | 9-7 | 56.2% |
| 42 | 42 | 32 | 12-4 | 75.0% |
| 1337 | 32 | 32 | 6-10 | 37.5% |
| 1337 | 42 | 32 | 7-9 | 43.8% |

Pooling only corresponding conditions across the two independent training seeds gives:

- XS 32 vs S 32: **15-17 over 32 games = 46.9%**, Wilson 95% CI about 30.9%-63.6%;
- XS 42 vs S 32: **19-13 over 32 games = 59.4%**, Wilson 95% CI about 42.3%-74.5%.

The compute-shift direction is therefore encouraging: in both training seeds, giving XS some of its inference saving back as additional search improved its aggregate result. But the interval still includes parity, and one training initialization remained below S. This is evidence for continuing the compute-allocation experiment, not evidence that XS has replaced S.

M did not show the same gain in the small Gumbel screen: over seeds 42 and 1337, M 32 vs S 32 was 4-4, while M 42 vs S 32 was 3-5.

### Search budget is not monotone

The strongest counterexample to "more search is always better" came from both Gumbel and PUCT experiments.

For Gumbel, arbitrary simulation counts are themselves a confound. The implementation starts with up to 16 root candidates and performs sequential halving. A budget of 16 completes one full pass over 16 candidates, while a budget of 21 completes that pass and then visits only five of the next eight survivors before the budget ends. In the seed-42 experiment, XS 16 vs S 16 scored 6-2, while XS 21 vs S 16 scored 0-8. This result must **not** be interpreted as a general law that five extra simulations are harmful; 21 is an uneven sequential-halving cutoff.

This exposed a new experimental requirement: Gumbel budget comparisons must either use stage-compatible budgets or explicitly redesign the allocator to avoid partial-round artifacts.

PUCT was used as a smoother budget-response diagnostic. It still did not support monotonic search scaling:

| S budget | candidate | same-sim result | empirical-inference-scaled result |
|---:|---|---:|---:|
| 8 | XS | 5-3 at 8 | 4-4 at 12 |
| 8 | M | 7-1 at 8 | 4-4 at 11 |
| 16 | XS | 4-4 at 16 | **6-2 at 24** |
| 16 | M | 3-5 at 16 | 4-4 at 21 |
| 32 | XS | 5-3 at 32 | 3-5 at 48 |
| 32 | M | 6-2 at 32 | 2-6 at 42 |

Each cell above is only eight games and has a wide interval, so the individual records are diagnostic rather than promotion evidence. The qualitative point is stronger than any cell: even after removing the Gumbel stage-boundary issue, more simulations did not produce a monotone strength curve.

Given the weak value metrics in these fixed-data checkpoints, the plausible working hypothesis is that additional tree expansion can amplify leaf-value error or policy/value mismatch. That hypothesis must be tested directly; it is not treated as proven by this experiment.

## Decision

### 1. Do not make parameter count a primary optimization target

TetraFormer development will not assume that a larger dense model is stronger enough to justify its inference cost. Parameter count is a capacity descriptor, not a project objective.

A new network architecture must be evaluated jointly on:

- Arena strength;
- inference latency under the actual search batching regime;
- search budget affordable at that latency;
- training cost and VRAM;
- held-out policy/value diagnostics;
- stability across training and Arena seeds.

### 2. Defer MoE as a capacity-scaling mechanism

MoE is not rejected permanently. It remains interesting if distinct strategic specialization becomes useful or if a dense network is demonstrably capacity-limited.

However, this experiment gives no current evidence that raw dense capacity is the bottleneck. A 54x parameter increase produces only a small fixed-data policy-loss gain, while much smaller models remain competitive in raw play and some searched settings.

Therefore **do not add MoE merely to increase parameter count now**. Revisit it only after a dense scaling curve shows a meaningful strength gain that cannot be purchased more cheaply through search, data quality, reanalysis, representation, or value improvement.

### 3. Do not replace "bigger network" with "maximum search"

The opposite simplification is also rejected. Search is not a free monotone strength knob in the current system.

Before spending substantially more inference on search, improve or verify:

- WDL/value calibration on search-relevant and off-policy states;
- policy/value compatibility after self-play adaptation or reanalysis;
- Gumbel budget allocation and calibration;
- the shape of the strength-vs-budget curve at more than one training seed.

### 4. Keep S as the reference; keep XS as a first-class fast-search baseline

TetraFormer-S remains the architecture/control reference. Nothing in this ADR promotes XS, changes the Gen14 champion, or changes production promotion rules.

XS is retained as a mandatory compute-efficiency baseline because it is cheap to train, much cheaper to infer, and competitive enough to expose whether added capacity is actually buying game strength.

M is retained as an optional intermediate point, but this experiment does not show a clean advantage for it.

### 5. Compare on the Pareto frontier

Future architecture and search work should report a frontier over at least:

- network parameters;
- measured search-runtime inference cost;
- simulations or nodes actually run;
- end-to-end move/search wall time when available;
- Arena score with confidence interval.

The target is not maximum model size or maximum node count. The target is maximum strength subject to the intended latency/compute envelope.

### 6. Preserve the existing training programme

This decision does not change:

- the frozen Gen14 Arena champion;
- the clean-stacking APP > 0.5 timing gate;
- timing being disabled until that gate;
- the adopted non-destructive reanalysis direction;
- the requirement to track VS Score, APM, APP, PPS, survival and tactical clear diagnostics.

Reanalysis may be especially relevant because a stronger or refreshed search teacher can improve policy/value/search consistency without first increasing network capacity.

## Consequences

The immediate architecture roadmap becomes more conservative and more measurable:

1. keep TetraFormer-S as the current control rather than expanding it or adding MoE by default;
2. keep XS beside it as the fast-search control in future ablations;
3. prioritize value/reanalysis/search-calibration work before another large capacity increase;
4. use PUCT for smooth diagnostic search-budget sweeps and Gumbel for production-style checks;
5. for Gumbel, avoid arbitrary budget points that terminate midway through a sequential-halving round unless that allocator behavior is itself the subject of the experiment;
6. treat synthetic forward throughput only as a microbenchmark, never as the final equal-compute conversion;
7. if a future larger or MoE model is proposed, require it to beat the best smaller-model-plus-search point at a matched practical compute/latency envelope.

## Limitations

The current evidence is intentionally not sufficient for a final model-size choice.

- The fixed corpus is an older 47,693-sample Gen-4 teacher distribution, not the current mature self-play distribution.
- Every model received only 400 updates; a larger model may benefit more from additional data or optimization.
- The value head is weak in this experiment, so this is partly a search-under-poor-value study.
- Most Arena cells contain only 8 or 16 games; even the pooled 32-game XS comparison has a wide confidence interval.
- The empirical "equal inference" ratios are calibrated from actual runtime samples but are not exact equal end-to-end wall-clock budgets.
- Search behavior changes trajectory length and therefore observed inference totals.
- The experiment varies model size while keeping architecture family fixed. It does not answer whether a better representation, CNN hybrid, MoE routing, or another search algorithm can shift the frontier.

These limitations are reasons to continue measuring the frontier, not reasons to return to parameter count as a proxy.

## Reproduction and artifacts

Training runner:

- `trainer/run_size_ablation.py`

Search/Arena runner:

- `trainer/run_size_search_arena.py`

Primary results:

- `models/size_search_ablation_20260816/seed42/train_results.json`
- `models/size_search_ablation_20260816/seed1337/train_results.json`
- `models/size_search_ablation_20260816/seed42/policyonly_4seeds.json`
- `models/size_search_ablation_20260816/seed1337/policyonly_4seeds.json`
- `models/size_search_ablation_20260816/seed42/arena_xs_same32_seed42.json`
- `models/size_search_ablation_20260816/seed42/arena_xs_cal32_seed42.json`
- `models/size_search_ablation_20260816/seed42/arena_xs_base32_seed1337.json`
- `models/size_search_ablation_20260816/seed42/arena_xs_base32_seeds777_5150.json`
- `models/size_search_ablation_20260816/seed1337/arena_xs_base32_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed1337/arena_xs_base32_seeds777_5150.json`
- `models/size_search_ablation_20260816/seed42/arena_m_base32_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed42/arena_base8_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed42/arena_base16_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed42/puct_base8_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed42/puct_base16_seeds42_1337.json`
- `models/size_search_ablation_20260816/seed42/puct_base32_seeds42_1337.json`

The failed naive XS~83-vs-S32 throughput conversion is intentionally retained in the result directory as evidence for why raw batch-forward throughput must not be used as the search-compute conversion.
