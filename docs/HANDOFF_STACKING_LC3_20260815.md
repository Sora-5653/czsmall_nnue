# TetraFormer stacking / LC3 runtime handoff — 2026-08-15

This document supersedes `docs/HANDOFF_STACKING_BALANCE_20260815.md` for the current state.

## 0. One-line status

**Gen14 remains the frozen Arena champion; Gen24 remains the best established clean-firepower specialist (3-seed clean mean APP 0.219). Gen39 failed its decisive seed-19000001 clean screen at APP 0.147 with a 231-piece top-out and is rejected. Timing stays OFF until the fixed clean benchmark mean exceeds 0.5 APP. Minimum viable Reanalyse is now implemented and accepted: exact dual-player historical reconstruction, full token/action parity gate, KL-based stale-position selection, selected-root GPU re-search, and non-destructive output/audit manifests. The canonical Gen24 clean corpus passed 5,366/5,366 reconstruction rows and produced 270 refreshed 64->128-simulation rows. The historical 0.424 APP Gen15 replay passed 255/255 rows and produced 13 refreshed 100->200-simulation rows with Gen14 champion; all 13 retained the historical chosen action as the deeper-search best action. This proves the plumbing and identifies stale policy distributions, not a stronger trained model. The controlled original-target vs Reanalyse training ablation is now the primary next step.**

---

## 1. Current hard constraints

Do not reinterpret these in the next session.

1. **Timing is frozen.** Do not resume WAIT_FOR_EVENT/timing-head production work.
2. Timing ablation may resume only after the **fixed multi-seed clean stacking mean APP is > 0.5**.
3. The timing gate is evaluated on the explicit **no-attack-delivery clean benchmark**, not ordinary 1v1 APP.
4. `models/gen14_rank100_100_20260814.best.pt` remains the **Arena/model-strength champion** until a candidate passes the frozen Arena screen convincingly.
5. Firepower specialists are allowed to be teachers/staging checkpoints without champion promotion.
6. Promotion is two-dimensional:
   - clean stacking/firepower must improve;
   - frozen Arena strength must not clearly regress.
7. **APP alone is not sufficient.** For every serious clean candidate, record at least:
   - Quad count/frequency;
   - T-Spin clears, preferably Single/Double/Triple or full/mini when recoverable;
   - B2B continuation count/streak information;
   - attack contribution / APP;
   - trajectory length / survival or early top-out.
8. T-Spin/Quad statistics are diagnostic of *what behaviour is being learned*, not a replacement for the APP gate.
9. Continue using VS Score in competitive evaluation alongside APM/APP/PPS/survival diagnostics.
10. **Reanalyse is adopted.** Do not return to broad random training ablations before testing this direction.
11. Reanalyse must be **non-destructive**: original self-play datasets/trajectory labels stay immutable; refreshed targets are written to a new source with provenance.
12. Reanalyse does **not** relax the timing gate, clean benchmark, or Arena promotion rules.

---

## 2. Model roles

### Arena champion

`models/gen14_rank100_100_20260814.best.pt`

Keep frozen as the strength reference.

### Established firepower specialist

`models/gen24_stack_expertmix020x8_40_20260815.best.pt`

Clean benchmark:

| seed | APP |
|---|---:|
| 19000000 | 0.207 |
| 19000001 | 0.230 |
| 19000002 | 0.220 |
| **mean** | **0.219** |

This is still the best *established/repeatable* clean specialist reference.

Frozen Arena vs Gen14:

- seed 42: 1-3
- seed 1337: 1-3
- pooled: 2-6

Therefore Gen24 is a teacher/specialist, not champion.

### Balance reference

`models/gen25_stack_arena_recover20_20260815.best.pt`

Clean APP:

- 0.200 / 0.193 / 0.130
- mean 0.174

Frozen Arena:

- seed 42: 2-2
- seed 1337: 3-1
- pooled: 5-3

Useful reference for the firepower-vs-competitive tradeoff, but not a promotion.

### Gen38 — rejected but informative

`models/gen38_cleanexpert_elite020x8_40_20260815.best.pt`

Construction:

- parent: Gen24 firepower specialist;
- clean-domain elite self-imitation;
- policy-head-only path;
- 40 update steps;
- clean elite selection used APP >= 0.20 with short/fragile trajectories excluded (minimum 200 samples for the later filtered set).

Clean validation improved consistently from about **3.8700 -> 3.7918**, showing that the model really did fit the clean teacher distribution.

Fixed clean rollout:

| seed | APP |
|---|---:|
| 19000000 | 0.232 |
| 19000001 | 0.110 |
| 19000002 | 0.233 |
| **mean** | **0.192** |

Interpretation:

- seed 0 and seed 2 improved beyond Gen24;
- seed 1 catastrophically moved into a bad search/stacking basin and topped out early;
- the direction is not useless, but **40 steps is too large / unstable**.

Known seed-0 clear behaviour for Gen38 included approximately:

- Quad: 14
- B2B continuation: 2
- T-Spin clear: 0

Gen38 is rejected as a replacement for Gen24.

### Gen39 — rejected after the decisive clean screen

Files:

- `models/gen39_cleanexpert_elite020x8_10_20260815.pt`
- `models/gen39_cleanexpert_elite020x8_10_20260815.best.pt`

Construction is the same clean-elite self-imitation direction as Gen38, but reduced from 40 to **10 steps**.

Clean validation:

- approximately **3.8700 -> 3.8388**

Decisive fixed-seed rollout on seed `19000001`:

- game summary: APP **0.147**, 231 pieces, loss/top-out #2;
- per-player trajectories: APP **0.194** and **0.147**, 232 samples each;
- Quad: **6** total;
- B2B continuations: **0**;
- T-Spin clears: **0** (three zero-line mini-spin actions were observed, but no T-Spin clear);
- no trajectory reached APP 0.20.

This is well below Gen24's seed-1 APP of about 0.230 and repeats the unstable/bad-basin behaviour that the shortened update was intended to avoid. Gen39 is rejected. Seeds `19000000` and `19000002` were intentionally not run, and no Arena compute was spent.

### Immediate next action

Do not Arena-test Gen39 and do not spend the other two fixed seeds on it.

A nominal five-step Gen40 follow-up was considered but deliberately not started. The handoff and checkpoint metadata together establish that Gen39 was a 10-step, policy-head-only update from Gen24 with learning rate `1e-5`, but the checkpoint does not preserve the original batch size or dataset command. More importantly, the old whole-corpus `--elite-app-repeat` path cannot express the documented minimum-200-sample trajectory filter. A guessed command would therefore not be a controlled "steps only" ablation.

Before the next specialist update:

1. preserve an explicit manifest/list of the clean source files;
2. make the APP and minimum-sample selection reproducible through the source-aware sampler or an equivalent explicit filtered dataset path;
3. record batch size, learning rate, seed, reset flags, loss weights and exact parent in a sidecar/run record;
4. then test a smaller update against seed `19000001` first.

---

## 3. Clean elite corpus and behaviour discovered

Gen24 clean no-attack-delivery data has now been expanded through seeds 0-9, giving 20 player trajectories.

Per-player trajectory analysis found:

- APP >= 0.20: **13 trajectories**
- total samples in those trajectories: **3,572**
- Quad: **75**
- B2B continuation: **17**
- T-Spin clears in this elite group: **0**

This is important: the current firepower improvement is clearly a **Quad/B2B stacking skill**, not yet a learned T-Spin system.

Do not claim that T-Spin has been learned merely because the engine supports spins. Keep recording it explicitly until real clean-domain T-Spin clears appear.

A stricter useful teacher filter was:

- reconstructed clean APP >= 0.20
- minimum trajectory length >= 200 samples

With the current `rank_clean_trajectories.py --min-app 0.20 --min-samples 200` implementation this now yields exactly:

- 12 displayed/eligible trajectories
- 3,426 clean teacher samples
- mean teacher APP ~0.220
- 70 Quads
- 15 B2B continuations
- 0 T-Spin clears

The older 8-trajectory / ~2,320-sample figure was an earlier approximation and should no longer be used as the corpus count.

The newly mined seed `19000009` contributed two long eligible trajectories:

| player | samples | APP | Quad | B2B continuation | T-Spin clear |
|---:|---:|---:|---:|---:|---:|
| +1 | 256 | 0.242 | 8 | 1 | 0 |
| -1 | 254 | 0.201 | 4 | 3 | 0 |

The second trajectory contained one zero-line mini-spin action, not a T-Spin clear. The four-clear B2B sequence on that side consisted of consecutive Quads.

Canonical source list for the current seed-0-through-9 corpus:

```text
data/benchmark/gen24_stackbench_seed19000000_20260815.tetradat
data/benchmark/gen24_stackbench_seed19000001_20260815.tetradat
data/benchmark/gen24_stackbench_seed19000002_20260815.tetradat
data/benchmark/gen24_stackbench_seed19000003_20260815.tetradat
data/production/gen24_cleanmine_seed19000004_20260815.tetradat
data/production/gen24_cleanmine_seed19000005_20260815.tetradat
data/production/gen24_cleanmine_seed19000006_20260815.tetradat
data/production/gen24_cleanmine_seed19000007_20260815.tetradat
data/production/gen24_cleanmine_seed19000008_20260815.tetradat
data/production/gen24_cleanmine_seed19000009_20260815.tetradat
```

The current explicit analysis contract is reconstructed APP >= 0.20 and at least 200 samples per player trajectory. Preserve both the source list and this filter in the next run record; do not silently substitute the unfiltered `--elite-app-repeat` population.

The minimum-length filter matters because a short trajectory can have high APP while being strategically fragile or ending in early top-out.

Useful analysis utilities now present:

- `trainer/rank_clean_trajectories.py`
- `trainer/analyze_clear_patterns.py`
- `trainer/inspect_tspin_context.py`

`rank_clean_trajectories.py` reports trajectory APP together with Quad / T-clear / full / mini / B2B information.

---

## 4. What was learned from the failed balance attempts

### Gen37 — source-aware clean elite mixed into Gen14

Source-aware replay was used to keep the competitive primary source and draw the secondary clean source at a controlled ratio. The clean secondary was filtered with APP >= 0.20 and minimum trajectory length.

Cheap policy diagnostics looked superficially reasonable:

- clean chosen-action agreement moved about 0.785 -> 0.822;
- competitive agreement moved about 0.891 -> 0.863.

But actual clean rollout on the fixed seed fell to about **APP 0.158**.

Reject.

Lesson: offline agreement/CE/ranking diagnostics are useful screens but **cannot replace actual search rollout**. Small policy changes can induce nonlinear search behaviour.

### Direct Gen14 <-> Gen24 interpolation

Policy-only checkpoint interpolation was tested and did not produce a behavioural Pareto curve.

On seed 19000001:

- alpha 0.50: APP 0.163
- alpha 0.75: APP 0.146

Search is too nonlinear for weight interpolation to be treated as behavioural interpolation.

### Partial policy-module transplantation

Analysis suggested that Gen24's difference from Gen14 is not simply the final Quad move. The useful divergence appears in the **roughly eight placements preceding a Quad**, i.e. the setup process.

Transplanting policy-attention components could increase setup imitation, but partial Q/K/V/module transplantation was unstable across seeds. Policy attention appears strongly co-adapted.

Do not continue random submodule transplantation as the primary route.

### Future-attack value bonus

Short-horizon attack auxiliary predictions were investigated as a search-value bonus.

- legacy/far-horizon attack correlations were too weak;
- the relatively best 0-1 second attack signal had only moderate rank correlation (~0.40 in the relevant probe);
- adding a small attack-search bonus reduced clean APP in the tested rollout (about 0.200 -> 0.183).

Reject for now. Do not turn attack aux into a reward/value shortcut without substantially better calibration.

---

## 5. Source-aware sampling is ALREADY IMPLEMENTED

The older handoff described this as a possible engineering improvement. It now exists in `trainer/train.py`.

Relevant flags include:

```text
--secondary-source-count
--secondary-source-fraction
--secondary-policy-teacher
--secondary-elite-app-threshold
--secondary-elite-min-samples
--anchor-primary-policy
```

Important behaviour:

- the final N datasets can be designated as the secondary source;
- every training batch can draw an explicit fraction from the secondary source;
- secondary trajectories can be filtered by reconstructed APP;
- a minimum number of trajectory samples can be required;
- primary policy anchoring and a separate secondary policy teacher are available;
- these mechanisms are default-off unless requested.

This is preferable to crude whole-file duplication. Do not go back to raw file-count mixing unless running an explicit ablation.

Also still available:

```text
--elite-app-threshold
--elite-app-repeat
```

for older whole-dataset elite replay experiments.

---

## 6. Clean benchmark protocol

Ordinary 1v1 APP is contaminated by incoming garbage, cancellation and opponent state. The timing restart gate therefore uses no-attack-delivery self-play.

Fixed protocol:

- timing OFF
- 64 simulations
- 2 determinizations
- root noise fraction 0
- policy temperature 1.0
- 300 pieces
- fixed seeds:
  - `19000000`
  - `19000001`
  - `19000002`
- `--no-attack-delivery`
- discrete GPU: `cuda:1` (RX 9070 XT)

Example:

```bash
./.venv-rocm714/Scripts/python.exe trainer/gpu_selfplay.py \
  models/CANDIDATE.pt data/benchmark/CANDIDATE_seed19000001.tetradat \
  --engine build-win/tetra_cli.exe --device cuda:1 \
  --games 1 --pieces 300 --sims 64 --batch 16 --seed 19000001 \
  --model-version 99 --determinizations 2 --root-noise-fraction 0 \
  --policy-temperature 1.0 --precision fp16 --no-attack-delivery
```

The **0.5 APP gate means the mean of this fixed clean protocol**, not an isolated trajectory and not ordinary Arena APP.

---

## 7. Frozen Arena protocol

Only candidates that first pass clean evaluation should spend Arena compute.

Candidate vs:

`models/gen14_rank100_100_20260814.best.pt`

Protocol:

- timing OFF both sides
- Gumbel ON
- 32 simulations
- 1 determinization
- policy T = 1.0
- Gumbel c/noise = 0.01 / 0.05 unless intentionally ablating
- paired mirrored Arena
- seed 42 and seed 1337 first

Record:

- W/L and Wilson/95% confidence information when enough games exist;
- VS Score;
- APM;
- APP;
- PPS;
- average pieces / survival / top-out diagnostics.

A tiny pooled 5-3 is encouraging only; it is not enough for champion promotion.

---

## 8. LC3-inspired runtime architecture — DONE AND PUSHED

User requested borrowing the useful parts of the LC3 engine architecture.

The implementation deliberately borrowed **runtime/search structure**, not LC3-specific game evaluation or reward semantics.

Commit:

```text
04d49ffe Adopt LC3-style search runtime and checkpoint training pipeline
```

Branch:

```text
codex/sample_eff_fix
```

Remote:

```text
origin/codex/sample_eff_fix
```

The remote head was explicitly verified as `04d49ffe` after push.

Design record:

`docs/adr/0013-lc3-search-runtime.md`

### 8.1 Shared streaming inference queue

`trainer/gpu_match.py` now contains a reusable `StreamingInferenceQueue` / telemetry object. `gpu_match.py` and `gpu_selfplay_parallel.py` share the scheduler.

Independent search/game workers submit evaluator requests to one bounded-window queue. Compatible requests are merged into GPU forwards.

Telemetry includes:

- wire requests;
- positions;
- GPU forward/batch count;
- queue wait time;
- deadline flushes;
- maximum positions/request count in a forward;
- target fill ratio.

A two-game RX 9070 XT smoke test observed approximately:

- 92 wire requests
- 230 positions
- 46 GPU forwards
- mean 5.0 positions/forward
- max 8 positions/forward
- fill ratio 0.625 for the target used in that test

A separate `gpu_match.py` smoke test also showed the queue combining requests correctly.

### 8.2 SearchPolicy separation

PUCT and Gumbel edge scoring are separated into `SearchPolicy` instead of being tangled with tree lifecycle.

This is intended to make future search-policy/calibration ablations local and measurable.

### 8.3 Edge-local P / Q / N / N_inflight

`SearchNode` no longer relies on independent parallel vectors for every edge statistic. `SearchEdge` owns:

- prior `P`;
- accumulated value / derived `Q`;
- visit count `N`;
- in-flight/virtual-loss count `N_inflight`;
- child node id.

This is the LC3-style invariant that matters for later asynchronous/streaming search work.

### 8.4 Gather -> Eval -> Backprop split

The old combined evaluation/backup path is structurally split into:

1. Gather leaves / reserve edges;
2. Eval batched leaves;
3. Backpropagate results / release in-flight reservations.

They still run serially inside one C++ search call today. **Do not claim true asynchronous overlap yet.** The point of the refactor is that overlapping Gather(N+1) with Eval(N) can now be implemented without first redesigning edge semantics.

### 8.5 Search telemetry

`SearchTelemetry` records:

- gather attempts;
- gathered leaves;
- selection steps;
- terminal backups;
- depth cutoffs;
- evaluation flushes;
- maximum edge in-flight count.

### 8.6 Intentionally deferred LC3 ideas

Do not implement these merely for architectural purity:

- a separate `NodeRepository` abstraction;
- aggressive transposition-DAG ownership changes;
- asynchronous legal-move-generation workers;
- new reward/value targets copied from another engine;
- timing expansion before the APP gate.

The current node vector + TT already supplies the functional repository/DAG baseline. Use telemetry to justify additional complexity first.

---

## 9. Validation after LC3 refactor

Full C++ test suite:

```text
289 tests
1,055,122 assertions
0 failed
```

New tests cover:

- `SearchPolicy` accounting for in-flight edge work;
- LC3 pipeline telemetry being reported without changing the simulation budget.

Python modules involved in the shared queue passed `py_compile`.

Real GPU smoke tests were run on:

```text
AMD Radeon RX 9070 XT = cuda:1
PyTorch 2.12.0+rocm7.14.0
```

Important hardware reminder:

- `cuda:0` = AMD integrated graphics
- `cuda:1` = RX 9070 XT

Use `--device cuda:1 --require-gpu` where supported for real training/evaluation. Never silently accept CPU fallback for intended GPU jobs.

---

## 10. Git / working-tree state

Current branch:

```text
codex/sample_eff_fix
```

Current pushed commit:

```text
04d49ffe
```

`origin/codex/sample_eff_fix` was verified at the same commit.

The checkout may appear very dirty under ordinary `git status` because many files in the Windows checkout are CRLF while the index is LF. After the commit, this was checked with:

```text
git diff --ignore-space-at-eol --name-only
```

and **no tracked semantic diff remained**.

Therefore:

- do not `git reset --hard` just to make status look clean;
- do not accidentally commit mass line-ending churn;
- when reviewing the current checkout, distinguish CRLF-only differences from semantic changes.

There are also intentional/local untracked items including:

- `.venv-rocm714/`
- build directories
- analysis scripts
- benchmark/experiment artefacts

Do not blindly `git add -A`.

Several useful analysis scripts are currently untracked and should not be deleted merely because they are not in HEAD.

---

## 11. Dataset / schema notes that still matter

`trainer/tetra_dataset.py::Dataset.concatenate` supports safe append-only auxiliary-schema mixing when core contracts match. Older rows are widened and appended aux targets have validity mask 0 rather than fabricated labels.

The current tree has progressed beyond the earlier v2/v3-only state, including later schema/chosen-action work. Do not assume all old `.tetradat` files expose every modern field.

For policy-only historical replay, missing aux labels are acceptable only when correctly masked.

Do not byte-concatenate dataset shards. Use dataset-aware loading/concatenation and preserve schema/manifests.

---

## 12. Reanalyse adoption — MINIMUM VIABLE PATH IMPLEMENTED

Reanalyse is now the preferred next sample-efficiency direction before further broad architecture/training ablations.

### 12.1 What "reanalyse" means here

The historical trajectory/state distribution is retained, but selected positions receive **new search-derived teacher targets from a newer/stronger searcher**. This is distinct from ordinary replay:

- replay trains again on the same `(state, target)`;
- reanalyse trains on the same historical state but recomputes an improved policy/search target.

For this project, the first implementation should refresh:

- `search_policy` / policy target;
- `search_value` and any explicitly search-derived confidence metadata added later.

It must **not** silently replace trajectory-grounded labels:

- terminal W/D/L outcome;
- future attack / garbage / top-out interval targets;
- the historical action actually taken (`chosen_action`) when that field is needed to reconstruct the original trajectory.

If desired, a reanalysed best action must be stored as separate metadata rather than overwriting the behavioural action.

### 12.2 KataGo-style selection principle to borrow

KataGo v1.17 experimentally re-searches a random subset of cheap-search positions after a self-play game, with preference for positions where the cheap search was surprising. The useful principle to borrow is **spend expensive search where the current model/search disagrees most**, rather than full-searching every stored state uniformly.

For TetraFormer, use a two-stage selector:

1. **cheap staleness/surprise screen** over replay candidates, using quantities such as policy KL divergence and value disagreement;
2. **full re-search** only for the selected positions.

A practical initial priority score can combine:

- policy surprise: divergence between the current raw policy prior and an available search target;
- target staleness: divergence between the historical search target and the current raw policy;
- value surprise: absolute difference between raw-network value and search value;
- rare-but-desired behavioural context, used only as a bounded diagnostic/stratification term rather than a reward shortcut.

Do not select solely by APP or Quad count; that would bias the state distribution toward one observed behaviour and risk losing general stacking stability.

### 12.3 First target domain

Start with the **clean stacking specialist path**, not mixed competitive data:

- historical source: reproducible Gen24 clean trajectories and later compatible clean replay;
- initial teacher/searcher: Gen24 or a verified stronger clean successor;
- re-search budget: materially larger than the search budget that created the original target, with the exact simulation count fixed and logged for the ablation;
- write output to a new reanalysed dataset source, preserving parent dataset, checkpoint hash, engine commit, ruleset, seed/move identity, old target and new-target provenance.

This lets Reanalyse act as search distillation: Gen24's stronger/deeper search can teach a child more than the original shallow target even before a stronger network exists. After a child passes the fixed clean screen, that child may become the next reanalysis teacher.

Do **not** mix the Gen14 competitive distribution into the first clean Reanalyse test. First establish whether refreshed clean search targets improve the specialist; Arena integration remains a second stage.

### 12.4 Exact-state reconstruction is an implementation gate

The search must receive the exact simulator state, not merely reconstructed neural-network tokens.

The current rectangular `.tetradat` format stores tokenized observations plus provenance such as game seed, move number and chosen action, while compact replay reconstruction already demonstrates deterministic replay for a restricted single-player path. Two-player self-play/reanalysis must therefore prove exact reconstruction of both players and all search-relevant state (queue/randomizer, hold, clocks/timing state, garbage, attack/B2B/combo state, RNG state where required) before historical re-search is trusted.

Preferred implementation order:

1. add/retain a trajectory/replay provenance stream sufficient to reconstruct the exact dual-player root state;
2. implement deterministic `state_at(game, move)` reconstruction;
3. parity-test reconstructed observations/legal actions against those recorded at generation time;
4. only then run the expensive search and emit refreshed targets.

Do not infer a full simulator state from token tensors alone, and do not fabricate hidden state.

### 12.5 Non-destructive dataset contract

Reanalysed rows should be a separate source/shard rather than an in-place rewrite. Each refreshed row/run should record at least:

- source dataset / trajectory identifier;
- ruleset hash and schema versions;
- game seed, player/trajectory index and move number;
- original generating checkpoint/model version;
- reanalysis teacher checkpoint hash;
- engine commit;
- original and reanalysis search budgets;
- selection score/reason;
- old policy/search value and refreshed policy/search value, or an audit reference to both.

The trainer can then use the existing source-aware sampling path to control the reanalysed fraction explicitly. Reanalyse and ordinary replay remain separate knobs.

### 12.6 Controlled first ablation

The first comparison should alter **only the target source**, not timing/model architecture/reward semantics simultaneously:

- baseline: current Gen24 clean training path with original clean targets;
- treatment: same parent, source list, optimizer/loss settings and update budget, but a controlled fraction of rows comes from reanalysed clean targets;
- evaluate both on fixed clean seeds `19000000`, `19000001`, `19000002`;
- record APP, Quad, B2B, T-Spin clear and survival/top-out;
- only a stable clean improvement earns the frozen Gen14 Arena screen, where VS/APM/APP/PPS and survival are also recorded.

The reanalysis fraction and full-search budget are hyperparameters to sweep narrowly after the plumbing/parity test works; do not guess a large fraction and immediately train a long run.

### 12.7 Why this is preferred now

The current bottleneck is not lack of model capacity evidence. Gen38/Gen39 showed that direct self-imitation can fit the clean teacher distribution while still falling into bad rollout/search basins. Reanalyse attacks a more relevant problem: **the teacher target itself can be improved by stronger/deeper current search before more gradient steps are spent imitating it**.

This does not guarantee an APP gain; search nonlinearity remains the main risk. Therefore actual fixed-seed rollout remains the acceptance test.

### 12.8 Implementation and first historical refresh — completed 2026-08-16

Implemented files:

- `include/tetra/reanalyse.hpp`: exact historical root reconstruction, token/action parity gate, policy KL score, non-destructive row selection;
- `tools/tetra_cli.cpp`: `gpu-reanalyse-protocol`, cheap current-model screen, selected-root search and audit JSONL;
- `trainer/reanalyze.py`: GPU teacher serving, overwrite refusal, atomic output publication, checkpoint/source/output hashes and run manifest;
- `tests/test_reanalyse.cpp`: reconstruction, label-preservation and missing-provenance rejection tests;
- `docs/adr/0015-reanalyse-historical-target-refresh.md`: accepted contract and the rectangular-v4 search-value boundary.

Canonical Gen24 clean run:

- sources: the 10 canonical seed `19000000..19000009` files listed in section 3;
- teacher: `models/gen24_stack_expertmix020x8_40_20260815.best.pt`;
- attack delivery OFF, timing OFF, Gumbel, determinizations 2, root noise 0;
- cheap-screened/reconstructed: **5,366/5,366 rows**;
- selected by top historical-policy KL vs current raw policy: **270 rows** (about 5%);
- search budget: **64 -> 128 simulations**;
- outputs: `data/reanalyze/*_gen24teacher_s128_top05.tetradat` plus `.audit.jsonl` and `.reanalyze.json` sidecars;
- all 10 outputs load, pass sanity check and concatenate through `Dataset.concatenate` to 270 rows with valid chosen actions.

Historical high-APP competitive replay:

- source: `data/production/gen15_rank100prod_seed15082005_20260814.tetradat`;
- historical player `+1`: **125 pieces / 53 attack / APP 0.424 / 8 Quads / 4 B2B continuations / max B2B clear streak 5 / 0 T-Spin clears**;
- opponent player `-1`: 130 pieces / 15 attack / APP 0.115;
- teacher: frozen competitive champion `models/gen14_rank100_100_20260814.best.pt`;
- attack delivery ON, timing OFF, Gumbel, determinizations 2, root noise 0;
- reconstructed: **255/255 rows**; selected **13 rows** (8 from player +1, 5 from player -1);
- search budget: **100 -> 200 simulations**;
- selected rows had historical-vs-current-raw policy KL mean about **1.378** and historical-vs-deep-search policy KL mean about **2.674**;
- despite the distribution shift, deeper-search best action matched historical `chosen_action` in **13/13** selected rows.

Interpretation: the 0.424 APP trajectory is a real, reproducible high-Quad/B2B trajectory and its selected actions are robust under the Gen14 champion's deeper search. Its full policy targets are nevertheless stale enough to be meaningful Reanalyse targets. Do not infer a trained-model APP gain from this; no Reanalyse child has been trained or fixed-seed screened yet.

The rectangular v4 dataset has no independent search-value training column. Refreshed search values are retained losslessly in the audit sidecar while terminal W/D/L stays unchanged in the training shard. Adding a valid-masked search-value column requires a later append-only dataset-version decision; do not reinterpret terminal `value_target`.

---

## 13. Recommended restart sequence for the next session

### Step 1 — verify environment only

Reuse project:

```text
/mnt/c/Users/eddyf/czsmall_nnue
```

Reuse DevSpace workspace if available:

```text
ws_a20042c0d7
```

Verify:

- branch `codex/sample_eff_fix`;
- HEAD / remote `04d49ffe` or a known successor;
- RX 9070 XT is `cuda:1`;
- Timing flags remain OFF.

Do not waste time trying to clean CRLF-only status noise.

### Step 2 — Gen39 seed-1 clean screen — completed / failed

Gen39 was run on `19000001` with the exact fixed clean protocol. It produced APP 0.147 and topped out at 231 pieces. Per-player analysis found APP 0.194 / 0.147, six Quads, no B2B continuation and no T-Spin clear. Gen39 is rejected and must not be Arena-tested.

### Step 3 — implement the minimum viable Reanalyse path — completed

Completed on 2026-08-16. Exact reconstruction and target/audit serialization tests pass, and both canonical clean and historical 0.424 APP data were refreshed as recorded in section 12.8.

1. establish exact dual-player state reconstruction/replay parity for historical self-play positions;
2. add a selector that can rank candidate positions by policy/value surprise without changing reward semantics;
3. re-search only the selected roots with a larger, fixed and logged search budget;
4. write refreshed policy/search-value targets to a separate dataset/shard with full provenance;
5. expose the reanalysed source through the existing source-aware sampler so its batch fraction is explicit and ablatable.

Do not start a long training run until reconstruction parity and target/audit serialization tests pass.

### Step 4 — run a controlled Gen24 clean Reanalyse ablation — next priority

Use the canonical reproducible Gen24 clean corpus first. Keep parent checkpoint, optimizer/loss settings, training update count, clean source list and evaluation seeds fixed. Compare original-target training against a narrow reanalysed-target treatment.

The purpose is to determine whether deeper/current search targets improve rollout behaviour, not merely held-out cross-entropy.

### Step 5 — fixed clean screen, then frozen Arena

A reanalysed child must first materially improve/stabilize clean firepower across seeds `19000000`, `19000001`, `19000002`. Record APP + Quad + B2B + T-Spin + survival/top-out.

Only then run the frozen Gen14 Arena screen at seeds 42 and 1337 and record VS/APM/APP/PPS/survival. Do not promote from a tiny sample.

### Step 6 — continue corpus mining only as supporting work

Gen24 seed `19000009` has been generated and added two APP >= 0.20, minimum-200-sample trajectories. Additional clean mining remains useful, especially long stable trajectories, but it is now supporting data work rather than the primary experimental direction. Preserve exact source lists and filtering contracts.

### Step 7 — continue T-Spin discovery monitoring

Current elite teacher behaviour contains Quad/B2B but zero T-Spin clears. Keep collecting T-Spin statistics and visually inspect representative trajectories when T-Spin counts first become nonzero, to distinguish intentional setup from accidental garbage/context clears.

Timing remains frozen even if cancellation aux improves.

---

## 14. Post-Reanalyse architecture plan — modern MoE

This is an **adopted implementation plan for a later architecture phase**, not the immediate next experiment. The first controlled Reanalyse test must remain architecture-frozen. Do not mix this plan into the initial Reanalyse ablation.

### 14.1 Goal and success criterion

The goal is not to use MoE merely because frontier LLMs use it. The goal is to increase tactical/strategic specialization **without reducing search strength at a fixed inference/search budget**.

The primary end-to-end comparison is therefore:

```text
strength at fixed wall-clock/search budget
```

not parameter count or held-out policy loss alone. Every serious architecture candidate must record at least:

- evaluator batch latency / throughput;
- search nodes or completed simulations per second;
- peak VRAM and active parameter/FLOP estimate;
- held-out policy/value/aux losses;
- fixed clean APP + Quad + B2B + T-Spin + survival/top-out;
- frozen Arena W/L + VS/APM/APP/PPS/survival.

A network that is more accurate per forward but slows search enough to reduce end-to-end strength is a regression.

### 14.2 Baselines and first candidates

Keep a dense Transformer baseline and add MoE behind a modular FFN interface so the architecture can be changed without touching the search/replay semantics.

Planned comparison order:

1. **Dense FFN baseline** — current architecture, unchanged.
2. **DeepSeek-style MoE baseline** — fine-grained routed experts plus always-active shared expert(s), with routing/load balancing separated from task loss as much as practical.
3. **Kimi K3-inspired Stable LatentMoE candidate** — project the normal hidden state from `d_model` into a smaller MoE latent width `d_moe`, run shared + top-k routed experts in that latent space, then project back to `d_model`.

Conceptually:

```text
h[d_model]
  -> latent projection
z[d_moe]
  -> shared expert(s) + top-k routed experts
z'[d_moe]
  -> output projection
h'[d_model]
```

The initial LatentMoE ablation should start around `d_moe ~= d_model / 2`, but the ratio is an ablation variable rather than a fixed architectural truth.

The reason this is especially relevant here is search economics: routed expert specialization is useful only if the active path remains cheap enough that the AlphaZero search loop can still evaluate many positions.

### 14.3 Routing / balancing ablations

Do not bundle all routing changes into one experiment. Compare them independently after a basic MoE path is stable.

Candidate axes:

- shared expert(s) vs no shared expert;
- full-width experts vs latent-width experts;
- number of routed experts;
- top-k active experts;
- DeepSeek-style score/bias balancing;
- DeepSeek V4-style router-affinity refinement / small sequence-wise anti-collapse regularization where applicable;
- Kimi K3-style quantile balancing.

The router should be allowed to become genuinely non-uniform when the data support it. The purpose of balancing is to prevent collapse and pathological hardware imbalance, not to force every tactical state to use every expert equally.

### 14.4 Do not pre-label tactical experts

Do **not** hard-code expert identities such as `T-spin expert`, `downstack expert`, or `spike expert` into the learning objective at first.

Let routing specialization emerge from the task. Log enough information to inspect it afterwards:

- selected expert IDs and router scores/probabilities;
- board/state token class and spatial position;
- game phase / stack height / garbage state;
- windows preceding Quad, B2B, T-Spin, downstack, spike, cancellation and top-out events;
- search disagreement / value uncertainty where available.

This preserves the long-term interpretability goal: expert routing can later be compared with human-recognizable tactical concepts or SAE features without contaminating the initial specialization with hand-authored labels.

### 14.5 Residual architecture is a separate axis

Kimi K3 Attention Residuals and DeepSeek V4 mHC are worth testing later, but **not in the same first MoE ablation**.

Order:

```text
stable Dense vs MoE comparison
    -> settle expert/routing design
    -> standard residual vs AttnRes-inspired path vs mHC-inspired path
```

This keeps causal attribution possible. Long-context-specific attention/KV-cache optimizations are low priority because the Tetris observation sequence is short and the network is used primarily as a batched encoder-style policy/value evaluator, not as a million-token autoregressive decoder.

### 14.6 Implementation sequence

After the controlled Reanalyse phase is complete and a frozen post-Reanalyse baseline exists:

1. refactor `trainer/tetraformer.py` so FFN/MoE is a configuration-selected module with no change to tokenizer, policy/value heads or search semantics;
2. add checkpoint/config versioning for dense vs MoE/LatentMoE variants;
3. implement a minimal shared + top-k routed expert path in PyTorch/ROCm first;
4. add router telemetry and per-batch expert-load diagnostics before large training runs;
5. reproduce dense-baseline parity through the refactored path;
6. run Dense vs DeepSeek-style MoE at controlled active compute;
7. run full-width MoE vs Stable LatentMoE at controlled active compute;
8. ablate balancing/routing only after the basic expert path is stable;
9. only then consider AttnRes/mHC-inspired residual experiments;
10. update `export_weights.py`, model-version/schema handling and C++ inference/parity only for architectures that survive the PyTorch/ROCm screen, rather than complicating the CPU/export path for discarded prototypes.

### 14.7 Experimental controls

For an architecture ablation, keep fixed wherever technically possible:

- training dataset/source mixture and Reanalyse fraction;
- parent/frozen baseline;
- optimizer, update count, batch semantics and seeds;
- tokenizer/observation schema;
- search algorithm and search hyperparameters;
- timing OFF;
- clean benchmark seeds `19000000` / `19000001` / `19000002`;
- frozen Arena reference and seeds.

Report both equal-search-budget and equal-wall-clock comparisons when useful. The wall-clock comparison is the decisive systems metric because MoE changes evaluator cost.

### 14.8 Entry gate

Do not begin this architecture campaign merely because the design is documented. Enter it after:

1. minimum viable Reanalyse is implemented and provenance/reconstruction tests pass;
2. the first controlled Gen24-domain Reanalyse ablation is resolved;
3. a reproducible baseline checkpoint/dataset/search configuration is frozen for architecture comparison.

Until then, this section is the implementation plan, not the current training task.

---

## 15. Things NOT to do next

- Do not resume timing because the code exists.
- Do not promote Gen24, Gen25, Gen38 or Gen39 without the required gates.
- Do not judge a stacking candidate by ordinary 1v1 APP alone.
- Do not use offline policy agreement as the final decision criterion.
- Do not assume parameter interpolation gives behavioural interpolation.
- Do not blindly replay competitive data after a clean improvement; Gen25 showed it can erase the firepower gain.
- Do not turn the weak attack aux into a search reward without new calibration evidence.
- Do not spend large Arena batches on a candidate that already fails clean seed 19000001.
- Do not conflate the new Gather/Eval/Backprop structure with already-implemented true async overlap.
- Do not `git add -A` in the dirty Windows/WSL checkout.
- Do not overwrite historical `.tetradat` files in place during Reanalyse.
- Do not treat tokenized observations as sufficient simulator state for re-search.
- Do not replace the historical `chosen_action` or trajectory-derived outcome/aux labels with a counterfactual reanalysis action/result.
- Do not spend full-search compute uniformly over the entire replay buffer before surprise/staleness selection is benchmarked.
- Do not combine Reanalyse with timing, reward shaping, MoE/major architecture changes, or a large training-schedule change in the first controlled test.

---

## 16. Key files to inspect first

Search/runtime:

- `include/tetra/search.hpp`
- `trainer/gpu_match.py`
- `trainer/gpu_selfplay_parallel.py`
- `docs/adr/0013-lc3-search-runtime.md`

Training/source-aware replay:

- `trainer/train.py`
- `trainer/tetra_dataset.py`
- `trainer/tetraformer.py`
- `include/tetra/dataset.hpp`
- `include/tetra/replay.hpp`
- `include/tetra/replay_buffer.hpp`
- `include/tetra/selfplay.hpp`

Reanalyse is implemented through `include/tetra/reanalyse.hpp`, `tetra_cli gpu-reanalyse-protocol` and `trainer/reanalyze.py`; inspect those entry points together with the files above before changing reconstruction or target semantics. Keep ADR 0015 as the contract.

Clean evaluation / pattern analysis:

- `trainer/gpu_selfplay.py`
- `trainer/rank_clean_trajectories.py`
- `trainer/analyze_clear_patterns.py`
- `trainer/inspect_tspin_context.py`

Arena:

- `trainer/gpu_arena.py`
- `tools/tetra_cli.cpp`

Previous context:

- `docs/HANDOFF_STACKING_BALANCE_20260815.md`
- `docs/GEN12_SEARCH_CALIBRATION_PROGRESS_20260812.md`

---

## 17. Final short handoff for another agent

**Keep Gen14 frozen as Arena champion. Gen24 remains the established clean specialist at 0.219 mean APP; Gen39 remains rejected. Minimum viable Reanalyse is implemented under ADR 0015. Exact reconstruction passed all 5,366 canonical Gen24 clean rows and refreshed the top 270 rows at 64->128 simulations into separate `data/reanalyze/*_gen24teacher_s128_top05` sources with audit/manifests. The historical Gen15 seed15082005 player +1 trajectory is confirmed at 125 pieces, 53 attack, APP 0.424, 8 Quads, 4 B2B continuations and 0 T-Spin clears; all 255 game rows reconstructed, and the top 13 stale rows refreshed at 100->200 simulations with Gen14 champion. Those 13 kept the same best action but have substantially changed policy distributions, so the replay is a valid robust-behaviour/stale-target example. This is not yet a strength result. Next, run a controlled Gen24 original-target vs narrow Reanalyse-source training ablation with identical parent/source list/optimizer/loss/update count, then gate both on clean seeds 19000000/1/2 before any frozen Gen14 Arena. Timing and MoE remain OFF. Tests: normal 289/1,055,122/0 plus Reanalyse 3/17/0.**
