# Training and evaluation protocol

This document is the short operational contract for learning experiments. The
more detailed target-generation plan remains in `SAMPLE_EFFICIENCY_PLAN.md`;
accepted design rationale lives in ADR 0014 and ADR 0015.

## 1. What counts as progress

The project optimizes for stronger play under a controlled compute budget, not
for one offline loss number. Evidence is ranked as follows:

1. **Paired Candidate-vs-Champion Arena performance** at a fixed search budget,
   reported with a confidence interval.
2. **VS Score and combat diagnostics** under the same paired conditions.
3. **APM, APP, PPS, survival, cancellation, and timing statistics** that explain
   *how* the result was obtained.
4. **Held-out policy/value/auxiliary metrics** that diagnose representation and
   training health.

A lower held-out policy loss can be useful without implying a stronger player.
The CNN ablation is the canonical example: the Transformer imitated its
Transformer-derived teacher better, while the CNN produced the stronger and
more stable search player in the relevant Arenas.

## 2. Objective contract

The terminal game result remains the value/reward anchor. The model may learn
additional dense targets, but they are not reward shaping.

Conceptually:

\[
L = L_{\pi} + \lambda_v L_v + \sum_i \lambda_i L_{\mathrm{aux},i}.
\]

The current rules for auxiliary objectives are:

- every target must be derivable from information or future events in the
  recorded trajectory under a defined player perspective;
- `terminated` and `truncated` trajectories must not be conflated;
- invalid future horizons are masked instead of trained as zero;
- auxiliary targets may not read hidden state that policy inference cannot see;
- loss weights are judged by shared-trunk gradient norms/cosines as well as raw
  loss magnitudes;
- an auxiliary loss is retained only if it is neutral or beneficial to policy
  learning and Arena performance under controlled ablation.

## 3. Dense targets

The sample-efficiency program extracts more supervision from each trajectory
without pretending that extra labels create new independent games. Current or
planned target families include:

- future attack over several time/placement horizons;
- future garbage received;
- self/opponent survival or top-out horizons;
- time-to-terminal / discrete time-to-event targets;
- action-conditioned immediate consequences when they can be computed exactly
  from the engine;
- combat summaries such as VS Score as an ablation-gated auxiliary target.

VS Score is first a **reported evaluation metric**. Using it as an auxiliary
prediction target is a separate experiment and must not change the WDL reward or
promotion rule.

## 4. Dataset provenance

Every serious run should make it possible to answer: which code, checkpoint,
ruleset, schema, search settings, and seeds produced this sample?

At minimum record or validate:

- repository commit;
- checkpoint identity/hash;
- ruleset hash;
- dataset/tokenizer/action/aux schema versions;
- search algorithm, simulations, determinizations, and root-noise settings;
- game/seed interval and shard identity;
- termination reason;
- sample count and source generation.

Generated shards are immutable inputs. Do not byte-concatenate datasets or
silently mix incompatible schemas. Colab/Drive/GAS may transport artifacts but
must not become the authority for seeds, labels, or merge semantics.

## 5. Train/validation splits

Split by game, seed, or shard—not by adjacent position. Consecutive seed ranges
must not be assigned by a naive ordered 80/20 split if that makes validation a
proxy for shard identity. A stable hash of the game seed is the preferred simple
split when sources occupy consecutive numeric ranges.

For architecture comparisons, share the exact split and, where practical, the
same sampled minibatch schedule.

## 6. Architecture ablation contract

When comparing Transformer, CNN, hybrid, or future MoE variants, hold fixed or
record:

- training sample set;
- split;
- optimizer and learning-rate schedule;
- update count and batch size;
- model parameter scale;
- training seed(s);
- search budget and Arena seed(s);
- replay mixture and opponent/champion checkpoint;
- wall-clock and accelerator time.

Do not replace the production/reference model because of one seed or one metric.
Architecture changes are first experiments, then Candidate checkpoints, and only
then eligible for Champion promotion.

## 7. Arena report

A useful Arena report contains at least:

| Category | Metrics |
|---|---|
| Promotion | wins/losses/draws, win rate, 95% confidence interval, paired seed protocol |
| Combat | VS Score, APM, APP |
| Speed | PPS |
| Survival | mean survival time, placements to top out |
| Garbage interaction | sent/received/cancelled attack, cancellation efficiency |
| Timing | FASTEST vs delayed-action frequency, WAIT-for-event use, timing-related cancellations |
| Search dependence | raw-policy and searched-policy results where useful |

VS Score is deliberately adjacent to, not above, win rate: a combat metric can
be more informative than APM/APP while still being a proxy.

## 8. Timing / cancellation curriculum

The move generator exposes delay bins including `WAIT_FOR_EVENT`, but timing is
not considered learned until the policy actually explores and uses them
productively.

The staged policy is:

1. first establish competent board tactics such as stable stacking, quads, and
   T-spins;
2. inspect play and combat metrics rather than rely on a single numerical gate;
3. use APP around the flat-stack Quad baseline (~0.5) only as a rough readiness
   signal, not as a promotion threshold;
4. then add or strengthen explicit exploration/data for garbage timing and
   cancellation avoidance (相殺外し);
5. verify the capability through action-frequency and paired Arena diagnostics.

The timing curriculum must not hand-code a positive reward for delaying. Search
and game outcome should determine when waiting is useful.

## 9. Self-play search mixture

Once the loop is closed, prefer a mixture that gets broad coverage cheaply but
still injects high-quality targets:

- mostly shallow-search self-play for volume;
- a smaller fraction of deeper-search games/positions for stronger policy
  targets;
- multiple search strengths or deliberately imperfect positions so the model
  sees recovery states rather than only its current narrow on-policy manifold.

Any position-start or stratified curriculum should be recorded as a separate
source in the dataset manifest so it can be ablated rather than silently mixed.

## 10. Promotion rule

Champion is a protected artifact. Training and experimentation produce
Candidates. A Candidate may replace Champion only through the configured Arena
gate; neither a lower validation loss nor an impressive APM/APP/VS Score is
sufficient by itself.

This distinction lets aggressive experiments proceed without making the
production comparison target drift during the experiment.
