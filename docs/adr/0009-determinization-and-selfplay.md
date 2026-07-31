# ADR 0009: Determinization, and the self-play pipeline

## Status
Accepted.

## Context
Two things stood between the search and a trainable network: the search was
reading hidden information, and there was nowhere to put training samples.

## The information leak

`Player` owns the whole piece queue, including the pieces beyond the preview.
The search copies a `Player` per node, so a tree deeper than `preview_count`
was planning against the *true* future:
* preview_count = 5
* player can see: S O J Z I
* search actually had: S O J Z I L Z L O T


Spec §3.2 and §18.3 forbid exactly this. It is also the most damaging kind of
bug for a self-play project, because it does not crash or fail a test — it
quietly inflates the value of setups that only work because the engine peeked,
and the resulting policy collapses the moment it faces a real opponent.

## Decision: determinize at the root

`Player::determinize(seed)` discards everything the player cannot legitimately
see and regenerates it. Two properties are preserved:

* **The visible preview is untouched.** Determinization must not change what the
  player is looking at.
* **The bag is preserved.** A 7-bag is public knowledge and a human counts it,
  so a resample that ignored the bag would be *less* informed than a person. The
  discarded pieces are returned to the bag and it is reshuffled. An earlier
  version skipped the return step and produced three O and one L inside a
  14-piece window — a measurable violation of the bag guarantee.

`SearchConfig::determinizations > 1` repeats the search over independently
sampled futures and averages the visit distributions, which is the particle form
of the chance node in spec §11.3.

Measured cost: **19.1 ms → 19.4 ms** for a 64-simulation search. Effectively
free, because resampling is a queue operation rather than a per-node one.

## Decision: samples store tokens, not states

`TrainingSample` holds the tokenized observation and the action embeddings
rather than a `Player`. This keeps samples compact, decouples the on-disk format
from simulator internals, and — most importantly — makes it structurally
impossible for a learner to read state that the observation mask removed.

Every sample carries `ruleset_hash` and `model_version`, and `ReplayBuffer` can
drop foreign rulesets, because spec §14 lists mixing rulesets without an
identifier as a forbidden augmentation.

## Decision: the reward is the game result, nothing else

Spec §12.3 is emphatic that adding a garbage penalty to the reward prevents the
bot from ever learning a sensible non-cancel. `GameRecorder` enforces this
structurally: `outcome` is set solely from win/draw/loss, while attack and
garbage are written to separate auxiliary fields.

A related detail: a game that survives to the piece limit is scored as a **draw,
not a win**. It did not beat anything, and rewarding it would teach the bot that
stalling is optimal.

## Consequences

The pipeline now runs end to end — self-play → samples → buffer → training batch
— at ~157 placements/s with a 16-simulation search on this 2-core CPU.

Attaching a TetraFormer is now an `Evaluator` implementation and nothing else:
the search, the sample format, the buffer and the token layout are all fixed and
tested against baselines whose behaviour is known.