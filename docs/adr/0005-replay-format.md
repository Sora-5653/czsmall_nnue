# ADR 0005: Replay format records inputs, not state

## Status
Accepted.

## Context
Spec §18.1 makes rule consistency the top requirement and §22 lists "replay diff
testing" as the mitigation for subtle rule drift. Spec §17 asks for Protobuf
under `/protocol`. Until now a replay existed only as an in-memory vector of
placements inside one test.

## Decision

**Record inputs, never derived state.** A replay stores the seed, the ruleset
hash, and the sequence of placements. Everything else — boards, attacks, timing
— is recomputed on playback. Storing derived state would make verification a
playback of cached results instead of a test of the simulator.

**Record spin provenance explicitly.** This is the one thing that cannot be
recomputed. A piece slid into a notch and a piece rotated into the same notch
occupy identical cells but score differently, so the final position does not
determine the spin class. The verifier additionally re-derives the spin and
compares it to the recorded value, which is what catches a spin-detection
regression.

**Interleave checkpoints.** Every Nth placement carries a board hash, the
sent/received line counts and the timestamp, so a divergence is reported at the
placement where it first appears rather than at the end. The interval is a
tunable size/locality trade-off.

**Custom binary chunk, not Protobuf (yet).** Explicit little-endian, versioned,
with a trailing FNV-1a checksum. This keeps the project dependency-free, which
matters more right now than wire compatibility; the field layout mirrors what a
`.proto` would declare, so migrating later is mechanical.

## Consequences

- ~19-21 bytes per placement (5.5 KB for a 300-piece game), which is acceptable
  for the volumes self-play will produce.
- `verify_replay()` reports `first_divergence`, plus the expected and actual
  board hashes, so a regression is localised immediately.
- Four distinct failure modes are detected and distinguished: ruleset hash
  mismatch, piece-sequence divergence (randomizer change), checkpoint mismatch
  (rule change), and file corruption (checksum).
- Writing the format immediately exposed a real bug: the first verifier marked
  every replayed piece as "rotated", which fabricated spins the original game
  never scored and diverged at placement 16 of a 53-placement game.

## Alternatives considered

- **Store the board after every placement.** Trivially "verifies" and catches
  nothing; it only proves the file was read correctly.
- **Protobuf now.** Would pull in a toolchain this sandbox cannot fetch, for a
  benefit (cross-language wire compat) nothing currently needs.