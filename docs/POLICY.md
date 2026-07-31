# Usage policy and scope

This repository implements a **local, offline Tetris simulator and research
codebase**. It is not a TETR.IO client and contains no code that talks to
TETR.IO in any way.

## What this repository is

- An exact, deterministic, event-driven rule core (`/include/tetra`).
- A legal-placement generator over that rule core.
- An observation/tokenizer layer for machine-learning experiments.
- Developer tools that run entirely on the local machine.

## What this repository deliberately is not

Per the specification's operational notes (spec §1, §3.2, §22):

- **No network code.** There is no HTTP client, no WebSocket client, and no
  dependency that provides one. `grep -r` for `socket`, `curl`, or `http` in
  `src/` and `include/` returns nothing.
- **No TETR.IO API access.** The main game API requires explicit written
  permission from the TETR.IO operators. This project does not use it, and no
  adapter for it is provided in the standard build.
- **No DOM/memory/private-endpoint scraping.**
- **No use of information a human player could not see.** Hidden state (the RNG
  state, the unshuffled remainder of the bag, the hole column of garbage that
  has not yet risen) lives only inside the simulator and is stripped by
  `observe()` in `include/tetra/observation.hpp`. This is enforced by tests in
  `tests/test_observation.cpp`.

## Competitive play

TETR.IO's rules prohibit bots, macros, and solver assistance for competitive
advantage. Do not use this code, or anything derived from it, to play on public
TETR.IO servers or in ranked modes. If you intend to run a bot in a public
environment, obtain explicit prior permission from the TETR.IO operators first.

## If a connection adapter is ever added

The specification requires that any such adapter be a **separate, opt-in
module** that is disabled in the standard build. It must not be linked into
`core-rules`, `movegen`, or the default tools.