# Third-party notices

## cobra-movegen

The move-generation backend vendored under `include/cobra/src` is from [Kixenon/cobra-movegen](https://github.com/Kixenon/cobra-movegen), commit `3da68bce69d0563bb0f51d39b43de07bdd9c431e`.

It is distributed under the Apache License, Version 2.0. The complete license text is preserved in `third_party/cobra-movegen/LICENSE`.

The vendored headers are retained as source files from the upstream project. The Tetra integration in `src/movegen.cpp` uses Cobra for the complete legal-placement and path-generation pass; there is no legacy/custom-board fallback. The adapter maps Cobra's fixed standard field, pieces, rules, and input paths to this project's existing action and timing interfaces.

`include/cobra/src/row/pathfinder.hpp` contains two local integration adjustments: it exposes an all-target traversal so Tetra does not rerun the same Cobra search for every target, and the canonical/search-size constants are spelled out at the call site to avoid a GCC template-instantiation compiler error. The movement rules are otherwise unchanged; the original single-target API remains as a wrapper.
