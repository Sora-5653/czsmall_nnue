# Third-party notices

## cobra-movegen

The move-generation backend vendored under `include/cobra/src` is from [Kixenon/cobra-movegen](https://github.com/Kixenon/cobra-movegen), commit `3da68bce69d0563bb0f51d39b43de07bdd9c431e`.

It is distributed under the Apache License, Version 2.0. The complete license text is preserved in `third_party/cobra-movegen/LICENSE`.

The vendored headers are retained as source files from the upstream project. The Tetra integration in `src/movegen.cpp` adapts their board, piece, ruleset, and path APIs to this project's existing action and timing interfaces.
