# 第三者ソフトウェアに関する通知

## cobra-movegen

`include/cobra/src` にvendorしているmove-generation backendは、[Kixenon/cobra-movegen](https://github.com/Kixenon/cobra-movegen) のcommit `3da68bce69d0563bb0f51d39b43de07bdd9c431e` を基にしています。

Apache License, Version 2.0の条件で配布されています。完全なlicense textは `third_party/cobra-movegen/LICENSE` に原文のまま保存しています。

vendor済みheaderはupstream project由来のsource fileとして保持しています。`src/movegen.cpp` のTetra integrationは、legal placement列挙とpath generation全体にCobraを使用し、legacy/custom-board fallbackは持ちません。adapterはCobraの固定standard field、piece、rule、input pathを、このproject既存のaction / timing interfaceへ変換します。

`include/cobra/src/row/pathfinder.hpp` にはlocal integrationのため2点の調整があります。

1. Tetraがtargetごとに同じCobra searchを繰り返さないよう、all-target traversalを公開しています。
2. GCC template-instantiation compiler errorを避けるため、canonical/search-size constantをcall siteで明示しています。

movement rule自体はそれ以外変更しておらず、元のsingle-target APIもwrapperとして残しています。
