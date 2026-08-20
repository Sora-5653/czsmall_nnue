# ADR 0001: rule coreはRustではなくC++で実装する

## 状態

採用。言語選択は現在も有効です。

**追補:** 当時はC++17を採用しましたが、Cobra integration後の現行 `Makefile` はC++23を要求します。したがって「C++を使う」という判断は維持しつつ、現行build standardはC++23です。

## 背景

Spec §17はrule coreとsearchの実装言語としてRustまたはC++を推奨しており、当初はRustを第一候補にしていました。

しかし初期開発sandboxではRust toolchainを取得できませんでした。`static.rust-lang.org`、`crates.io`、`index.crates.io`、GitHub release assetへ到達できず、利用可能なdistribution package、container registry、mirrorもありませんでした。一方でPyPI、npm、GitHub API/codeload endpointは利用できました。

この状態でRustを選ぶと、**一度もcompileもtestもしていないrule coreを出荷する**ことになります。これはM0/M1をrobustにするという目的、とくにspec §18.1の「rule consistencyを最重要要件とする」という方針に反します。

## 決定

rule coreをC++で実装します。

当初は、spec §17が明示的に許容していたC++17と、sandboxに存在した `g++ 12` を使いました。外部dependencyを持たず、standard compilerだけでbuildできる構造を維持します。

後にvendored Cobra backendがC++23を要求するようになったため、現行compiler modeは `-std=c++23` へ更新されています。これはRust/C++という本ADRの根本判断を変更するものではありません。

## 帰結

- 初期段階から実際にfull test suiteを実行でき、`-Werror`、AddressSanitizer、UBSanで検証できました。
- 当時の初期suiteは117 tests・約518k assertionsであり、static reviewだけでは見逃した複数の実bugを検出しました。この数値は**当時のhistorical measurement**で、現在のtest countではありません。
- `cargo` を使わず、header dependency trackingを持つhand-written `Makefile` を採用しました。header-heavy projectとしては十分な構成です。
- 将来Rustへ移行する場合も、C++ implementationをreference oracleとして残し、property/determinism testを両実装へ流してdiffできます。
- language-level memory safetyは保証されないため、sanitizerをoptionalではなくCI requirementとして扱います。
- 現在はCobraの要件に合わせてC++23へ移行しています。現行環境の要件は `README.md` と `docs/SETUP.md` を参照してください。
