# Astra向けエージェント設定の監査

監査日: 2026-09-05。対象は、このリポジトリが管理する指示、エージェント設定、運用手順、CIテンプレートです。

## 変更内容

| 対象 | 確認した問題 | 対応 |
|---|---|---|
| `AGENTS.md` | GPUの詳細手順が常時読み込まれ、セットアップ・Colab文書と重複していた | 共通の作業指針と契約に縮約。GPUコマンドは `GPU_WORKFLOW.md` へ移動 |
| `.codex/config.toml` | 親モデルがSolで、委任手順が長かった | Astraに変更。reasoning effortの `high` とLuna Maxワーカーを維持し、役割と引き渡し条件を短縮 |
| `.codex/agents/worker.toml` | 親側と重なる一般的な作業説明があった | 所有範囲、他者の変更保護、再委任なし、検証報告に集約 |
| スキル | リポジトリ内に `skill.md` / `SKILL.md` がなかった | 変更対象なし。既存文書を複製するスキルは追加せず、必要時に参照する構成を採用 |
| GPU workflow | AGENTS、セットアップ、Colab手順が混在していた | GPU実行例を独立文書へ移し、環境導入とColabは既存文書へ参照を集約 |
| `docs/ci.yml` | 権限、重複実行、実行時間の上限が明示されていなかった | `contents: read`、同一event/refの古いrunのキャンセル、各jobの30分上限を追加 |

`AGENTS.md` はUTF-8換算で9,455 bytesから3,401 bytesへ約64%縮小しました。GPUコマンド例を必要時に読む構成へ移した効果であり、モデルのトークン数や速度を測定した結果ではありません。

## 試験的コンテキスト管理

プロジェクト設定に次を追加しました。

```toml
[features.context_management]
experimental_mode = true
```

公式設定リファレンスでは、単一の要約への反復圧縮に代わり、ノートと検索可能な履歴を使う方式と説明されています。Astra専用の設定とは記載されていません。ChatGPTでのサインインとPlus、Pro、Pro Liteのいずれかが必要です。

Codex CLI 0.153.4で、このリポジトリを作業ディレクトリにして `codex features list` を実行し、`context_management` が `true` になることを確認しました。ChatGPTでのサインインも確認しました。契約プランの適合性、稼働中のデスクトップセッションへの反映、長時間作業での履歴検索と情報保持は未検証です。

このリポジトリの設定を読み込む次のセッションで利用状況を確認してください。無効に戻す場合は、この設定だけを `false` にします。全体設定や独自の圧縮プロンプトは変更していません。

## 検証結果

- Pythonの `tomllib` で両方のTOMLを解析し、Astra/high、Luna/max、ワーカー設定の参照先を確認しました。
- PyYAMLでCIテンプレートを解析し、既存のtriggerと3つのjob、追加した権限、concurrency、時間上限を確認しました。
- 変更したMarkdownのローカルリンクとコードフェンスを確認しました。移動した8個のGPUコマンドブロックは元の内容と一致しています。
- 作業前から未コミットだった27ファイルは、SHA-256で内容が変わっていないことを確認しました。
- 対象ファイルの `git diff --check` は成功しました。文書・設定の変更のため、C++のbuild、GPU学習、GitHub上のCIは実行していません。

## 残る制約

- CIは `docs/ci.yml` のテンプレートです。GitHub Actionsとして実行される場所へは移していません。既存文書に記録されたvendored Cobraの `-Werror` 問題は未解決です。今回の設定整理ではcompilerやthird-party warning policyを変更していません。
- スキルの公式ガイドに従い、名前と説明から必要なスキルを選び、使用時に本文を読む構成を前提とします。ユーザー共通スキルとインストール済みプラグインはリポジトリの管理外であり、変更していません。
- 学習やモデル評価のコード、実験計画、既存の停止条件、Championは変更していません。GPU学習、Arena、モデル性能の比較実験も実行していません。

## 公式資料

- [Astraモデルガイド](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra): 指示の競合、継続条件、説明量、委任、検証範囲の調整。
- [Codex設定リファレンス](https://learn.chatgpt.com/docs/config-file/config-reference): モデル、ワーカー、試験的コンテキスト管理の設定。
- [AGENTS.mdの読み込み](https://learn.chatgpt.com/docs/agent-configuration/agents-md): 指示の探索とサイズ上限。
- [スキルの構成](https://learn.chatgpt.com/docs/build-skills): 必要時に本文を読み込む方式。
- [GitHub Actionsのworkflow構文](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax): 権限、並行実行、jobの時間上限。
- [Googleの文書スタイルガイド](https://developers.google.com/style/highlights): 明確な文章、構成、用語、リンク。
