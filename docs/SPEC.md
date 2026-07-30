# TETR.IO向け自己対戦型Transformer Bot仕様書

**文書名:** TetraZero / TetraFormer（仮称）
**版:** v0.1
**対象:** オフラインシミュレータ、研究用カスタム環境、許可されたBot対戦環境

> **運用上の注意**
> TETR.IOの公開ルールでは、Bot・マクロ・解答支援ツール等による競技上の優位獲得は禁止されています。また、メインゲームAPIは書面による明示的な許可なしには使用できません。したがって、本仕様の標準実装は**ローカルシミュレータ専用**とし、実サービスへ接続するアダプタは分離・無効化します。公開環境で運用する場合は、事前にTETR.IO運営から明示的な許可を取得してください。
>
> 実装上の遵守状況は [`POLICY.md`](POLICY.md) を参照。

---

## 1. 概要

本システムは、Leela Chess Zero型の以下の構成を対戦テトリスへ適用する。

1. 正確なルールシミュレータ
2. 合法手生成器
3. Transformer型Policy–Value Network
4. MCTSまたはGumbel Search
5. 自己対戦による強化学習
6. 過去モデルを含むOpponent League
7. 探索結果から軽量Policyへの蒸留

対戦テトリスにはチェスとの重要な相違がある。

- 交互手番ではなく**同時進行**
- 行動に所要時間がある
- NEXT以降やお邪魔穴などに確率性がある
- 相殺、着弾、せり上がりが時間に依存する
- 相手の速度と攻撃タイミングが勝率に直結する
- 同じ最終配置でも、回転経路や固定タイミングで結果が異なり得る

したがって、最終的には**部分観測・確率的・イベント駆動型の同時対戦ゲーム**として実装する。

---

## 2. 設計上の結論

### 2.1 推奨方式

> 正確なイベント駆動シミュレータ
> ＋ Encoder-only Transformer
> ＋ 可変長合法手Policy Head
> ＋ WDL Value Head
> ＋ Gumbel/PUCT探索
> ＋ 自己対戦

ルールが既知で高速に計算できるため、初期段階ではMuZeroのような学習済み世界モデルではなく、**AlphaZero型の正確なモデルベース探索**を採用する。

MuZero系は、スクリーンショットしか取得できない、内部状態が観測できない、完全な遷移モデルが構築できない場合にのみ追加する。

### 2.2 Decision Transformerの位置付け

単独ではLeela型の探索器ではないため、次の用途に限定する。

- 人間リプレイによる事前学習
- 自己対戦系列からのPolicy蒸留
- 相手の次行動予測
- プレイスタイル条件付きPolicy
- 長期的な積み方の表現学習

---

## 3. 目的と非目的

### 3.1 目的

- 盤面のみから高品質な配置を選択する
- HOLD、NEXT、B2B、Combo、Spin、Garbageを判断に含める
- 着弾時刻と攻撃時刻を考慮する
- 意図的に相殺しない、または送受信のタイミングをずらす判断を学習する
- 相手盤面から次の攻撃規模と攻撃時刻を予測する
- ルール変更やカスタムルールに対応する
- 自己対戦のみでも段階的に強くなる
- Policy-onlyから高計算量MCTSまで、同一モデルで動作する

### 3.2 非目的

- TETR.IOクライアントへの無許可接続
- 公開ランク戦での使用
- DOM、内部メモリ、非公開APIからの情報取得
- 相手から見えないNEXTや乱数状態の利用
- 初期版における1フレーム単位の直接キー入力学習

---

## 4. システム構成

```text
                        ┌─────────────────────┐
                        │  Ruleset Registry   │
                        └──────────┬──────────┘
                                   │
┌─────────────┐   ┌────────────────▼───────────────┐
│ Observation │──▶│ Exact Event-Driven Simulator   │
└─────────────┘   │ + Legal Placement Generator    │
                  └───────────────┬────────────────┘
                                  │ legal actions
                  ┌───────────────▼────────────────┐
                  │ TetraFormer Policy–Value Net   │
                  │ ・Board Adapter                │
                  │ ・Queue/Garbage Adapter        │
                  │ ・Event History Adapter        │
                  │ ・Opponent Adapter             │
                  │ ・Global Transformer           │
                  │ ・Policy/Value/Aux Heads       │
                  └───────────────┬────────────────┘
                                  │ priors, value
                  ┌───────────────▼────────────────┐
                  │ Gumbel Search / PUCT MCTS      │
                  │ + Opponent Rollout             │
                  └───────────────┬────────────────┘
                                  │ placement action
                  ┌───────────────▼────────────────┐
                  │ Low-level Input Controller     │
                  │ placement → canonical inputs   │
                  └────────────────────────────────┘
```

学習系:

```text
Champion Model
      │
      ▼
Self-play Actors ──▶ Replay Buffer ──▶ Learner
      ▲                                  │
      │                                  ▼
Opponent League ◀──── Arena/Gating ◀─ Candidate Model
```

---

## 5. ゲームモデル

### 5.1 数理的な扱い

内部環境をイベント駆動型の部分観測確率ゲームとして定義する。

- 内部完全状態: `S_t`
- Botが見られる観測: `o_t = Ω(S_t, M)`
- マクロ行動: `a_t`
- 行動所要時間: `Δt(a_t)`
- 報酬: `r_t`
- 次イベント状態: `S_{t+Δt}`

`M`は観測マスクであり、対戦環境で実際に見える情報だけを許可する。

### 5.2 時間単位

内部計算は浮動小数点秒ではなく、整数フレーム／整数マイクロ秒／サーバー同期Tickのいずれかを使用する。すべての攻撃、固定、着弾、Garbage activationはイベントキューで処理する。

---

## 6. RulesetConfig

ロジックをハードコードせず、全ルールをバージョン付き設定として管理する。

```text
RulesetConfig
├── id
├── version
├── geometry (width, internal_height, visible_height)
├── randomizer (type, preview_count, hold_enabled)
├── movement (kick_table, allow_180, gravity, lock_delay, reset_limit, DAS, ARR, SDF)
├── clear_rules (spin_detection, mini_detection, all_clear, line_clear_delay)
├── attack (base_attack_table, combo_table, b2b_mode, charge_or_surge, multiplier, rounding_mode)
├── garbage (travel_time, activation_rule, blocking_mode, cancellation_rule, cap,
│            messiness, hole_change_rule, entry_mode, passthrough_mode)
└── targeting
```

すべてのリプレイ、学習サンプル、モデルに`ruleset_hash`を保存する。

---

## 7. 状態表現

### 7.1 完全な内部状態

```text
GameState
├── timestamp
├── ruleset_hash
├── public_event_queue
├── hidden_rng_state        # シミュレータ専用。モデル入力禁止
├── players[]
└── round_status

PlayerState
├── field (occupancy_mask[H], garbage_mask[H])
├── active_piece (type, x, y, rotation, last_action, last_kick, lock_timer)
├── input_controller_state (held_keys, das_charge, previous_direction)
├── hold_piece / hold_used
├── visible_next[] / bag_belief
├── combo_state / b2b_or_charge_state
├── pending_garbage[] / attack_accumulator
└── alive / topout_state
```

### 7.2 Garbageエントリ

```text
GarbageEntry
├── lines, sent_at, arrival_at, activation_at
├── cancellable, tankable
├── hole_known, hole_column
├── source_player
└── rule_metadata
```

`hole_column`など、プレイヤーから見えない値は観測へ含めない。

### 7.3 履歴

直近16〜64イベントを保持する（piece_spawn, piece_lock, line_clear, attack_sent, attack_cancelled, garbage_arrived, garbage_raised, topout）。相手のPPS、攻撃周期、意図的な待機、カウンター準備を推定するために使用する。

---

## 8. 行動空間

### 8.1 基本方針

初期版ではキー入力を直接出力せず、**1ミノの最終固定結果**をマクロ行動とする。

```text
PlacementAction
├── use_hold, final_piece, final_x, final_rotation
├── lock_outcome_id, path_class
├── execution_profile
└── delay_bin
```

### 8.2 合法手生成

現在の操作状態からBFSまたはDijkstra探索を行い、到達可能な固定結果を列挙する。探索状態にはミノ座標・回転、経過フレーム、Lock Delay、DAS charge、押下中キー、**最終操作が回転か**、使用されたkick、HOLD使用状況を含める。

### 8.3 同値な入力経路の統合

以下がすべて同一の場合のみ同じ行動として統合する：固定後盤面、消去行、Spin/Mini分類、攻撃結果、相殺結果、所要時間bin、次ミノ出現時のコントローラ状態。各行動には`canonical_input_sequence`を付与する。

### 8.4 タイミング行動

```text
delay_bin ∈ { FASTEST, +1F, +2F, +4F, +8F, WAIT_FOR_EVENT }
```

`WAIT_FOR_EVENT`はGarbage activation／相手の次固定／Lock Delay上限／最大待機時間までに制限する。

---

## 9. Transformerアーキテクチャ

### 9.1 全体

```text
各モダリティのAdapter → Global Fusion Transformer
        ├── Legal Action Policy Head
        ├── WDL Value Head
        ├── Time-to-KO Head
        ├── Opponent Intent Head
        └── Auxiliary Prediction Heads
```

### 9.2 Board Adapter

セルをそのままGlobal Transformerへ入れると2盤面で数百〜千トークンとなり、MCTS中の反復推論コストが高い。したがって階層型構造を使用する：セル埋め込み → 軽量Axial Attention（縦・横）→ Row/Column Tokenへpool → Global Transformer。

1盤面あたりのGlobal Tokenは `H` Row Tokens + `W` Column Tokens + 1 Board Summary Token。

### 9.3 その他の入力トークン

Active / Hold / Next / Garbage / Counter / Event / Rule / Time / Opponent / Missing。

### 9.4 位置表現

固定の1次元位置埋め込みだけでなく、Attention biasとして row_delta / column_delta / same_board / same_player / modality_pair / time_delta を与える。

### 9.5 Transformer本体

**TetraFormer-S**: board local width 96, axial blocks 2, global width 256, layers 8, heads 8, FFN 768, Pre-Norm RMSNorm, SwiGLU, context 128〜256 tokens, 約10〜20M params

**TetraFormer-M**: board local width 192, axial blocks 4, global width 512, layers 16, heads 16, FFN 1536〜2048, 約50〜100M params

---

## 10. 出力Head

### 10.1 可変長Policy Head

固定サイズ分類ではなく、合法手ごとにAction Queryを作成する。

```text
ActionEmbedding
├── piece, use_hold, final_x / rotation
├── locked_cell_footprint, changed_rows
├── clear_descriptor, spin_descriptor
├── elapsed_time, cancellation_result
└── resulting_surface summary
```

各Action Queryが状態トークンへCross-Attentionし、`p(a|s) = softmax(Score(q_a, H_s))`。

利点：合法手数が可変でも扱える／カスタム盤面幅に対応／特殊kick経路を表現できる／同座標でも異なるSpin経路を区別／不合法手を学習で覚えさせる必要がない。

### 10.2 Value Head

主ValueはWDL分布（P(win), P(draw), P(loss)）。追加Head: expected_time_to_ko, topout_within_4/8_pieces, expected_net_attack_1s/2s, expected_received_garbage, opponent_topout_probability。

### 10.3 Opponent Intent Head

next_lock_eta_distribution, next_attack_size_distribution, clear_type_distribution, hold_probability, deliberate_delay_probability, topout_risk。表現学習用の補助損失と、探索中の相手行動サンプリングの両方に使用する。

---

## 11. 探索

### 11.1 単盤面・初期版

合法配置を辺、固定後盤面をノードとするMCTS。PUCT:

```
a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]
```

探索深さは8〜20ミノ程度。葉ノードはバッチ推論、Transposition Table使用、状態ハッシュにRuleset・時刻・Bag beliefを含める。

### 11.2 Gumbel Search

低シミュレーション数でPolicy改善を安定させるため、ルートではGumbel Sequential Halvingを推奨。

| モード | シミュレーション数 |
|---|---:|
| Policy-only | 0 |
| 軽量リアルタイム | 16〜64 |
| 標準 | 128〜256 |
| 解析 | 800〜3200 |

### 11.3 確率事象

可視NEXT以降のBag、未確定のお邪魔穴、Garbage messiness、確率的rounding、観測誤差、相手のPolicyはChance NodeまたはParticleとして扱う。

### 11.4 1対1同時進行探索

1. 自分の候補行動を展開
2. Opponent Intent Headから相手行動と所要時間をサンプル
3. 先に完了する固定イベントまでシミュレート
4. 攻撃送信、相殺、着弾イベントを処理
5. 次の意思決定イベントへ進む
6. 相手サンプルについて期待値を平均する

強敵への耐性を高める場合は、下位10〜25%のValue／CVaR／maximin／複数Opponent Model混合を選択可能とする。

---

## 12. 相殺外しの学習仕様

「相殺外し」を広く次の行動として扱う：今すぐ相殺せずGarbageを受ける／攻撃を先送りして相手の攻撃後に送る／相手にGarbageを受けさせるため攻撃時刻をずらす／綺麗なお邪魔をあえて盤面資源として利用する／カウンタースパイクを優先する／相殺量より相手の致死タイミングを優先する。

### 12.1 必須入力

Pending garbage量、到着・activationまでの時間、自分の次攻撃までの予測時間、相手盤面の高さと穴、相手の攻撃準備状況、相手の次固定ETA、自分がGarbageを受けた後の生存確率。

### 12.2 必須行動

最速固定、数フレーム遅延、Garbage activation待ち、攻撃を伴わない整地、HOLDによる攻撃順序変更。

### 12.3 報酬上の注意

`受けたGarbage量`を単純な負報酬にすると、Botは合理的な相殺外しを学習できない。したがって主報酬は勝敗のみとする（Win +1 / Draw 0 / Loss -1）。Garbage、攻撃、盤面高などは原則として**補助予測対象**にし、報酬へ直接加算しない。Dense Rewardを使う場合はPotential-based shapingとし、訓練後半で0へ減衰させる。

---

## 13. 学習方式

### 13.1 第1段階: 教師あり事前学習

合法的に取得したデータ（自分で保存したリプレイ、明示的な許可を得たプレイヤーのリプレイ、ローカルシミュレータ生成の対局、Beam Search／既存Heuristic Botの棋譜）を用いる。TETR.IOのメインAPIや非公開エンドポイントから自動収集しない。

`L_BC = -Σ_a y(a) log p_θ(a|s)`

### 13.2 第2段階: 単盤面カリキュラム

積み上げ生存 → 穴を増やさない整地 → 固定Garbageの掘削 → 攻撃効率 → B2B/Combo/Spin維持 → 時間制約付き配置 → ランダムGarbage。

### 13.3 第3段階: Garbage-aware自己対戦

相手盤面はまだ入力せず、相手からの攻撃を確率過程として生成する（一定周期／バースト／高速小／遅い大／Cheese／Clean／停止と再開）。

### 13.4 第4段階: 1対1自己対戦

Opponent Leagueを使用（current model, champion, historical checkpoints, fast attacker, defensive downstacker, spike-oriented, timing-randomized, heuristic baselines）。対戦相手はPFSP型に、勝率が極端すぎないモデルを優先して選択する。

### 13.5 学習ターゲット

```text
TrainingSample
├── observation, legal_actions[]
├── search_policy π, final_outcome z
├── n_step_return, time_to_terminal
├── opponent_action
├── future_attack_labels, future_garbage_labels
└── model_version, ruleset_hash
```

```
L = λ_π L_policy + λ_v L_WDL + λ_t L_time + λ_o L_opponent
    + λ_g L_garbage + λ_a L_attack + λ_r ‖θ‖²
```

---

## 14. データ拡張

**使用可能:** 盤面左右反転／穴列・x座標・回転・入力方向の同時反転／Self-Opponent交換とValue符号反転／同じ盤面に対する異なる合法入力経路／ルール設定のランダム化／PPS・操作遅延のランダム化／一部モダリティのdropout。

**使用禁止:** ミノ種類の任意置換／観測できないNEXTの追加／結果を変える時間順序の交換／異なるRulesetのサンプルを識別子なしに混在。

> **実装上の注記（M1で判明）:** 左右反転はSRS+では全ピースについて厳密に成立するが、guideline SRSではIピースのkick順序が非対称なため成立しない。またTETR.IOの180 kick表は意図的に非対称であり、180有効時は一部局面で反転同値が崩れる。詳細は `README.md` の「Rule findings」および `tests/test_rotation.cpp` を参照。

---

## 15. 将来のマルチモーダル化

自盤面／相手盤面の2次元構造、NEXT/HOLD系列、Garbage時系列、攻撃イベント列、タイミングスカラー、ルール設定、任意でスクリーンショットを統合する。

### 15.1 Adapter構造

Board / Queue / Garbage / Event / Opponent / Vision / Ruleset Adapter → Fusion Transformer。各モダリティにmodality ID、player ID、timestamp、confidence、missing flagを付加する。

### 15.2 Perceiver型拡張

多数の相手盤面、長いイベント履歴、スクリーンショットを同時に扱う場合は、固定数のlatent tokenへCross-AttentionするPerceiver IO型構造を候補とする。

### 15.3 複数対戦相手

Royale型への拡張では、相手の並び順に依存しないSet Encoderを使用する。各相手にtarget status、targeting count、estimated threat、estimated vulnerability、attack historyを付与する。

### 15.4 画像入力

画像入力は最後の手段とする。Screenshot → ViT/CNN detector → probabilistic structured state → belief particles → TetraFormer。画像認識誤差を直接盤面へ確定せず、複数の候補状態として探索する。

---

## 16. 推論API

```text
DecisionRequest
├── observation, ruleset_hash
├── compute_budget (max_simulations, max_time_ms, max_batch)
├── risk_mode
└── debug_level

DecisionResponse
├── selected_action_id, canonical_input_sequence
├── value_wdl, expected_time_to_ko
├── candidates[] (action_id, policy, visits, q_value)
├── principal_variation[]
└── diagnostics
```

デバッグ出力には、選択理由に寄与したトークン、自盤面と相手盤面のAttention、相手攻撃ETA、予測される相殺量、意図的Garbage受けの有無、Policy-onlyと探索後Policyの差を含める。

---

## 17. 実装モジュール

```text
/core-rules     正確な盤面・攻撃・Garbage・時間処理
/movegen        合法配置と入力経路の生成
/protocol       Observation/Action/Replayスキーマ
/model          TetraFormer実装
/search         PUCT、Gumbel、Chance Node、Opponent rollout
/controller     配置から入力列への変換
/selfplay       自己対戦worker
/replay-buffer  サンプル保存と優先度管理
/trainer        分散学習
/arena          モデル評価、昇格判定
/replay-tools   リプレイ検証、可視化
/vision         将来の画像認識Adapter
```

推奨技術構成: ルールコア・探索はRustまたはC++／学習はPyTorchまたはJAX／推論形式ONNX／GPU推論TensorRT・ONNX Runtime／データはProtobuf + 圧縮chunk／分散通信はgRPCまたはメッセージキュー。

> 本リポジトリの選択とその理由は [`adr/0001-cpp-instead-of-rust.md`](adr/0001-cpp-instead-of-rust.md) を参照。

---

## 18. テスト仕様

### 18.1 シミュレータ整合性

**最重要要件はモデル性能ではなくルール整合性である。**

同一seed・同一入力で完全再現／各固定後の盤面一致／消去分類一致／Spin・Mini判定一致／攻撃量一致／相殺量一致／Garbage activation一致／Combo・B2B・Charge一致／Topout時刻一致。

### 18.2 合法手生成テスト

短い入力列の全列挙との比較／kickを必要とする配置／180度回転／IRS・IHS／DAS保存／Lock Delay限界／同一盤面・異なるSpin判定／HOLD後の到達可能性／Garbageせり上がり中の操作。

### 18.3 Property Test

7-bag等の個数制約／盤面外固定の禁止／相殺後Garbage量の非負性／攻撃保存則／同一状態・同一行動の決定性／左右反転の等価性／非表示情報がObservationに混入しないこと。

---

## 19. 評価指標

### 19.1 強さ
Heuristic Botに対する勝率／Policy-onlyに対するSearch Botの勝率／過去checkpointに対するElo・Glicko／未学習Rulesetでの勝率／未知のOpponent styleに対する勝率。

### 19.2 盤面・攻撃
Attack per piece／Net attack per second／Garbage cleared per piece／Hole creation rate／Topout rate／1秒・2秒最大スパイク／B2B維持率／Garbage受け後の生存率。

### 19.3 相殺・タイミング
相殺可能量に対する実相殺率／意図的相殺外し後の勝率／Garbage受けからカウンターまでの時間／相手攻撃ETAの誤差／相手攻撃量予測の交差エントロピー／不要な待機時間／致死攻撃のタイミング精度。

### 19.4 推論性能

- Policy-only GPU p95: 5 ms以下
- 64 simulation p95: 30 ms以下
- 128 simulation p95: 50 ms以下
- バッチ葉評価率: 80%以上
- 同一seedで決定的再現可能

速度条件の異なるBotを比較する場合は、PPSまたは入力遅延を固定する。

---

## 20. モデル昇格条件

1. Championと同じ計算予算で対戦
2. 盤面左右反転・同一piece sequenceのpaired gameを使用
3. 複数Opponent Leagueで評価
4. 既知ルールと未学習ルールの双方で評価
5. 勝率の信頼区間が昇格閾値を超える
6. シミュレータ整合性テストを全通過
7. 特定Botへの極端な退化がない

```text
- Champion直接対戦: 55%以上
- League平均: 52%以上
- 最悪Opponent群: 45%以上
- Simulator regression: 0件
```

---

## 21. 開発マイルストーン

- **M0: ルールコア** — 盤面、ミノ生成、HOLD/NEXT、回転・kick、消去、攻撃、Garbage、リプレイ再生、Ruleset versioning
- **M1: 単盤面Policy** — 合法配置生成、Row/Column Tokenizer、Transformer Policy、教師あり学習、Policy-only対局
- **M2: Leela型探索** — Policy–Value Network、PUCT/Gumbel Search、自己対戦、Replay Buffer、Candidate gating
- **M3: Garbage-aware** — Pending garbage token、イベント時刻、行動duration、相殺と着弾、意図的Garbage受け
- **M4: 相手盤面対応** — Opponent Board Adapter、Opponent Intent Head、1対1イベント駆動探索、Opponent League、相殺外し評価
- **M5: マルチモーダル** — Event Transformer、Perceiver型Fusion、複数相手、画像Adapter、観測不確実性

進捗は [`ROADMAP.md`](ROADMAP.md) を参照。

---

## 22. 主なリスクと対策

| リスク | 対策 |
|---|---|
| ルール再現の微妙な誤差 | リプレイ差分テスト、Ruleset hash |
| 行動空間の爆発 | 最終配置単位、経路同値化、delay bin |
| 同時進行探索の爆発 | 相手行動サンプリング、Top-K、CVaR |
| 最新モデル同士の過学習 | Historical League、複数スタイル |
| 相殺を過度に優先 | 勝敗報酬を主としGarbage罰を避ける |
| 速度だけで勝つBotになる | PPS・入力遅延を固定した評価 |
| Transformer推論が重い | Axial Adapter、token pooling、蒸留 |
| 相手の隠し情報を学習する | Observation mask、情報漏洩テスト |
| ルール変更でモデルが壊れる | Rule token、domain randomization |
| 画像誤認識 | belief particleとconfidence token |
| 公開環境での不正利用 | 接続Adapter分離、標準ビルドでは無効 |

---

## 23. MVPとして推奨する最小構成

```text
正確なローカルルールシミュレータ
+ 1ミノ単位の合法配置生成
+ 自盤面/HOLD/NEXT/Pending Garbage
+ Row/Column Token Transformer
+ 可変長合法手Policy Head
+ WDL Value Head
+ 64〜128回Gumbel Search
+ 自己対戦
```

その後、行動所要時間 → Garbage arrival/activation → 意図的な待機 → 相手盤面 → 相手攻撃ETA → Opponent-conditioned Search → 画像・複数相手 の順に拡張する。

**最も重要な設計判断は、相殺外しを専用のルールベース機能として実装しないこと**である。時間、相手盤面、Garbage queueを正しく状態へ含め、勝敗を主目的に自己対戦させることで、「相殺する」「受ける」「遅らせる」を同じ行動最適化問題として学習させる。
