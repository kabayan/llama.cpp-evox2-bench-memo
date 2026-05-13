# llama.cpp evox2 (Strix Halo) Speculative Decoding ベンチメモ

> Language: [English](README.md) | **日本語** — *README のみ日本語化済み。`results/` 配下の詳細ページは英語のみ*

`llama.cpp` の Speculative Decoding を **AMD Strix Halo (gfx1151) + Vulkan** で動かした実測ノート。中心的な発見は次の通り: **Qwen3.5-27B-Q4_0 target + Qwen3.5-0.8B-Q4_0 draft + `--spec-draft-n-max=4`** の組み合わせで、実用プロンプト (code / chat / reasoning) が **1.49× – 2.05× の speedup** を、**run-to-run variance < 1.04×** という安定性とともに記録した — 同じハードで動く n-gram 系 spec-dec の最高記録を大きく上回る。

本リポジトリは整備されたベンチスイートではなく *lab notebook* として運用している。Phase 1 (結果) / Phase 2 (再現環境: Docker + scripts) / Phase 3 (per-cell full table + raw JSON) / Phase 4 (Qwen3.6-27B-MTP self-speculation) / Phase 5 (Qwen3.6-35B-A3B-MTP — MoE variant) はいずれも push 済み。未測定項目は [results/00-quick-take.md](results/00-quick-take.md) の "What we didn't measure (yet)" 節を参照。

## TL;DR

| 推奨/設定 | Spec-dec 構成 | Qwen3.5-27B-Q4_0 での結果 |
|---|---|---|
| **27B-Q4_0 の default** ⭐ | `--spec-type` (draft model)、`-md Qwen3.5-0.8B-Q4_0.gguf`、`--spec-draft-n-max=4 --spec-draft-n-min=1` | **1.49× – 2.05×** (mean 1.82×)、accept 96-98%、variance < 1.04×。**tg 19.8–27.4 t/s、pp 119–185 t/s** (baseline tg ≈ 13.3 t/s、pp ≈ 220–300 t/s) |
| P_code を最大化する設定 | 同設定で `--spec-draft-n-max=16 --spec-draft-n-min=1` | **P_code で 2.45×** (tg 32.6 t/s、pp 188 t/s)、ただし accept 92-97% (variance ↑、kernel efficiency ↓) |
| **代替: Qwen3.6-27B + MTP self-spec** | target `Qwen3.6-27B-UD-Q4_K_XL.gguf` + `--spec-type mtp --spec-draft-n-max=4` (`-md` 不要) | **1.83× – 2.33×** (mean 2.15×)、accept 54-81%。**tg 21.6–27.7 t/s、pp 106–155 t/s** (baseline tg ≈ 11.8 t/s、pp ≈ 230–350 t/s)。GGUF 1 ファイル完結で外部 draft 不要。[Unsloth の HF model card](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)<sup>[↘](#rel-unsloth-mtp)</sup> は「**~1.5-2× faster generation**」と謳うが、公式設定 K=3 で本実測 **2.13× avg = 主張上限を再現**、speedup は [4096-token generation まで維持・むしろ向上](results/04-mtp.md#does-the-speedup-hold-over-a-512-token-generation) (T=4096 で sweep 最高値 **avg 2.17×**、最悪 windowed 局所でも 1.43×、+20% には到達せず) ([04-mtp.md](results/04-mtp.md)) |
| **Qwen3.6-35B-A3B (MoE) + MTP** | target `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` + `--spec-type mtp --spec-draft-n-max=2` (`-md` 不要) | **1.22× – 1.48×** (mean 1.42×) at K=2 ⭐。**tg 78.2–85.4 t/s、pp 219–313 t/s** (baseline tg ≈ 58.4 t/s、pp ≈ 252–347 t/s)。**Unsloth recipe K=3 は本ハードで suboptimal** (P_chat 1.11× に低下)。[HF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)<sup>[↘](#rel-unsloth-mtp-35b)</sup> は同一の「~1.5-2×」文言だが、**peak が 1.5× に届かず**、claim 下限すら未達。K=8 では全 prompt baseline 倒れ ([05-mtp-moe.md](results/05-mtp-moe.md)) |
| 35B-A3B + 外部 draft は非推奨 | 同 35B target + 外部 0.8B draft + K=4 | gain +11% のみ、P_chat は **0.90×** に slowdown。**tg 52.7–65.2 t/s、pp 222–302 t/s** (MTP self-spec K=2 が上記表示の通り上回る) |
| n-gram 系全般 | `--spec-type ngram-{simple,mod,cache}` | 棄却。best case (ngram-mod on P_code、35B-A3B のみ) で 1.52× だが **variance 1.76×**、chat/reason は flat か悪化 |

1 つだけ覚えるなら: **target = 27B-Q4_0、draft = 0.8B-Q4_0、K = 4、min = 1**。

## なぜ興味深いか

1. **「K=1 がメモリ帯域律速ハードでの上限」という通念は壊れた。** この通念は `lorem ipsum` micro-benchmark に由来する — 反復的なトークン列で draft model が confidence を失い spec-dec round が `p_min` で早期 break するパターン (参照: [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)<sup>[↘](#rel-pr-22673)</sup>、[`common/speculative.cpp:339`](#rel-speculative-cpp))。実プロンプトでは draft 0.8B が K=4 まで 96-100% acceptance を維持するため K↑ ceiling が **そもそも発動せず**、1 round 当たりのトークン yield が伸び続ける。
2. **Strix Halo の Vulkan batched-verify kernel は *部分的に* (完全にではなく) 非効率。** kernel efficiency は K=1 で 75% / K=16 で 47% まで落ちるが、**K=4 でも 68%** を保つ。K=4 が sweet spot なのはここが理由で、K=1 (DLS-045 の論点) や K=16 (raw speedup は最大だが accept と kernel が両方劣化) ではない。
3. **35B-A3B (MoE) は構造的に挙動が異なる。** baseline tg (~58 t/s) の時点で 256 GB/s 帯域の 56% を消費しており、active 3B parameters とは言え既に memory-bound。accept された各 round で 0.8B draft の forward pass を 1 度走らせるコストが spec-dec の gain をほぼ食い潰す。結論: **spec-dec speedup は baseline tg が遅いことに強く依存する**。

## 計測内容

- **ハードウェア**: AMD Strix Halo、gfx1151、256 GB/s LPDDR5X (Vulkan path、ROCm flash-attn off — [llama.cpp issue #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](#rel-vulkan-flash-attn)</sup> 参照)
- **llama.cpp build**: `am17an:mtp-clean` head SHA `5d5f1b46` (PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)<sup>[↘](#rel-pr-22673)</sup>) — Qwen3.5 系 hybrid linear+full attention model に必要な checkpoint ベースの spec-dec path を含む
- **Quants**: Qwen3.5-27B-Q4_0 (target、15.7 GB)、Qwen3.5-0.8B/2B/4B-Q4_0 (drafts)、Qwen3.6-35B-A3B-UD-Q6_K (追加 target、28 GB)
- **プロンプト**: 3 つの実用プロンプトを固定 (Python `binary_search`、3 日間の京都旅行プラン、2 列車の相対運動問題)、`max_tokens=512`、`temp=0`、`ctx=16384`、chat-template は `--jinja --reasoning-format auto` 経由
- **方法論**: cell ごとに warmup + 3 measurement runs、cell 間で server restart (`pkill llama-server` → `/v1/models` を poll)、シーケンシャル実行 (run 中の GPU 共有なし)

## 結果ファイル

- **[results/00-quick-take.md](results/00-quick-take.md)** — K-sweep と draft size sweep の表を含む 1 ページサマリー
- **[results/01-headline.md](results/01-headline.md)** — K=4 推奨の根拠、kernel efficiency curve、35B-A3B との対比
- **[results/02-context.md](results/02-context.md)** — ハード/ソフト構成と本実測に固有な前提
- **[results/03-full-tables.md](results/03-full-tables.md)** — 00/01 に登場する全 cell の per-cell full table (raw data link 付き)
- **[results/04-mtp.md](results/04-mtp.md)** — Qwen3.6-27B-MTP self-speculation の K-sweep (MTP head は GGUF 内蔵、外部 draft 不要) + [length-dependence 検証](results/04-mtp.md#does-the-speedup-hold-over-a-512-token-generation) max_tokens 32 → 4096 全範囲 (「最初だけ速く平均 +20%」反論への追試結果、T=4096 で sweep 最高 2.17×、P_chat 長文で逆に高速化)
- **[results/05-mtp-moe.md](results/05-mtp-moe.md)** — Qwen3.6-35B-A3B-MTP K-sweep (MoE variant、K=2 sweet spot、Unsloth recipe K=3 は suboptimal、claim 下限未達)
- **[data/raw/](data/raw/)** — sanitize 済み per-run JSON (10 ファイル、1 ベンチセッション = 1 ファイル)。`scripts/run_bench.py` で再現可能

## Phase 構成 (公開ロードマップ)

| Phase | Status | 内容 |
|---|---|---|
| 1. Results (結果) | ✅ done | README + `results/00..02` + LICENSE |
| 2. Reproduction (再現) | ✅ done | [`docker/`](docker/) (Dockerfile.mtp-vulkan + 2 patches + build/run doc)、[`scripts/`](scripts/) (単一ファイルの `run_bench.py` + sweep recipe) |
| 3. Per-cell tables + raw data | ✅ done | [`results/03-full-tables.md`](results/03-full-tables.md) (per-cell tg/accept/draft_n median) + [`data/raw/`](data/raw/) (sanitize 済み 8 JSON files) |
| 4. MTP self-speculation (Qwen3.6-27B) | ✅ done | [`results/04-mtp.md`](results/04-mtp.md) (MTP head 内蔵の K-sweep、K=3-4 = 2.13-2.15× avg、K=8 で P_chat 0.90× に崩壊) + [`data/raw/specdec_qwen36_27b_mtp_sweep.json`](data/raw/specdec_qwen36_27b_mtp_sweep.json) |
| **5. MTP on MoE (Qwen3.6-35B-A3B)** | ✅ this push | [`results/05-mtp-moe.md`](results/05-mtp-moe.md) (K=2 sweet spot = 1.42× avg、Unsloth recipe K=3 は P_chat suboptimal、K=8 で全 prompt 崩壊) + [`data/raw/specdec_qwen36_35b_a3b_mtp_sweep.json`](data/raw/specdec_qwen36_35b_a3b_mtp_sweep.json) |

## 数値の再現手順

```bash
# 1. イメージを build (~10 min on Strix Halo)
docker build -f docker/Dockerfile.mtp-vulkan -t llama-cpp-evox2-bench .

# 2. GGUF ディレクトリを mount してコンテナ起動
docker run -d --name llama-evox2 --device /dev/dri \
  -v /path/to/gguf:/gguf:ro -p 10001:10001 \
  llama-cpp-evox2-bench

# 3. 推奨設定で実行 (~7 min)
pip install httpx
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.5-27B-Q4_0.gguf \
  --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --draft-n-max 4 --draft-n-min 1 \
  --output bench_k4.json
```

イメージ内部の詳細は [`docker/README.md`](docker/README.md)、sweep recipe (K sweep、draft size sweep、35B-A3B variant) は [`scripts/README.md`](scripts/README.md) を参照。

## 関連リンク

- <a id="rel-pr-22673"></a>llama.cpp Speculative Decoding 上流 PR: https://github.com/ggml-org/llama.cpp/pull/22673 (`am17an:mtp-clean`、2026-05-12 時点で open / unmerged)
- <a id="rel-vulkan-flash-attn"></a>Strix Halo Vulkan flash-attn 関連 note: https://github.com/ggml-org/llama.cpp/issues/12629
- <a id="rel-speculative-cpp"></a>`llama.cpp` `common/speculative.cpp` (`p_min` default 0.75。`lorem ipsum` で K↑ regression を起こす early-break の出元)
- <a id="rel-server-context"></a>`tools/server/server-context.cpp:2480` (`n_min > draft.size()` で round を全破棄するロジック。`--spec-draft-n-min == --spec-draft-n-max` を K≥8 で禁則化する根拠)
- <a id="rel-unsloth-mtp"></a>Unsloth の Qwen3.6-27B-MTP-GGUF release (Phase 4 target): https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF — README は「MTP speculative decoding for ~1.5-2x faster generation」と謳う。Strix Halo Vulkan + `-fa off` での本実測は公式 K=3 で 2.13× avg = 主張上限を再現 (詳細は [results/04-mtp.md](results/04-mtp.md))
- <a id="rel-unsloth-mtp-35b"></a>Unsloth の Qwen3.6-35B-A3B-MTP-GGUF release (Phase 5 target, MoE): https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF — 同一の「~1.5-2× faster generation」文言で同 `--spec-draft-n-max=3` recipe を公開。但し本ハードでは recipe K=3 が 1.33× avg、peak K=2 でも 1.42× = claim 下限すら未達 (詳細は [results/05-mtp-moe.md](results/05-mtp-moe.md))

## License

MIT — [LICENSE](LICENSE) を参照。
