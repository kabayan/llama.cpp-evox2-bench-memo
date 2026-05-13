# 04 — Qwen3.6-27B MTP self-speculation (built-in draft head)

Phase 1-3 covered the **external-draft** path: a separate draft GGUF (Qwen3.5-0.8B-Q4_0) feeds tokens into the 27B target via `-md`. This page covers an alternative: Qwen3.6-27B-UD-Q4_K_XL ships an **MTP (Multi-Token Prediction) head inside the same GGUF**, and `llama.cpp` can run self-speculation against it via `--spec-type mtp` — no separate draft model required.

## TL;DR

| K (`--spec-draft-n-max`) | P_code | P_chat | P_reason | avg | accept (min-max) | notes |
|---:|---:|---:|---:|---:|---:|---|
| baseline | 11.86 (1.00×) | 11.81 (1.00×) | 11.84 (1.00×) | 1.00× | — | no spec-dec |
| 1 | 19.48 (1.64×) | 18.59 (1.57×) | 19.68 (1.66×) | 1.63× | 84-95% | |
| 2 | 25.63 (2.16×) | 22.34 (1.89×) | 25.28 (2.14×) | 2.06× | 73-93% | |
| **3** | **26.98 (2.28×)** | **21.75 (1.84×)** | **27.05 (2.29×)** | **2.13×** | 60-84% | Unsloth-recipe default |
| **4** ⭐ | **27.65 (2.33×)** | **21.55 (1.83×)** | **27.05 (2.29×)** | **2.15×** | 54-81% | peak avg |
| 5 | 26.50 (2.23×) | 19.17 (1.62×) | 26.73 (2.26×) | 2.04× | 45-74% | |
| 8 | 20.41 (1.72×) | 11.67 (**0.90×** ⚠️) | 18.49 (1.56×) | 1.42× | 27-59% | P_chat collapses below 1× |

**Operational recommendation: K=3 or K=4.** K=3 is Unsloth's published recipe; K=4 picks up a marginal +0.02× on average but eats ~6 percentage points of accept rate. K=5 already loses ground on P_chat; K=8 actively regresses there.

Speedups are vs. the per-session baseline `A0_baseline` (no spec-dec, same hardware/build/prompts). Accept range is the worst-to-best run across the 3 prompts × 3 measure runs.

## Setup difference vs. Phase 1-3

| | Phase 1-3 (external draft) | Phase 4 (MTP self-speculation) |
|---|---|---|
| target GGUF | `unsloth/Qwen3.5-27B-Q4_0.gguf` | [`unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)<sup>[↘](../README.md#rel-unsloth-mtp)</sup> (17.9 GB) |
| draft model | `unsloth/Qwen3.5-0.8B-Q4_0.gguf` (507 MB), passed via `-md` | none — the MTP head is inside the target GGUF |
| llama-server flags | `--spec-type` (default: draft model), `-md draft.gguf`, `--spec-draft-n-max=K --spec-draft-n-min=1` | `--spec-type mtp --spec-draft-n-max=K` |
| disk footprint | 15.2 GB target + 0.5 GB draft = 15.7 GB | 17.9 GB (single file, includes MTP head) |
| compatibility | requires `am17an:mtp-clean` [PR #22673](../README.md#rel-pr-22673) build | same build; the `mtp-clean` branch is also the source of MTP support |

Same hardware (AMD Strix Halo, Vulkan, gfx1151, `-fa off`), same prompts (P_code / P_chat / P_reason), same methodology (warmup + 3 measure runs, median tg, server restarted between cells).

## Comparison to Phase 1-3 external draft

Phase 1-3 numbers were measured on a different target quant (`Qwen3.5-27B-Q4_0`) on a different day, so direct ratio comparison would be muddled. But the **absolute tg ranges line up closely**:

| | tg (t/s) range across 3 prompts | source |
|---|---|---|
| Qwen3.5-27B-Q4_0 (baseline) | 13.27 – 13.32 | [00-quick-take.md](00-quick-take.md) §1 |
| **+ 0.8B draft, K=4** | 19.77 – 27.36 | same |
| Qwen3.6-27B-UD-Q4_K_XL (baseline) | 11.81 – 11.86 | this page baseline |
| **+ MTP head, K=4** | 21.55 – 27.65 | this page C4_mtp_K4 |

In absolute terms, MTP K=4 at 21.55-27.65 t/s and external draft K=4 at 19.77-27.36 t/s land in the same band. The MTP path **does not save tokens/sec** — it saves operator overhead: no second GGUF to download, no draft-vocab compatibility check, no `-md` flag to keep in sync with the target.

The 27B-UD-Q4_K_XL baseline (11.84 t/s) is ~11% slower than Qwen3.5-27B-Q4_0 (13.30 t/s) — UD-Q4_K_XL is a larger quant (17.9 GB vs. 15.2 GB), so each baseline forward pass reads more weights. Whether MTP or external-draft wins depends on whether you weight setup simplicity vs. absolute tg.

## Why P_chat collapses at K=8 (and not at K=4)

The MTP head accepts more aggressively than external 0.8B-Q4_0 at low K (95% at K=1) but degrades faster as K grows:

| K | best accept (P_code or P_reason) | worst accept (P_chat) | spread |
|---:|---:|---:|---:|
| 1 | 95% | 84% | 11 pt |
| 4 | 81% | 54% | 27 pt |
| 8 | 59% | 27% | 32 pt |

At K=8, P_chat's median accept hits 27% — meaning the MTP head proposes 8 tokens but only ~2 stick. The verify pass costs roughly the same as K=4's verify pass (Vulkan batched-verify is sensitive to batch size = K+1, see [01-headline.md §2](01-headline.md#2-why-k4-is-the-operational-answer-and-not-k16)), but each round delivers fewer accepted tokens. The net is a 0.90× *slowdown* on P_chat — the spec-dec mechanism is doing work but losing tokens to it.

This is consistent with the Phase 1-3 finding that P_chat is the bottleneck across spec-dec regimes: conversational text has shorter high-confidence runs than code or reasoning, so any draft model (external 0.8B or built-in MTP head) hits a confidence wall sooner on chat.

The K=4 sweet spot for MTP is structurally similar to K=4 for external 0.8B: both balance accept rate against per-round yield. MTP's accept-rate curve falls off harder than 0.8B-Q4_0's, which is why MTP's K=4 is at the edge of viability rather than (as for 0.8B) a comfortable middle of the curve.

## Unsloth's claims vs. what we measured

The [Qwen3.6-27B-MTP-GGUF model card on Hugging Face](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)<sup>[↘](../README.md#rel-unsloth-mtp)</sup> advertises MTP self-speculation prominently:

> **NEW: MTP speculative decoding for ~1.5-2x faster generation**

Unsloth's published recipe uses `--spec-draft-n-max=3` with `-fa on`. We ran on Vulkan with `-fa off` (the Strix Halo gfx1151 flash-attention path produces wrong outputs — see [02-context.md](02-context.md)) and kept every other knob aligned.

| Unsloth's claim | What we measured | Comment |
|---|---|---|
| ~1.5–2× faster generation | **2.13× avg at K=3** (Unsloth's recommended K), **2.15× avg at K=4** (peak) | Upper end of the claimed range reproduces; K=4 nudges past it. P_code K=4 hits 2.33×, well above the headline range |
| Recipe default K=3 | K=3 = 2.13× avg, K=4 = 2.15× avg | Statistically a tie on the average; K=4 wins on P_code (2.33× vs. 2.28×) and loses ~6 pt of worst-case accept. Treating both as interchangeable is fine |
| K↑ keeps helping (implicit) | K=5 = 2.04×, K=8 = 1.42× (P_chat collapses to 0.90×) | The model card doesn't bound K from above; in our measurement, beyond K=4 the MTP head's joint-prediction confidence falls off fast and K=8 is a runtime regression on chat workloads |
| Single GGUF, no separate draft model | confirmed | The MTP head is loaded from the same file as the target. Server log: `loading MTP head from … (override_arch=qwen35_mtp)` and `set_mtp: MTP draft head registered (ctx_mtp=…, n_ubatch=512, n_embd=5120)` |

**Short version: the headline claim holds.** K=3 (Unsloth's recipe) reproduces the upper end of the ~1.5-2× range; K=4 gives a marginal +0.02× over that on the average. Beyond K=4 the MTP path falls off — chat workloads in particular — which is not something the model card warns about. The K=4 sweet spot here matches the Phase 1-3 finding that external-draft spec-dec on this hardware also tops out at K=4 (for different reasons: kernel batch efficiency on the verify pass).

**One gap we can't close from this sweep**: Unsloth recommends `-fa on`. Our Strix Halo Vulkan build defaults to `-fa off` because of [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>, so we cannot evaluate the recipe's `-fa on` portion on this hardware. Whether `-fa on` changes the K=3 / K=4 trade-off remains an open question — see "What we didn't measure (yet)" in [00-quick-take.md](00-quick-take.md).

## Does the speedup hold over a 512-token generation?

External feedback (2026-05-13): "the 1.5-2× headline only applies to the very start of generation; the long-run average is closer to +20%". We re-ran the sweep with `max_tokens ∈ {32, 64, 128, 256, 512}` *and* recorded streaming-chunk wall-clock timestamps so we can compute cumulative tg at every token position. Same target (`Qwen3.6-27B-UD-Q4_K_XL`), same K=3 (Unsloth recipe), same hardware/build.

**Short version: on this hardware the speedup does not collapse with length.** Every max_tokens setting averages 2.0–2.2× over the three prompts, and the cumulative tg at position 500 is still 1.81–2.31× of baseline. The windowed (50-token bucket) tg dips to 1.40× for one bucket on P_chat near position 250, but recovers to 2.15× by position 500.

### max_tokens sweep — speedup vs. generation length

| max_tokens | P_code (mtp / base) | P_chat | P_reason | **avg** |
|---:|---:|---:|---:|---:|
| 32 | 25.35 / 12.39 = 2.05× | 29.31 / 12.25 = 2.39× | 23.51 / 12.25 = 1.92× | **2.12×** |
| 64 | 27.84 / 12.03 = 2.31× | 24.26 / 12.11 = 2.00× | 25.23 / 12.15 = 2.08× | **2.13×** |
| 128 | 27.88 / 12.07 = 2.31× | 26.28 / 12.03 = 2.18× | 26.91 / 12.00 = 2.24× | **2.24×** |
| 256 | 27.33 / 11.94 = 2.29× | 22.54 / 12.04 = 1.87× | 26.52 / 12.04 = 2.20× | **2.12×** |
| 512 | 26.88 / 12.01 = 2.24× | 21.78 / 12.01 = 1.81× | 27.82 / 12.01 = 2.32× | **2.12×** |

There is no length-dependent collapse. P_chat shows ~+10pp drop from K=32 → K=512, which lands at 1.81×, still well above the claim's lower bound of 1.5×. P_code and P_reason are essentially flat or rise slightly with length.

### Cumulative tg curve at max_tokens=512

For each generated-token position N, we compute the average tg of the first N tokens (tg(N) = N / time elapsed since the first token). If "first only fast" were true, tg(50) would be much higher than tg(500).

| position | P_code | P_chat | P_reason |
|---:|---:|---:|---:|
| 50 | **2.26×** | 1.99× | 1.99× |
| 100 | 2.21× | 2.00× | 2.20× |
| 150 | 2.23× | 2.01× | 2.23× |
| 200 | 2.21× | 1.97× | 2.21× |
| 250 | 2.26× | **1.82×** | 2.20× |
| 300 | 2.21× | 1.77× | 2.27× |
| 350 | 2.24× | 1.79× | 2.30× |
| 400 | 2.21× | 1.79× | 2.32× |
| 450 | 2.24× | 1.78× | 2.32× |
| 500 | **2.24×** | **1.81×** | **2.31×** |

P_code and P_reason are flat or slightly rising with position. P_chat shows a small step-down between position 200 and 300 (1.97 → 1.77) and then stabilises around 1.78–1.81×. Even at its worst position, the cumulative speedup is **+77%, not +20%**.

### Windowed (50-token bucket) tg — local instantaneous speedup

The cumulative curve smooths out local variation. The windowed view shows what's happening in each 50-token slice:

| position | P_code (wind) | P_chat (wind) | P_reason (wind) |
|---:|---:|---:|---:|
| 100 | 2.15× | 2.02× | 2.46× |
| 150 | 2.30× | 2.02× | 2.31× |
| 200 | 2.14× | 1.89× | 2.14× |
| 250 | 2.46× | **1.40×** ⚠️ | 2.14× |
| 300 | 2.01× | 1.55× | **2.68×** |
| 350 | 2.47× | 1.90× | 2.48× |
| 400 | 2.02× | 1.80× | 2.48× |
| 450 | 2.48× | 1.70× | 2.30× |
| 500 | 2.29× | **2.15×** | 2.29× |

P_chat has one bucket (positions 200-250) where the instantaneous speedup drops to 1.40× — likely a transition from the reasoning preamble to the body of the answer, where the MTP head's accept rate falls temporarily. The bucket immediately recovers to 1.55× → 1.90× → 1.80× and ends at 2.15×. P_code and P_reason show no comparable dip.

### Conclusion: where does "+20%" come from?

On Strix Halo + Vulkan + `-fa off` + `Qwen3.6-27B-UD-Q4_K_XL` + K=3, **the speedup never approaches +20% at any granularity** — not on average over 32/64/128/256/512 tokens, not on cumulative tg over a 500-token window, not even on a single 50-token instantaneous bucket (the worst observed bucket is 1.40×, well above 1.2×). Likely sources of the +20% number:

- A different hardware (CPU-only, low-bandwidth dGPU) where the MTP head's accept rate falls faster
- A different quant or model — note that Phase 5's [MoE 35B-A3B](05-mtp-moe.md) lands at peak 1.42× regardless of length, so its averages are closer to +30-40%
- An `n-gram-{simple,mod,cache}` configuration confused for spec-dec generally — Phase 1's [n-gram evaluation](00-quick-take.md#what-we-rejected) does land in the 0.94-1.20× range
- An extremely long generation (max_tokens ≫ 512) where the MTP head's accept rate finally degrades — we didn't measure beyond 512 tokens, this is listed under "what we didn't measure" in [00-quick-take.md](00-quick-take.md)

Within the bounds we measured, the 1.5-2× claim holds end-to-end over a full 512-token generation.

Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep.json) (10 cells × 3 prompts × (warmup + 3 measure runs); each run includes a `chunk_history` array of `[cumulative_gen_tokens, time_since_first_token]` pairs for post-hoc per-position analysis).

## Caveats

- **`-fa on` (Vulkan flash-attn) not measured here.** Unsloth's published recipe uses `-fa on` but our Strix Halo Vulkan path defaults to `-fa off` per [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>. All Phase 1-3 numbers also use `-fa off`, so the comparison stays apples-to-apples — but the absolute speedups might shift with `-fa on`. Re-running this sweep under `-fa on` is the most obvious next experiment.
- **MTP + `--mmproj` not supported.** The target is an image-text-to-text model, but the mtp-clean branch disallows `--mmproj` alongside `--spec-type mtp`. Open upstream constraint, no workaround.
- **One model family tested.** MTP is a per-model architecture feature; the K=4 sweet spot from Qwen3.6-27B may not transfer to other MTP-capable models. As of 2026-05-12 Qwen3.6-35B-A3B does not have an MTP-GGUF release.
- **[PR #22673](../README.md#rel-pr-22673) is unmerged.** The build (`am17an:mtp-clean@5d5f1b46`) is the same as Phase 1-3. MTP self-speculation rides on the same checkpoint-based spec-dec mechanism.

## Raw data

[`data/raw/specdec_qwen36_27b_mtp_sweep.json`](../data/raw/specdec_qwen36_27b_mtp_sweep.json) — 45 KB, 7 cells × 3 prompts × (warmup + 3 measure runs), `--spec-type mtp` with `--spec-draft-n-max` ∈ {1, 2, 3, 4, 5, 8}. Schema matches the Phase 3 spec-dec files (see [`data/raw/README.md`](../data/raw/README.md)).

Cell IDs: `A0_baseline` (no spec), `C{1,2,4,5,8}_mtp_K{1,2,4,5,8}`, plus `B3_mtp_K3` (different prefix is a bench-script artifact; values are comparable).
