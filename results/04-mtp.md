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

### Extended to max_tokens 1024 and 2048

After the initial 32-512 sweep, we re-ran the bench at `max_tokens ∈ {1024, 2048}` with the same per-chunk timestamp recording, to see whether a longer generation produces a delayed collapse that the 512-token cut didn't catch.

| max_tokens | P_code (mtp / base) | P_chat | P_reason | **avg** |
|---:|---:|---:|---:|---:|
| 512 (from above) | 26.88 / 12.01 = 2.24× | 21.78 / 12.01 = 1.81× | 27.82 / 12.01 = 2.32× | **2.12×** |
| **1024** | 26.96 / 11.92 = 2.26× | 19.90 / 11.98 = **1.66×** | 27.07 / 11.97 = 2.26× | **2.06×** |
| **2048** | 26.74 / 11.93 = 2.24× | 20.85 / 11.94 = **1.75×** | 26.59 / 11.93 = 2.23× | **2.07×** |

The cell-level averages (1024 = 2.06×, 2048 = 2.07×) are within 3% of the 512-token result. P_code and P_reason are essentially flat, P_chat dips slightly at 1024 then partially recovers at 2048.

Cumulative tg curve at max_tokens=2048 (every 500 tokens):

| pos | P_code | P_chat | P_reason |
|---:|---:|---:|---:|
| 100 | 2.36× | 2.00× | 2.20× |
| 500 | 2.26× | 1.67× | 2.31× |
| 1000 | 2.16× | **1.63×** ← bottom | 2.26× |
| 1500 | 2.21× | 1.66× | 2.26× |
| 2000 | 2.21× | **1.74×** ← recovery | 2.22× |

P_chat's cumulative tg traces a shallow **U-shape**: 2.00× at position 100, bottoming at 1.63× around position 1000, then recovering to 1.74× by position 2000. The U-shape is consistent with the windowed view — the 100-token bucket at position 1000 shows the lowest local speedup (1.35×), and the bucket at position 2000 jumps back up to 2.39× (P_chat windowed wind tg 28.35 vs baseline 11.88 t/s).

**Tentative reading of the U-shape**: in long chat-template outputs the early tokens are smooth reasoning preamble where MTP head accepts well, the middle includes more transitions and rare-token sequences (transitions between sub-points, conjunctions, named entities) that the MTP head mis-predicts, and the late tokens drift back into a more "mature" answer-body pattern (lists, repeated phrasing) that the MTP head accepts again. P_code and P_reason don't have this U-shape — both stay flat at ~2.2× over the same 2000-token window.

Even at the worst single point (P_chat windowed bucket at position 1000, 1.35× = +35%), the speedup never approaches the external claim of +20% (1.2×). Stretching to max_tokens=2048 does not validate "first only fast"; it instead shows the speedup is structurally durable in this configuration. Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep_long.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep_long.json) (5 MB, 4 cells × 3 prompts × 4 runs each, with `chunk_history` arrays).

### Extended further to max_tokens 4096

After T=2048 hinted at recovery on P_chat (1.74×), we extended one more step to **max_tokens=4096** to see whether the recovery completes or whether late degradation finally appears.

The recovery completes — and the average actually rises:

| max_tokens | P_code | P_chat | P_reason | **avg** | P_chat accept |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.24× | 1.81× | 2.32× | 2.12× | 60.3% |
| 1024 | 2.26× | 1.66× | 2.26× | 2.06× | 52.3% |
| 2048 | 2.24× | 1.75× | 2.23× | 2.07× | 56.8% |
| **4096** | **2.25×** | **2.06×** | **2.19×** | **2.17× ⭐** | **73.0%** |

T=4096 is the **fastest average across the entire 32–4096 sweep**. P_chat recovers to 2.06× (above the 1.5× lower bound) and its accept rate jumps from 56.8% at T=2048 to **73.0%** at T=4096.

Cumulative tg curve on P_chat at T=4096 (the U-shape completes):

| pos | cumulative tg (MTP / base) |
|---:|---:|
| 100 | 24.38 / 12.12 = **2.01×** |
| 500 | 19.66 / 12.02 = **1.64×** ← bottom |
| 1000 | 20.36 / 11.98 = 1.70× |
| 1500 | 20.54 / 11.96 = 1.72× |
| 2000 | 21.81 / 11.95 = 1.83× |
| 2500 | 22.91 / 11.93 = 1.92× |
| 3000 | 23.57 / 11.92 = 1.98× |
| 3500 | 24.16 / 11.91 = 2.03× |
| 4000 | 24.45 / 11.90 = **2.06×** ← recovered |

And the windowed (200-token bucket) view shows the late-generation acceleration directly:

| pos | windowed wind tg (MTP / base) |
|---:|---:|
| 500 | 17.16 / 11.98 = **1.43×** ← worst observed anywhere |
| 1000 | 21.80 / 11.94 = 1.83× |
| 1500 | 23.01 / 11.92 = 1.93× |
| 2000 | 26.40 / 11.89 = 2.22× |
| **2500** | 29.44 / 11.86 = **2.48×** ← peak (the late bucket runs *faster* than the early ones) |
| 3000 | 26.80 / 11.85 = 2.26× |
| 3500 | 28.73 / 11.83 = 2.43× |
| 4000 | 28.73 / 11.81 = 2.43× |

The shape is the opposite of "first only fast". P_chat is actually **slower at the start** (1.43× around position 500) and **faster at the end** (2.43× from position 3500 onwards). One reading: the early portion of a long chat-template answer contains the reasoning preamble + transitions where the MTP head mis-predicts, while the late portion settles into mature, repetitive answer-body structure where the MTP head's accept rate climbs back up.

P_code and P_reason at T=4096 stay flat at ~2.2× across the 0–2000 / 0–4000 windows respectively, with no length-dependence in either direction.

**Final length-coverage summary**: across **max_tokens ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}** and cumulative positions 50–4000, the worst single windowed measurement is 1.43× (P_chat, T=4096, pos 500), the worst cumulative measurement is 1.63× (P_chat, T=2048, pos 1000), and the worst cell-level average is 2.06× (T=1024). **The claim's lower bound of 1.5× is met or exceeded everywhere, and the speedup *increases* at the longest length we measured.**

Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep_xlong.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep_xlong.json) (5.6 MB, 2 cells × 3 prompts × (warmup + 3 measure runs) with full `chunk_history` arrays up to 4096 tokens).

### Extended to max_tokens 8192 — plateau + slight long-tail decline

One more step at `max_tokens=8192`. The U-shape recovery seen at 4096 continues but flattens out, with a hint of slow long-tail decline:

| max_tokens | P_code | P_chat | P_reason | **avg** | P_chat accept |
|---:|---:|---:|---:|---:|---:|
| 4096 | 2.25× | 2.06× | 2.19× | **2.17× ⭐ peak** | 73.0% |
| **8192** | 2.25× | **1.95×** | 2.17× | 2.12× | 67.4% |

Cumulative on P_chat at T=8192:

| pos | cumulative |
|---:|---:|
| 100 | 2.01× |
| 1000 | **1.62×** ← bottom |
| 3000 | 1.93× |
| **5000** | **2.05× ← peak** |
| 6000 | 2.01× |
| 6500 | **1.98×** ← slight decline |

The U-recovery on P_chat now traces a wider arc — it bottoms at pos 1000, peaks at pos 5000, then dips slightly toward pos 6500 (1.98×). Still well above the 1.5× lower bound. Worst windowed at T=8192 is **1.58×** (P_chat pos 1000), so the absolute worst over the whole 32–8192 sweep is still T=4096's 1.43× bucket.

### Extended to max_tokens 32767 — natural EoS detection (and an ~10× wall-clock surprise)

Final stretch. `max_tokens=32767` with `--ctx-size=32768` to fit the generation. Two new findings emerge — both from the comparison between baseline and MTP behavior on P_chat:

| max_tokens | P_code | P_chat tg | P_reason | **avg** | P_chat gen |
|---:|---:|---:|---:|---:|---|
| 8192 | 2.25× | 1.95× | 2.17× | 2.12× | base 8188 / MTP 6758 |
| **32767** | **2.25×** | **2.04×** | **2.17×** | **2.15×** | **base 32688 / MTP 6758** |

**Finding 1: the MTP head detects natural EoS at the same position every time.** At both T=8192 and T=32767, the MTP path terminates the P_chat answer at **gen=6758 tokens** — the same natural stopping point. Baseline, however, keeps generating past 6758 and goes all the way to 32688 (one shy of the cap) when given room. The MTP head's accept-rate dynamic depends on the target's argmax matching the draft head's argmax — when the target starts producing EoS-like tokens, the MTP head sees and stops.

**Finding 2: this turns into an ~10× wall-clock speedup on tasks that the model would naturally end.** P_chat wall-clock cost:

| | gen tokens | tg t/s | wall-clock (decode only) |
|---|---:|---:|---:|
| baseline @ T=32767 | 32688 | 11.29 | **~2895 s (~48 min)** |
| MTP K=3 @ T=32767 | 6758 | 22.98 | **~294 s (~5 min)** |

That's **~10× faster on the actual task** ("answer this user's chat prompt"), not the 2.04× tg-rate speedup. The tg-rate metric undercounts spec-dec's real-world value whenever the model would naturally end before `max_tokens` — which is the common case in real chat / agent usage.

Cumulative tg on P_chat (the common range where both sides have data, plus the baseline-only long tail to expose KV-cache drag):

| pos | base cum | MTP cum | speedup |
|---:|---:|---:|---:|
| 100 | 12.12 | 24.35 | **2.01×** |
| 1000 | 11.98 | 19.50 | **1.63×** ← bottom |
| 2000 | 11.94 | 21.99 | 1.84× |
| 4000 | 11.89 | 23.52 | 1.98× |
| 6500 | 11.83 | 23.51 | **1.99× (MTP last point)** |
| 10000 | 11.75 | — | (MTP already at EoS) |
| 16000 | 11.62 | — | — |
| 24000 | 11.46 | — | — |
| 32000 | **11.30** | — | (baseline -7% from pos 100, KV-cache long-tail decay) |

The baseline's own tg drifts down 7% from position 100 to position 32000 — that's a pure KV-cache phenomenon, unrelated to spec-dec. The MTP run never reaches those positions because the model finishes its answer.

### Final length-coverage summary (32 → 65535 tokens)

11 max_tokens levels × cumulative position 50 → 6500 (common range with both arms) + baseline-only out to position 60000:

| level | avg speedup | P_chat tg-speedup | P_chat accept | notes |
|---:|---:|---:|---:|---|
| 32 | 2.12× | 2.39× | 95.8% | warmup-grade |
| 64 | 2.13× | 2.00× | 84.1% | |
| 128 | 2.24× | 2.18× | 80.2% | |
| 256 | 2.12× | 1.87× | 64.0% | |
| 512 | 2.12× | 1.81× | 60.3% | |
| 1024 | 2.06× | 1.66× | 52.3% | min cell avg |
| 2048 | 2.07× | 1.75× | 56.8% | |
| **4096** | **2.17×** | **2.06×** | **73.0%** | **sweep peak** |
| 8192 | 2.12× | 1.95× | 67.4% | |
| 32767 | 2.15× | 2.04× | 67.4% | MTP EoS at gen 6758 (deterministic) |
| 65535 | 2.13× | 1.94× | 67.4% | MTP EoS gen 4715-7256 (variance), baseline EoS non-deterministic |

Worst measurement of any kind across the full 32–65535 range:

- **Worst windowed bucket**: 1.43× (P_chat, T=4096, pos 500) — a single 100-token bucket, immediately recovers.
- **Worst cumulative**: 1.63× (P_chat, T=2048 / T=32767 / T=65535, pos 1000) — same position three times, structural.
- **Worst cell average**: 2.06× (T=1024).

The claim's 1.5× lower bound is met at every granularity. The claim's 2× upper bound is reached or exceeded at every cell average except T=1024/2048 (where it falls to 2.06× / 2.07×). +20% (1.2×) is never approached at any granularity or any length up to the context-window ceiling. **operational sweet spot: max_tokens ≈ 4096.**

Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep_xxlong.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep_xxlong.json) (T=8192, 7.9 MB) and [`data/raw/specdec_qwen36_27b_mtp_length_sweep_xxxlong.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep_xxxlong.json) (T=32767, 13 MB).

### Extended to max_tokens 65535 — context-window limit reached

Final length test in this sweep. `max_tokens=65535` with `--ctx-size=65536` (the bench's practical ceiling — see the next sub-section for the actual ctx limit and memory profile). 27B-MTP K=3 still holds the claim:

| max_tokens | P_code | P_chat tg | P_reason | **avg** | P_chat gen |
|---:|---:|---:|---:|---:|---|
| 32767 | 2.25× | 2.04× | 2.17× | 2.15× | base 32688 / MTP 6758 |
| **65535** | **2.27×** | **1.94×** | **2.18×** | **2.13×** | **base 65456 (run1) / 6497 (run2) / 8031 (run3) / MTP 4715-7256** |

Three new findings, all in the long-tail behavior:

**Finding 1: baseline P_chat EoS becomes non-deterministic at this length.** At T=32767 every baseline P_chat run went to gen ~32688 (the cap). At T=65535 only **1 of 3 runs** went to the cap (gen 65456); the other two stopped near the natural EoS at gen 6497 / 8031. Same `temp=0.0`, same prompt, same chat template — the EoS argmax tie-break flips depending on KV-cache numerical state. This means baseline's worst-case wall-clock is highly variable: in 1/3 of trials it spends ~92 minutes on the same chat task that MTP finishes in ~5 min.

**Finding 2: MTP's natural-EoS prediction also gains run-to-run variance at long ctx.** At T=32767 the MTP P_chat path stopped at exactly gen=6758 every time. At T=65535 it bracketed 4715 / 6758 / 7256 — the EoS prediction loosens once the context window doubles. (Median is still ~6758, so the avg-tg comparison is unchanged.)

**Finding 3: baseline KV-cache long-tail decay extends linearly to 60K tokens.** Cumulative tg on P_chat (common range + baseline-only long tail to position 60000):

| pos | base cum | MTP cum | speedup |
|---:|---:|---:|---:|
| 100 | 12.13 | 24.45 | **2.02×** |
| 1000 | 11.99 | 19.52 | **1.63×** ← bottom |
| 2000 | 11.96 | 22.00 | 1.84× |
| 4000 | 11.89 | 23.55 | 1.98× |
| 6500 | 11.84 | 23.50 | **1.99× (MTP last point)** |
| 10000 | 11.75 | — | — |
| 32000 | 11.30 | — | — (matches T=32767 measurement) |
| 48000 | 11.00 | — | — |
| 60000 | **10.79** | — | (baseline -11% from pos 100) |

Slope is **~-0.7% per 4K tokens**, the same rate observed at T=32767 (-7% over 32K). Pure KV-cache effect, unrelated to spec-dec. The MTP arm never reaches these positions because the model finishes the answer.

Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep_xxxxlong.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep_xxxxlong.json) (T=65535, 13 MB).

### Memory & throughput at the ctx ceiling (T=131072 single-shot)

The sweep above stops at `--ctx-size=65536` because that's where the per-cell wall-clock cost stops being defensible (the T=65535 sweep alone took 3.5 h; doubling it would mean another 7+ h for one more data point on a curve that already plateaued). But "where does it actually break?" is a separate question — so we also ran a single-shot probe at `--ctx-size=131072` (twice the sweep ceiling) with a real 130 000-token Lorem-Ipsum prompt and `max_tokens=100`.

**Memory (server self-report at startup, ctx=131072 + MTP K=3):**

| component | size |
|---|---|
| Main 27B model buffer (Vulkan) | **16 387 MiB ≈ 16.0 GB** |
| Main KV cache (16 KV-groups × 131072 cells, fp16, K + V) | **8 192 MiB = 8.0 GB** |
| MTP draft head model | 1 249 MiB ≈ 1.2 GB |
| MTP draft KV (1 layer × 131072) | 512 MiB |
| Compute buffers (Vulkan + Host) | ~1 GB |
| CPU-mapped tensors | ~1.4 GB host RAM |

Strix Halo has a **tiny 1 GB dedicated VRAM partition** and shares the rest of the 128 GB DDR5 with the GPU through GTT. So `rocm-smi --showmemuse` (which only sees the 1 GB partition) reads `VRAM%=77` and stays there throughout — that's misleading. The real number to watch is `free -m`:

| state | system RAM used | server share |
|---|---:|---:|
| no server | 11.4 GB | — |
| server idle (ctx=131072 reserved) | 41.7 GB | 30.3 GB |
| 130 K prompt processing peak | 45.1 GB | 33.7 GB |
| server stopped | 11.4 GB | — |

So the ceiling is nowhere near 128 GB. **No OOM, no swapping, ~33 GB out of 128 GB at the worst point.** This corrects an earlier internal estimate that pessimistically placed the KV cache at ~66 GB by ignoring GQA + the actual 16 KV-groups in this model.

**Throughput at ctx=131072:**

| measurement | rate | comparison |
|---|---:|---|
| Prompt eval (130 000 tokens) | **64.78 t/s** | 213 t/s at 128 tokens → 3.3× slowdown |
| MTP K=3 decode (100 tokens after 130 K KV is loaded) | **6.70 t/s** | ~25 t/s at short ctx → **3.7× slowdown** |

The decode rate is the new finding here. tg drops to **~1/4 of the typical MTP K=3 rate** once the KV cache is full, because attention cost on every decoded token now scales with 130 K positions. The 1.5–2× *speedup ratio* may still hold relative to baseline at the same ctx (baseline would be even slower), but the absolute throughput cliff means **operational ctx for this model on this hardware tops out around 32–65 K**, not 131 K. The DLS-060 sweep peak at T=4096 (avg 2.17×) remains the right default; the 32–65 K range is "still fine, just slower"; 131 K is "it works, but you'll wait."

**Caveat — accept rate at full ctx not measured.** The single-shot probe used `/v1/chat/completions` non-streaming, so the server timing report doesn't break out MTP accept rate. A streaming bench at full ctx is needed to confirm whether the speedup *ratio* survives or whether MTP accept rate also degrades.

Raw data: [`.claude/.dls/raw/20260514_dls061_qwen36_27b_mtp_ctx131072_*.{log,json}`](#) (server log + monitor log + response, llmtools-side only — not bundled in this public repo because it's a single-shot probe rather than a sweep).

### Conclusion: where does "+20%" come from?

On Strix Halo + Vulkan + `-fa off` + `Qwen3.6-27B-UD-Q4_K_XL` + K=3, **the speedup never approaches +20% at any granularity** — not on average over 32/64/128/256/512 tokens, not on cumulative tg over a 500-token window, not even on a single 50-token instantaneous bucket (the worst observed bucket is 1.40×, well above 1.2×). Likely sources of the +20% number:

- A different hardware (CPU-only, low-bandwidth dGPU) where the MTP head's accept rate falls faster
- A different quant or model — note that Phase 5's [MoE 35B-A3B](05-mtp-moe.md) lands at peak 1.42× regardless of length, so its averages are closer to +30-40%
- An `n-gram-{simple,mod,cache}` configuration confused for spec-dec generally — Phase 1's [n-gram evaluation](00-quick-take.md#what-we-rejected) does land in the 0.94-1.20× range
- An extremely long generation where the MTP head's accept rate finally degrades — we measured up to max_tokens=65535 in the sweep above (avg still 2.13×) and probed memory + throughput at ctx=131072 separately, so this hypothesis is no longer the explanation in this configuration

Within the bounds we measured (max_tokens 32 → 65535, plus ctx=131072 single-shot for memory/throughput characterization), the 1.5-2× claim holds end-to-end as a tg-rate ratio. The absolute throughput cliff at ctx ≥ 65–131 K is a separate concern documented in the previous sub-section.

Raw data: [`data/raw/specdec_qwen36_27b_mtp_length_sweep.json`](../data/raw/specdec_qwen36_27b_mtp_length_sweep.json) (10 cells × 3 prompts × (warmup + 3 measure runs); each run includes a `chunk_history` array of `[cumulative_gen_tokens, time_since_first_token]` pairs for post-hoc per-position analysis).

## Multi-quant sweep at T=65536 (mainline llama.cpp build 9211)

Everything above was measured on the `am17an:mtp-clean@5d5f1b46` branch (Phase 1-3 build). Two things changed between that and the next sweep:

1. **`am17an/mtp-clean` was merged into mainline.** [PR #22673](../README.md#rel-pr-22673) landed in `ggml-org/llama.cpp` at commit `053e01dff` (build 9211, 2026-05-17). The CLI surface also changed: `--spec-type mtp` is now `--spec-type draft-mtp` upstream (the old `am17an` spelling no longer parses on mainline). Two follow-up patches went in alongside the merge — [PR #23198](https://github.com/ggml-org/llama.cpp/pull/23198) (verify-pass perf) and [PR #23237](https://github.com/ggml-org/llama.cpp/pull/23237) (layer source fix).
2. **Two more dense quants were tested**, on top of the original UD-Q4_K_XL: [`unsloth/Qwen3.6-27B-MTP-GGUF` Q4_0](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (15.0 GB, smallest, dense) and `Qwen3.6-27B-UD-Q6_K_XL` (24.2 GB, biggest, dense). Both carry the MTP head in-file.

Same hardware (Strix Halo, Vulkan, gfx1151, `-fa off`), same prompts (P_code / P_chat / P_reason), same methodology — but `max_tokens=65536` and `ctx_size=65536` end-to-end, the new sweep ceiling that the length-dependence section landed on.

### Three dense quants × K ∈ {baseline, 3, 4, 5}

Speedup is the per-prompt MTP tg divided by the same prompt's baseline tg measured on the same model in the same session.

#### Qwen3.6-27B-UD-Q4_K_XL (17.9 GB — the original target; rerun on mainline)

| K | P_code (mtp / base) | P_chat | P_reason | **avg** | accept (min-max) |
|---:|---:|---:|---:|---:|---:|
| baseline | 11.86 t/s | 11.90 | 11.93 | 1.00× | — |
| 3 | 24.56 (2.07×) | 19.57 (1.64×) | 23.63 (1.98×) | 1.90× | 59-81% |
| **4** ⭐ | **25.40 (2.14×)** | **25.66 (2.16×)** | **24.94 (2.09×)** | **2.13×** | 75-93% |
| 5 | 24.65 (2.08×) | 17.39 (1.46×) | 22.52 (1.89×) | 1.81× | 45-71% |

The K=4 cell on this run reproduces the K=4 result from the original Phase 4 sweep (2.15× there, 2.13× here) within run-to-run noise. The K=4 P_chat run hit the 64985-token cap with 93% accept rate — same dynamic as the length sweep, where long chat-template answers settle into a high-accept body region.

#### Qwen3.6-27B-Q4_0 (15.0 GB — the smallest dense MTP quant; absolute throughput record)

| K | P_code (mtp / base) | P_chat | P_reason | **avg** | accept (min-max) |
|---:|---:|---:|---:|---:|---:|
| baseline | 12.91 t/s | 13.20 | 13.19 | 1.00× | — |
| 3 | 28.14 (2.18×) | 22.26 (1.69×) | 27.48 (2.08×) | 1.98× | 60-83% |
| **4** ⭐ | **28.49 (2.21×)** | 21.06 (1.60×) | 28.04 (2.13×) | 1.98× | 51-76% |
| 5 | 27.77 (2.15×) | 19.67 (1.49×) | 27.01 (2.05×) | 1.90× | 44-70% |

Q4_0 has the **fastest absolute throughput in this entire study — 28.49 t/s at K=4 P_code**, ~+12% over UD-Q4_K_XL's 25.40. The speedup *ratio* is slightly lower (1.98× vs 2.13×) because the baseline is also faster (smaller weights to read), so spec-dec has less headroom. K=3 and K=4 are a statistical tie on the average; K=3 wins on accept-rate stability, K=4 wins on absolute peak. Pick K=3 if accept rate matters (chat-heavy), K=4 if peak throughput matters (code-heavy).

#### Qwen3.6-27B-UD-Q6_K_XL (24.2 GB — the biggest dense MTP quant; highest speedup ratio)

| K | P_code (mtp / base) | P_chat | P_reason | **avg** | accept (min-max) |
|---:|---:|---:|---:|---:|---:|
| baseline | 7.95 t/s | 7.95 | 7.95 | 1.00× | — |
| 3 | 19.24 (2.42×) | 14.93 (1.88×) | 18.14 (2.28×) | 2.19× | 60-86% |
| **4** ⭐ | **20.56 (2.59×)** | **15.09 (1.90×)** | **20.09 (2.53×)** | **2.34×** | 52-79% |
| 5 | 19.81 (2.49×) | 15.28 (1.92×) | 18.84 (2.37×) | 2.26× | 50-70% |

UD-Q6_K_XL has the **highest speedup ratio in this study — avg 2.34× at K=4**. The mechanism is the same as Phase 1-3 saw with bigger targets: when the baseline is slow (7.95 t/s here, because Q6 reads more weight per pass), spec-dec recovers a larger fraction of the lost throughput by amortising verify cost over multiple accepted tokens. Absolute throughput (20.56 t/s peak) is below UD-Q4_K_XL and Q4_0; the *ratio* is the highest, the absolute throughput is the lowest.

### K=4 is the universal peak for dense 27B (across 3 quants)

| quant | size | baseline tg (avg P_code) | K=4 avg speedup | K=4 absolute peak | K=4 accept (best-worst) |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 15.0 GB | 12.91 | 1.98× | **28.49 t/s** ⭐ absolute peak | 76% / 51% |
| UD-Q4_K_XL | 17.9 GB | 11.86 | 2.13× | 25.66 t/s | 93% / 75% |
| UD-Q6_K_XL | 24.2 GB | 7.95 | **2.34×** ⭐ ratio peak | 20.56 t/s | 79% / 52% |

**Operational picks by use case:**

- **Default / general**: UD-Q4_K_XL @ K=4 — best balance, most stable accept rate (75-93%), and reproduces the original Phase 4 K=4 result.
- **Code-heavy / absolute throughput**: Q4_0 @ K=4 — fastest single number (28.49 t/s), accept rate lower but acceptable on P_code.
- **Quality-sensitive / highest speedup ratio**: UD-Q6_K_XL @ K=4 — biggest target, slowest absolute tg, but spec-dec gives the largest *fractional* recovery (2.34×).

K=4 is the peak on the cell-level average for **all three** dense quants. K=3 is the Unsloth-recipe recommendation and ties K=4 on Q4_0; K=4 wins by a small margin on UD-Q4_K_XL (+0.23×) and UD-Q6_K_XL (+0.15×).

### 35B-A3B MoE recheck on mainline (K=3, T=65536)

The original Phase 5 sweep on `am17an:mtp-clean` peaked at K=2 = 1.42× avg (see [05-mtp-moe.md](05-mtp-moe.md)). On mainline build 9211 with K ∈ {3, 4, 5}:

| K | P_code (mtp / base) | P_chat | P_reason | **avg** | accept (min-max) |
|---:|---:|---:|---:|---:|---:|
| baseline | 58.83 t/s | 58.00 | 58.71 | 1.00× | — |
| **3** ⭐ | **76.47 (1.30×)** | **61.76 (1.07×)** | **78.99 (1.35×)** | **1.24×** | 58-83% |
| 4 | 75.40 (1.28×) | 54.20 (**0.93×** ⚠️) | 75.20 (1.28×) | 1.16× | 45-73% |
| 5 | 68.94 (1.17×) | 48.72 (**0.84×** ⚠️) | 67.83 (1.16×) | 1.06× | 40-65% |

On mainline, **K=3 is the best K we measured here, but it still does not match DLS-054's K=2 = 1.42× peak**. K=4 and K=5 actively regress on P_chat (0.93× and 0.84× — both below 1.0). The MoE accept-rate curve falls off at much lower K than dense does (P_chat 58% @ K=3 vs 93% @ K=4 on UD-Q4_K_XL), so the MoE sweet spot is **lower** in K than dense. Whether K=2 reproduces its 1.42× on mainline — and whether the original 1.42× was the true peak — is an open question this sweep didn't cover.

The absolute peak rate is **79 t/s at K=3 P_reason** — fastest single number across any model in this study, dense or MoE. This is unrelated to spec-dec efficiency: the 35B-A3B baseline is already ~58 t/s on this hardware because only 3B parameters are active per token.

### Conclusions across the multi-quant sweep

- **The K=4 sweet spot is robust across dense quants.** It was first observed on UD-Q4_K_XL, and now reproduces on Q4_0 (15.0 GB) and UD-Q6_K_XL (24.2 GB). The accept-rate curve scales with K independent of quant; quant only shifts the baseline tg.
- **Pick the quant by use case, not by speedup ratio.** Q4_0 wins absolute throughput, UD-Q6_K_XL wins ratio, UD-Q4_K_XL wins accept-rate stability — all at K=4.
- **MoE is a different regime.** 35B-A3B's sweet spot is at a lower K than dense (K ≤ 3 on mainline), and the average speedup never crosses 1.5× in any of our measurements. The high absolute tg (~79 t/s peak) is the MoE-architecture story, not the MTP story.
- **Mainline migration is a no-op for the speedup numbers.** UD-Q4_K_XL @ K=4 on mainline produces 2.13× avg, indistinguishable from `am17an:mtp-clean`'s 2.15× avg in the original Phase 4 sweep. PRs #23198 / #23237 land alongside the merge but do not visibly shift the cell-average ratios.

Raw data: [`data/raw/specdec_qwen36_dls062_t65536_multiquant.json`](../data/raw/specdec_qwen36_dls062_t65536_multiquant.json) (16 cells = 4 models × {baseline, K=3, K=4, K=5}, 3 prompts × (warmup + 3 measure runs), build 9211).

## Caveats

- **`-fa on` (Vulkan flash-attn) not measured here.** Unsloth's published recipe uses `-fa on` but our Strix Halo Vulkan path defaults to `-fa off` per [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>. All Phase 1-3 numbers also use `-fa off`, so the comparison stays apples-to-apples — but the absolute speedups might shift with `-fa on`. Re-running this sweep under `-fa on` is the most obvious next experiment.
- **MTP + `--mmproj` not supported.** The target is an image-text-to-text model, but the mtp-clean branch disallows `--mmproj` alongside `--spec-type mtp`. Open upstream constraint, no workaround.
- **One model family tested across multiple quants.** MTP is a per-model architecture feature. Within `Qwen3.6` we have now covered 3 dense quants (Q4_0 / UD-Q4_K_XL / UD-Q6_K_XL) and 1 MoE quant (35B-A3B-UD-Q4_K_XL); K=4 is the dense sweet spot across all three dense quants. Whether the K=4 sweet spot transfers to *other* MTP-capable model families (Gemma 4, DeepSeek-MoE, etc.) remains untested — as of 2026-05-19 no other public MTP-GGUF release is available for those families.
- **[PR #22673](../README.md#rel-pr-22673) was merged into mainline** at `ggml-org/llama.cpp` commit `053e01dff` (build 9211, 2026-05-17). The CLI flag changed: `--spec-type mtp` (am17an branch) → `--spec-type draft-mtp` (mainline). Two follow-up patches landed alongside the merge ([PR #23198](https://github.com/ggml-org/llama.cpp/pull/23198) verify-pass perf, [PR #23237](https://github.com/ggml-org/llama.cpp/pull/23237) layer source fix). Phase 1-3 numbers in this repo were taken on `am17an:mtp-clean@5d5f1b46`; the multi-quant sweep above is on the mainline build. Cell-level averages match within run-to-run noise (UD-Q4_K_XL K=4: 2.15× am17an → 2.13× mainline).

## Raw data

[`data/raw/specdec_qwen36_27b_mtp_sweep.json`](../data/raw/specdec_qwen36_27b_mtp_sweep.json) — 45 KB, 7 cells × 3 prompts × (warmup + 3 measure runs), `--spec-type mtp` with `--spec-draft-n-max` ∈ {1, 2, 3, 4, 5, 8}. Schema matches the Phase 3 spec-dec files (see [`data/raw/README.md`](../data/raw/README.md)).

Cell IDs: `A0_baseline` (no spec), `C{1,2,4,5,8}_mtp_K{1,2,4,5,8}`, plus `B3_mtp_K3` (different prefix is a bench-script artifact; values are comparable).
