# 05 — Qwen3.6-35B-A3B-MTP self-speculation (MoE target)

Phase 4 covered the dense **Qwen3.6-27B** with built-in MTP head. This page covers the MoE variant — **Qwen3.6-35B-A3B-MTP-GGUF** (UD-Q4_K_XL, 22.9 GB), released by Unsloth with the same "~1.5-2× faster generation" claim and the same `--spec-draft-n-max=3` recipe. On Strix Halo + Vulkan + `-fa off`, **the same recipe does not transfer**: K=2 turns out to be the operational sweet spot at 1.42× avg, and the recipe's K=3 lands below the claim's lower bound.

## TL;DR

| K (`--spec-draft-n-max`) | P_code (tg / pp) | P_chat (tg / pp) | P_reason (tg / pp) | g_avg | accept (min-max) | notes |
|---:|---:|---:|---:|---:|---:|---|
| baseline | 57.84 (1.00×) / 346.8 | 58.62 (1.00×) / 252.3 | 58.64 (1.00×) / 333.7 | 1.00× | — | no spec-dec |
| 1 | 74.27 (1.28×) / 317.1 | 71.60 (1.22×) / 227.7 | 76.24 (1.30×) / 297.7 | 1.27× | 83-97% | |
| **2** ⭐ | **85.42 (1.48×)** / 313.1 | **78.23 (1.33×)** / 219.0 | **84.31 (1.44×)** / 286.6 | **1.42×** | 73-93% | peak avg, P_chat's only viable K |
| 3 | 84.55 (1.46×) / 295.9 | 64.91 (**1.11×**) / 207.7 | 84.24 (1.44×) / 261.5 | 1.33× | 57-85% | **Unsloth recipe — suboptimal on P_chat** |
| 4 | 78.83 (1.36×) / 287.5 | 59.44 (**1.01×**) / 198.7 | 80.08 (1.37×) / 266.1 | 1.25× | 48-75% | P_chat ≈ baseline |
| 5 | 70.93 (1.23×) / 286.6 | 57.10 (**0.97×** ⚠️) / 204.7 | 78.51 (1.34×) / 272.3 | 1.18× | 45-71% | P_chat below baseline |
| 8 | 46.82 (**0.81×** ⚠️) / 288.9 | 31.32 (**0.53×** ⚠️) / 206.6 | 47.64 (**0.81×** ⚠️) / 275.5 | **0.72×** ⚠️ | 28-53% | **all prompts collapse** |

`tg` is median tokens-per-second (decode); `pp` is median prompt-processing tokens-per-second. Speedup is vs. the per-session baseline `A0_baseline`. Accept range is the worst-to-best across 3 prompts × 3 measure runs.

**Operational recommendation: K=2 for 35B-A3B MTP.** Not K=3 (the Unsloth recipe), not K=4 (where 27B's sweet spot was), not K=8 (a runtime regression on every prompt). The dense-27B → MoE-35B-A3B transition shifts the sweet spot down by two notches.

## Setup difference vs Phase 4 (27B MTP)

| | Phase 4 (dense 27B) | Phase 5 (MoE 35B-A3B) |
|---|---|---|
| target GGUF | [`unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)<sup>[↘](../README.md#rel-unsloth-mtp)</sup> (17.9 GB) | [`unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)<sup>[↘](../README.md#rel-unsloth-mtp-35b)</sup> (22.9 GB) |
| architecture | dense | MoE (36B total / 3B active, 256 experts / 8 routed + 1 shared) |
| baseline tg | ~11.84 t/s | ~58.4 t/s |
| llama-server flags | `--spec-type mtp --spec-draft-n-max=K` | (same, only K differs) |
| Unsloth recipe K | 3 | 3 (identical text) |
| Measured sweet spot K | **4** (avg 2.15×) | **2** (avg 1.42×) |
| Claim "~1.5-2×" | **reproduced** at K=3 (2.13× avg) | **below lower bound** at peak K=2 (1.42× avg) |

Everything else aligned: same Strix Halo + Vulkan + `-fa off` + `am17an:mtp-clean@5d5f1b46` build, same three prompts, same warmup + 3 measure runs methodology, same `--no-warmup -np 1 ctx=16384 temp=0`.

## Why K=2 is the answer (and K=3 is not)

The two cells are very close on `P_code` and `P_reason` (1.48× vs 1.46× and 1.44× vs 1.44×), but `P_chat` diverges sharply:

| K | P_chat tg | P_chat speedup | P_chat accept_med | what changes |
|---:|---:|---:|---:|---|
| 2 | 78.23 | **1.33×** | 73.2% | viable |
| 3 | 64.91 | **1.11×** | **56.6%** | accept halves, gain collapses |
| 4 | 59.44 | 1.01× | 47.6% | ≈ baseline |

At K=3, P_chat's MTP head accepts only 57% of proposed drafts — the verify pass at batch=4 is essentially running for one accepted token on average, which is roughly the same wall-clock as a non-spec-dec single-token forward pass. There is no headroom to push K↑ further without going net negative on this prompt class.

The pattern is the same as Phase 4's "P_chat is the bottleneck" finding, just shifted lower: on 35B-A3B the wall hits at K=3, on dense 27B it hits at K=8.

## K=8: every prompt below baseline

All three prompts at K=8 measure below baseline: P_code 0.81×, P_chat 0.53×, P_reason 0.81×. P_chat lands at 31 t/s — about half of the 58 t/s baseline. Accept rates collapse to 28-53% and `draft_n` per generated token explodes (P_chat: K=2 = 414 drafts for ~290 accepted tokens vs K=8 = 1248 drafts for ~340 accepted). The verify kernel runs at batch=9 for almost no marginal accepted tokens — the spec-dec mechanism is doing work and losing to it.

This is structurally worse than dense 27B at K=8, where only P_chat went slightly below baseline (0.90×) and P_code and P_reason stayed positive. **On MoE 35B-A3B, K=8 is a runtime regression on every workload.**

## Unsloth's claims vs. what we measured

The [model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)<sup>[↘](../README.md#rel-unsloth-mtp-35b)</sup> uses identical wording to the 27B version:

> **NEW: MTP speculative decoding for ~1.5-2x faster generation**

With the same `--spec-type mtp --spec-draft-n-max 3` recipe and `-fa on`. We ran `-fa off` on Vulkan gfx1151 (see [02-context.md](02-context.md)) to keep apples-to-apples with Phase 1-4.

| Unsloth's claim | What we measured | Comment |
|---|---|---|
| ~1.5-2× faster generation | **peak 1.42× avg at K=2**; K=3 (recipe) is 1.33× avg | **Claim lower bound not reached** — contrasts sharply with Phase 4's 27B reproduction at K=3 = 2.13× avg |
| Recipe default K=3 | K=2 = 1.42× avg, K=3 = 1.33× avg | K=3 is suboptimal on this hardware/quant; K=2 wins on every workload (notably P_chat 1.33× vs 1.11×) |
| Same MTP-decoding wording as 27B-MTP | 27B: 2.13× at K=3, 35B-A3B: 1.33× at K=3 | Identical recipe text, very different outcomes — implies a hardware/quant-dependent caveat the model card doesn't surface |
| Single GGUF, no external draft | confirmed | Same MTP head loading mechanism as Phase 4 (`override_arch=qwen35_mtp` in server log) |

The two model cards are nearly verbatim, but on Strix Halo + Vulkan + `-fa off`, dense 27B reproduces the claim while MoE 35B-A3B does not. The most likely missing context is **baseline tg**: 27B sits at 12 t/s (memory-bound, lots of headroom for spec-dec); 35B-A3B sits at 58 t/s (each spec-dec round's expert-routing overhead is already comparable to baseline's per-token cost).

If the claim was measured on `-fa on` (Unsloth's recipe default), it would explain part of the gap, but `-fa on` isn't usable on Vulkan gfx1151 because of [#12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>.

## Comparison vs Phase 1-3 (35B-A3B + external 0.8B draft)

Phase 1-3 measured the same target (Qwen3.6-35B-A3B family, but at UD-Q6_K) with an external Qwen3.5-0.8B-Q4_0 draft model at K=4:

| | 35B-A3B baseline | + spec-dec config | result |
|---|---:|---|---|
| Phase 1-3 (DLS-052, UD-Q6_K target + 0.8B-Q4_0 draft K=4) | 58.52 t/s | `-md 0.8B-Q4_0.gguf --spec-draft-n-max=4` | +11% avg, **P_chat 0.90× ⚠️** |
| Phase 5 (UD-Q4_K_XL target + built-in MTP head K=2) | 58.36 t/s avg | `--spec-type mtp --spec-draft-n-max=2` | **+42% avg, P_chat 1.33×** ⭐ |

MTP self-speculation at K=2 substantially outperforms external draft at K=4 on MoE 35B-A3B: **+42% vs +11%, and P_chat goes from a slowdown to a 1.33× speedup**. The MTP head's tighter coupling with the target (no separate forward pass per round) avoids the overhead that DLS-052 identified as the gain-eater for external-draft + MoE.

Phase 5 partially overturns DLS-035's "MTP not viable on 35B-A3B" rejection (which was written before the MTP-GGUF release): with a built-in MTP head and K=2 specifically, MoE 35B-A3B *does* see meaningful spec-dec gain — just not at the Unsloth recipe's K=3 and not up to the claim's lower bound.

## Caveats

- **`-fa on` not measured.** Unsloth's recipe uses `-fa on`; we ran `-fa off` because of [#12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>. Whether the recipe's K=3 reaches the 1.5× lower bound under `-fa on` is open — listed under "What we didn't measure (yet)" in [00-quick-take.md](00-quick-take.md).
- **One MoE family tested.** The K=2-vs-K=3 shift may be specific to Qwen3.6-35B-A3B's expert-routing pattern (256 experts / 8 routed + 1 shared). Another MoE family with different sparsity could behave differently.
- **PR #22673 still unmerged** ([↘](../README.md#rel-pr-22673)). Same `am17an:mtp-clean@5d5f1b46` build as Phase 1-4.
- **`pp` (prompt-processing) drops mildly under spec-dec.** Baseline `pp` averages ~311 t/s across the three prompts; K=8 averages ~257 t/s (-17%). Most of the K-dependence is in `tg` (decode), not `pp`.

## Raw data

[`data/raw/specdec_qwen36_35b_a3b_mtp_sweep.json`](../data/raw/specdec_qwen36_35b_a3b_mtp_sweep.json) — 45 KB, 7 cells × 3 prompts × (warmup + 3 measure runs). Same schema as `specdec_qwen36_27b_mtp_sweep.json` (see [`data/raw/README.md`](../data/raw/README.md)). `tg_med` and `pp_med` are recorded alongside `accept_med` and `draft_n_med`.

Cell IDs match Phase 4's naming for direct comparison: `A0_baseline` (no spec), `C{1,2,4,5,8}_mtp_K{1,2,4,5,8}` and `B3_mtp_K3` (Unsloth recipe). The B/C prefix split is a bench-script artifact; values are comparable.
