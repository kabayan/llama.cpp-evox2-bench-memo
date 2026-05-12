# 03 — Full per-cell tables

This page is the raw-data companion to [00-quick-take.md](00-quick-take.md) and [01-headline.md](01-headline.md). Every cell that contributes a number on those pages is listed here with the median values from each prompt; the underlying `runs[]` arrays (warmup + 3 measure runs each) live in [`data/raw/`](../data/raw/) and are linked per section.

If you want to re-derive the speedups: divide each cell's `tg_med` by the matching baseline cell's `tg_med`. We don't include per-run breakdowns here — they're available verbatim in the JSON.

## 1. K-sweep on Qwen3.5-27B-Q4_0 + Qwen3.5-0.8B-Q4_0

Per-cell median tg with speedup vs. the per-session baseline (Qwen3.5-27B-Q4_0, no spec-dec). `B*` cells use `--spec-draft-n-max=K --spec-draft-n-min=1`.

Raw data: [`data/raw/specdec_27b_k_sweep.json`](../data/raw/specdec_27b_k_sweep.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 13.32 t/s, P_chat 13.27 t/s, P_reason 13.29 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `A0_baseline` (baseline) | 13.32 (1.00×) / acc 0.0% / draft_n 0 | 13.27 (1.00×) / acc 0.0% / draft_n 0 | 13.29 (1.00×) / acc 0.0% / draft_n 0 |
| `B1_08b_K1` (K=1) | 17.47 (1.31×) / acc 100.0% / draft_n 218 | 15.94 (1.20×) / acc 100.0% / draft_n 174 | 17.04 (1.28×) / acc 100.0% / draft_n 208 |
| `B2_08b_K2` (K=2) | 22.36 (1.68×) / acc 99.7% / draft_n 293 | 18.28 (1.38×) / acc 98.7% / draft_n 230 | 21.07 (1.59×) / acc 98.6% / draft_n 280 |
| `B3_08b_K4` (K=4) ⭐ | 27.36 (2.05×) / acc 98.9% / draft_n 349 | 19.77 (1.49×) / acc 96.3% / draft_n 267 | 25.42 (1.91×) / acc 96.8% / draft_n 341 |

## 2. K-high: K=8 / K=16 on Qwen3.5-27B-Q4_0

Same target + draft as §1 but with `--spec-draft-n-max=8` and `=16`. Baseline cell is run fresh in this session to bound drift; the K=1..4 values from §1 use a different baseline.

Raw data: [`data/raw/specdec_27b_k_high.json`](../data/raw/specdec_27b_k_high.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 13.31 t/s, P_chat 13.12 t/s, P_reason 13.27 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `A0_baseline` (baseline) | 13.31 (1.00×) / acc 0.0% / draft_n 0 | 13.12 (1.00×) / acc 0.0% / draft_n 0 | 13.27 (1.00×) / acc 0.0% / draft_n 0 |
| `B4_08b_K8` (K=8) | 28.19 (2.12×) / acc 97.1% / draft_n 378 | 20.59 (1.57×) / acc 93.1% / draft_n 290 | 26.96 (2.03×) / acc 95.2% / draft_n 378 |
| `B5_08b_K16` (K=16) | 32.56 (2.45×) / acc 97.2% / draft_n 402 | 20.87 (1.59×) / acc 92.9% / draft_n 295 | 28.58 (2.15×) / acc 94.9% / draft_n 393 |

## 3. Draft size sweep at K=1 (Qwen3.5-27B-Q4_0 target)

K is fixed at 1 to isolate per-step draft-model overhead. Same `data/raw/specdec_27b_k_sweep.json` source. 0.8B is the optimum across all three prompts.

Raw data: [`data/raw/specdec_27b_k_sweep.json`](../data/raw/specdec_27b_k_sweep.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 13.32 t/s, P_chat 13.27 t/s, P_reason 13.29 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `A0_baseline` (baseline) | 13.32 (1.00×) / acc 0.0% / draft_n 0 | 13.27 (1.00×) / acc 0.0% / draft_n 0 | 13.29 (1.00×) / acc 0.0% / draft_n 0 |
| `B1_08b_K1` (draft 0.8B-Q4_0) | 17.47 (1.31×) / acc 100.0% / draft_n 218 | 15.94 (1.20×) / acc 100.0% / draft_n 174 | 17.04 (1.28×) / acc 100.0% / draft_n 208 |
| `C1_2b_K1` (draft 2B-Q4_0) | 16.90 (1.27×) / acc 100.0% / draft_n 221 | 15.44 (1.16×) / acc 100.0% / draft_n 184 | 16.81 (1.26×) / acc 100.0% / draft_n 224 |
| `D1_4b_K1` (draft 4B-Q4_0) | 14.80 (1.11×) / acc 100.0% / draft_n 237 | 14.04 (1.06×) / acc 100.0% / draft_n 201 | 14.99 (1.13×) / acc 100.0% / draft_n 237 |

## 4. K=1: `--spec-draft-n-min=0` vs `=1` (no measurable difference)

At K=1 with accept=100% on real prompts, the `p_min` early-break never fires regardless of `--spec-draft-n-min`. Confirmed by direct measurement.

Raw data: [`data/raw/specdec_27b_k1_min.json`](../data/raw/specdec_27b_k1_min.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 13.32 t/s, P_chat 13.27 t/s, P_reason 13.30 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `A0_baseline` (baseline) | 13.32 (1.00×) / acc 0.0% / draft_n 0 | 13.27 (1.00×) / acc 0.0% / draft_n 0 | 13.30 (1.00×) / acc 0.0% / draft_n 0 |
| `A1_k1_min1` (min=1) | 17.41 (1.31×) / acc 100.0% / draft_n 218 | 15.97 (1.20×) / acc 100.0% / draft_n 174 | 17.05 (1.28×) / acc 100.0% / draft_n 208 |
| `A2_k1_min0` (min=0) | 17.41 (1.31×) / acc 100.0% / draft_n 218 | 15.94 (1.20×) / acc 100.0% / draft_n 174 | 17.04 (1.28×) / acc 100.0% / draft_n 208 |

## 5. Qwen3.6-35B-A3B-UD-Q6_K + 0.8B-Q4_0 draft, K=4

Target switched to the MoE 35B-A3B (active 3B parameters). Baseline is already ~58 t/s (~56% of 256 GB/s memory bandwidth), so spec-dec overhead eats most of the gain — P_chat goes negative (0.90× slowdown).

Raw data: [`data/raw/specdec_35b_a3b_k4.json`](../data/raw/specdec_35b_a3b_k4.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 58.52 t/s, P_chat 58.53 t/s, P_reason 58.51 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `E0_35b_baseline` (baseline) | 58.52 (1.00×) / acc 0.0% / draft_n 0 | 58.53 (1.00×) / acc 0.0% / draft_n 0 | 58.51 (1.00×) / acc 0.0% / draft_n 0 |
| `E1_35b_08b_K4_min1` (0.8B draft, K=4, min=1) | 65.16 (1.11×) / acc 96.7% / draft_n 363 | 52.65 (0.90×) / acc 95.7% / draft_n 216 | 64.38 (1.10×) / acc 97.5% / draft_n 354 |

## 6. n-gram on Qwen3.6-35B-A3B-UD-Q6_K, real prompts (patched ngram-simple)

Post-patch (`docker/patches/01,02_*.patch`) ngram-simple on real chat-template prompts. None of the n-gram families breaks 1.30× on chat or reason; only ngram-mod on P_code marginally helps.

Raw data: [`data/raw/ngram_35b_a3b_real.json`](../data/raw/ngram_35b_a3b_real.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 58.44 t/s, P_chat 58.57 t/s, P_reason 58.54 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `S0_none` (baseline) | 58.44 (1.00×) / acc 0.0% / draft_n 0 | 58.57 (1.00×) / acc 0.0% / draft_n 0 | 58.54 (1.00×) / acc 0.0% / draft_n 0 |
| `S1_ngram_simple` (ngram-simple, patched, size-n=12) | 55.08 (0.94×) / acc 20.3% / draft_n 192 | 58.60 (1.00×) / acc 0.0% / draft_n 0 | 58.59 (1.00×) / acc 0.0% / draft_n 0 |
| `S2_ngram_mod` (ngram-mod default) | 89.07 (1.52×) / acc 64.3% / draft_n 503 | 70.39 (1.20×) / acc 72.9% / draft_n 192 | 63.86 (1.09×) / acc 52.7% / draft_n 256 |
| `S3_ngram_cache` (ngram-cache dynamic) | 41.76 (0.71×) / acc 33.9% / draft_n 488 ⚠️ run: timed out | 42.98 (0.73×) / acc 18.5% / draft_n 311 | 36.90 (0.63×) / acc 14.1% / draft_n 518 |

## 7. ngram-cache: oracle (per-prompt corpus) vs project corpus vs ngram-mod

Even when the n-gram cache is hand-built from the *exact* expected output (oracle), accept rate caps at 41-65% and draft-validation cost outweighs the gain. This rules out n-gram-cache as a viable spec-dec mode on this hardware/target combination.

Raw data: [`data/raw/ngram_35b_a3b_cache_oracle.json`](../data/raw/ngram_35b_a3b_cache_oracle.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 58.53 t/s, P_chat 58.70 t/s, P_reason 58.57 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `A_none` (baseline) | 58.53 (1.00×) / acc 0.0% / draft_n 0 | 58.70 (1.00×) / acc 0.0% / draft_n 0 | 58.57 (1.00×) / acc 0.0% / draft_n 0 |
| `B_cache_oracle` (ngram-cache static, oracle corpus per-prompt) | 41.47 (0.71×) / acc 46.1% / draft_n 516 | 52.61 (0.90×) / acc 65.4% / draft_n 198 | 40.47 (0.69×) / acc 41.6% / draft_n 548 |
| `C_cache_project` (ngram-cache static, project repo corpus) | 39.31 (0.67×) / acc 41.4% / draft_n 461 | 44.79 (0.76×) / acc 40.3% / draft_n 211 | 41.18 (0.70×) / acc 30.8% / draft_n 315 |
| `D_ngram_mod` (ngram-mod baseline) | 88.61 (1.51×) / acc 64.3% / draft_n 503 | 71.15 (1.21×) / acc 72.9% / draft_n 192 | 64.75 (1.11×) / acc 52.7% / draft_n 256 |

## 8. ngram-simple size-n=3 vs size-n=12 (DLS-049 rejection)

Reducing the n-gram match window from 12 to 3 (Option C in DLS-049) explodes match count by ~7-100× — but each match is much weaker, accept drops to 5-9%, and the round-discard logic erases the gain. Baseline cell `S1_n12` is the ngram-simple default; `S1_n3` is the rejected option.

Raw data: [`data/raw/ngram_simple_size_n3.json`](../data/raw/ngram_simple_size_n3.json). Values are median of 3 measure runs (warmup discarded). `acc` is median accept_rate; `draft_n` is median drafted-tokens count across `gen_tokens=512`.

Baseline tg (no spec-dec): P_code 57.46 t/s, P_chat 58.58 t/s, P_reason 58.03 t/s.

| cell | P_code | P_chat | P_reason |
|---|---|---|---|
| `S1_n12` (baseline) | 57.46 (1.00×) / acc 15.8% / draft_n 240 | 58.58 (1.00×) / acc 0.0% / draft_n 0 | 58.03 (1.00×) / acc 22.7% / draft_n 22 |
| `S1_n3` (ngram-simple, size-n=3) | 39.05 (0.68×) / acc 8.3% / draft_n 1875 | 38.91 (0.66×) / acc 5.4% / draft_n 1629 | 31.12 (0.54×) / acc 7.3% / draft_n 2125 |

## 9. n-gram on Qwen3.5-27B-Q4_0, lorem-ipsum 4096→256 (DLS-041 legacy)

Legacy workload (one fixed prompt, lorem-ipsum 4096 tokens → 256 gen) used for the n-gram comparison in DLS-041. Included here because the same `ngram-simple` (size-n=12) on this prompt does *not* reach the bug-fix-era 5.74× — the patched build still measures ~1.0× when called against the chat-template pipeline. The 5.74× number on `lorem ipsum` requires raw `/completion` calls bypassing the chat template.

Raw data: [`data/raw/ngram_27b_lorem.json`](../data/raw/ngram_27b_lorem.json). Single legacy workload `W1_pp4096_gen256` (lorem-ipsum 4096 prompt / 256 gen, no chat template). Median of 3 runs.

External baseline (Qwen3.5-27B-Q4_0 no-spec on this build): ≈ 13.32 t/s.

| cell | tg (t/s) | accept | draft_n | pp (t/s) |
|---|---|---|---|---|
| `N1_mtp_vulkan_ngram_simple` (ngram-simple default size-n=12) | 13.19 (0.99×) | 0.0% | 0 | 301.74 |
| `N2_mtp_vulkan_ngram_mod` (ngram-mod default) | 18.45 (1.39×) | 39.5% | 256 | 280.80 |

## Data file index

| File | Size | What's in it |
|---|---:|---|
| [`specdec_27b_k_sweep.json`](../data/raw/specdec_27b_k_sweep.json) | 37 KB | §1 + §3 source. K=1/2/4 sweep + draft size 0.8B/2B/4B at K=1. |
| [`specdec_27b_k_high.json`](../data/raw/specdec_27b_k_high.json) | 19 KB | §2 source. K=8/16 sweep with fresh baseline. |
| [`specdec_27b_k1_min.json`](../data/raw/specdec_27b_k1_min.json) | 19 KB | §4 source. K=1 + min=0/1 comparison. |
| [`specdec_35b_a3b_k4.json`](../data/raw/specdec_35b_a3b_k4.json) | 13 KB | §5 source. MoE target + 0.8B draft K=4. |
| [`ngram_35b_a3b_real.json`](../data/raw/ngram_35b_a3b_real.json) | 23 KB | §6 source. MoE + patched ngram-{simple, mod, cache} real prompts. |
| [`ngram_35b_a3b_cache_oracle.json`](../data/raw/ngram_35b_a3b_cache_oracle.json) | 33 KB | §7 source. ngram-cache oracle/project vs ngram-mod. |
| [`ngram_simple_size_n3.json`](../data/raw/ngram_simple_size_n3.json) | 4 KB | §8 source. size-n=3 rejection data. |
| [`ngram_27b_lorem.json`](../data/raw/ngram_27b_lorem.json) | 8 KB | §9 source. Legacy lorem-ipsum workload on 27B. |

All files share the same top-level shape (`cells.<cell_id>.{prompts | workloads}.<prompt_or_workload>.{warmup, tg_med, accept_med, draft_n_med, runs[]}`). The `runs[]` array contains per-run values for `ttft`, `pp_ts`, `tg_ts`, `gen_tokens`, `actual_prompt`, `draft_n`, `draft_n_accepted`, `accept_rate` — open the file directly if you need to recompute variance or per-prompt accept distributions.

`prompts.*` schemas are the real-prompt format (P_code / P_chat / P_reason). `workloads.W1_pp4096_gen256` is the lorem-ipsum legacy format used only in §9.
