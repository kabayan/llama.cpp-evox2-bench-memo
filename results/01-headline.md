# Headline numbers, with reasoning

The Quick Take page gave you the tables. This page explains why each finding matters and how we got to "K=4 default" as the operational answer.

## 1. The K↑ ceiling is workload-specific, not hardware

Conventional wisdom from earlier `lorem ipsum` micro-benchmarks (DLS-045 in the internal log): on memory-bound GPUs, the K=1 cell hits the highest speedup and K↑ regresses. That came from this scaling table on the same Strix Halo + 27B-Q4_0 + 0.8B draft + perfect-accept setup (lorem ipsum, 4096 prompt → 256 gen):

| cell | draft_max | draft_min | tg t/s | speedup | accept | draft_n |
|---|---:|---:|---:|---:|---:|---:|
| N1 | 1 | 1 | 15.06 | **1.144×** | 100% | 79 |
| N2 | 2 | 2 | 14.73 | 1.119× | 99% | 68 |
| N4 | 4 | 4 | 14.66 | 1.114× | 100% | 60 |
| N8 | 8 | 8 | 12.49 | 0.949× (anomaly) | 100% | 8 |
| N16 | 16 | 5 | 14.26 | 1.083× | 100% | 47 |
| N32 | 32 | 5 | 14.26 | 1.083× | 100% | 47 |

The K↑ regression there is real, but its cause is subtle: at K=N with `--spec-draft-n-min=N --spec-draft-n-max=N`, the draft model's `p_min=0.75` early-break (`common/speculative.cpp:339-341`) can leave fewer than N tokens in the draft buffer. The server then discards the round entirely (`tools/server/server-context.cpp:2480`, `if (slot.task->params.speculative.n_min > (int) draft.size()) continue;`). On `lorem ipsum`, with per-token confidence `q ≈ 0.85`, the survival probability `q^K` decays fast — at K=8 you keep only 27% of rounds, which is why the N8 cell shows 0.95× slowdown.

On real prompts the draft 0.8B keeps `q ≈ 0.99` (acc=100% measured at K=1, 96-99% at K=4). The early-break essentially never fires, and the K↑ regression vanishes. Instead we get:

| K | mean spd | accept (worst) | what's changing |
|---:|---:|---:|---|
| 1 | 1.26× | 100% | 42% of rounds run spec-dec; the rest are non-spec single-token forward passes |
| 4 | 1.82× | 96% | 68% of rounds run spec-dec; per-round yield ↑ |
| 16 | 2.07× | 92% | 81% of rounds run spec-dec; accept slips, kernel efficiency halved |

**Takeaway**: the K↑ ceiling on `lorem ipsum` is an artifact of the draft model's confidence on that particular token stream, *not* a property of the kernel. On any prompt where the draft stays confident, K can scale up.

## 2. Why K=4 is the operational answer (and not K=16)

K=16 produces the highest single number (P_code 2.45×). But three things degrade at the same time:

**(a) Accept rate slides.** 100% (K=1) → 96-98% (K=4) → 92-97% (K=16). At K=16, individual prompts drift to 92.9% accept (P_chat) — close enough to where the `p_min` early-break can fire on adversarial prompts that you should expect occasional regressions.

**(b) Kernel efficiency halves.** Forward-pass count for `gen_tokens=512`:

| K | spec-dec rounds | non-spec rounds | total fwd pass | theoretical speedup | measured | efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 218 | 76 | 294 | 1.74× | 1.31× | 75% |
| 4 | 87 (~349/4) | 163 | 250 | 2.05× | n/a | n/a |
| 16 | 26 (~414/16) | 72 | 98 | 5.22× | 2.45× | 47% |

(These numbers are for P_code; theoretical speedup is `gen_tokens / fwd_pass` assuming perfect accept; "efficiency" is `measured / theoretical`.) At K=16, the kernel runs each round at roughly 2× the cost it "should" — Strix Halo's Vulkan batched-verify can't keep batch=K+1 in a single weight read past ~K=4.

**(c) Variance stays low but accept variance grows.** All cells have run-to-run tg variance < 1.04×. But P_code at K=16 shows 94.9% / 97.3% / 97.2% accept across the three runs — a 2.4% spread. Still good, but you've consumed most of the safety margin.

**K=4 is where all three (accept stability, kernel efficiency, gain) are still in good shape**. K=8 is a defensible "+5% if you want" choice. K=16 is "I only run P_code and accept the variance" mode.

## 3. The 35B-A3B (MoE) case kills the universal K=4 recommendation

We expected the same K↑ pattern on Qwen3.6-35B-A3B-UD-Q6_K. We got:

| cell | P_code | P_chat | P_reason | accept |
|---|---:|---:|---:|---:|
| 35B baseline | 58.52 | 58.53 | 58.51 | — |
| 35B + 0.8B + K=4 | 65.16 (1.11×) | 52.65 (**0.90×**) | 64.38 (1.10×) | 96.7% / 92.6-96.3% / 97.5% |

The baseline at 58 t/s pushes ~56% of 256 GB/s LPDDR5X bandwidth, and with only 3B active experts per token the model is effectively small from the memory side. Adding a 0.8B draft forward pass per accepted round becomes a ~20% overhead, which on P_chat (where accept drops to 92.6%) outweighs the gain entirely.

**Structurally**: spec-dec on this hardware works when `baseline_tg / draft_tg << K`. For 27B-Q4_0, baseline ~13 t/s and draft 0.8B ~80 t/s give ratio ~0.16, so K=4 (and K=16) are far from saturating. For 35B-A3B, baseline ~58 t/s and draft 0.8B ~80 t/s give ratio ~0.73 — the overhead is already comparable to one draft round, K=4 is right at the edge, K=8 would be net negative.

So the **universal default** answer is K=4, knowing it gives +11% on 35B-A3B (with one prompt going negative) and 1.49-2.05× on 27B-Q4_0. The model-aware answer would be `K=4` on 27B-Q4_0 and `none` (or `ngram-mod` for code only) on 35B-A3B.

## 4. Draft size: 0.8B optimum, with no surprises

| Draft | size | P_code | P_chat | P_reason | accept (K=1) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q4_0 | 507 MB | 1.31× | 1.20× | 1.28× | 100% |
| Qwen3.5-2B-Q4_0 | 1.2 GB | 1.27× | 1.16× | 1.27× | 100% |
| Qwen3.5-4B-Q4_0 | 2.6 GB | 1.11× | 1.06× | 1.13× | 100% |

All three drafts hit accept=100% at K=1 with `temp=0` — they all argmax to the same token the target would have chosen, for these prompts. So the only differentiator is the draft's own forward-pass cost, and the smallest one wins.

Worth noting: prior measurements at K=16 lorem ipsum (the conditions DLS-036 used to reject all three drafts) showed accept dropping to 92-93% for 2B/4B and the rejection logic was different. K=1 + real prompts is a much friendlier regime, but the relative ordering (0.8B best, 4B worst) holds.

## 5. The n-gram families: rejected, with structural reasons

Previous evaluations on this same hardware (DLS-041 internal):

| spec-type | P_code | P_chat | P_reason | notes |
|---|---:|---:|---:|---|
| `ngram-simple` (default size-n=12) | 0.94× | 1.00× | 1.00× | accept 14-20% on code, 0% chat/reason |
| `ngram-mod` (default) | **1.52× ✓** | 1.20× | 1.09× (var 2.18×) | only P_code passes ≥1.30×, and even there variance is 1.76× |
| `ngram-cache` (dynamic, in-memory) | 0.71× | 0.73× | 0.63× | net slowdown across all prompts |
| `ngram-cache` (static, per-prompt oracle) | 0.71× | 0.90× | 0.69× | even a *perfect* corpus can't beat the draft validation cost |

We also separately patched `ngram-simple` to fix two bugs in the upstream PR ([DLS-048 patches](../../docker/patches/) — not in this Phase 1 push, see Phase 2). With the patch + lorem ipsum + raw `/completion`, ngram-simple reached **5.74×**. But on chat-template + real prompts (where most production use lives), the patched version still degenerates to 0.94-1.00×.

The structural reason n-gram fails on chat-template real prompts: the prompt template + reasoning preamble shift the token distribution enough that ngrams from the generated output don't match anything in the immediate past. ngram-mod has heuristics that partially compensate for code (run-of-tokens like `def `, `return `, `): `), but they don't generalize to conversation or math.

## 6. Per-prompt-type insight: P_chat is the bottleneck

Across every config we measured, P_chat has the lowest speedup. K=4 gives P_code 2.05× / P_reason 1.91× but P_chat is stuck at 1.49×, and going to K=16 only nudges it to 1.59×.

`draft_n` per generated token gives the story: P_code 0.68 (i.e. 68% of generated tokens come from spec-dec rounds at K=4), P_reason 0.67, P_chat 0.52. Conversational text has shorter "high-confidence runs" — the draft model is confident for ~3-4 tokens at a time (e.g. mid-sentence content words) and then hits a low-confidence boundary (sentence breaks, transitions, named entities). Code and reasoning have longer runs (variable names, repeated keywords, mathematical step sequences).

If you wanted to optimize per workload, you'd pick K=4 for chat and K=16 for code/reasoning. The simple default (K=4 for all) keeps chat optimal and leaves 0.5× of gain on the table for code-heavy workloads.

## 7. What this implies for tooling defaults

If you operate `llama-server` on this hardware with a chat-style UI:

- Set `--spec-draft-n-max=4 --spec-draft-n-min=1` as your default (assuming a Qwen3.5/3.6 family target).
- Don't set `--spec-draft-n-min == --spec-draft-n-max`. The `p_min` early-break + round-discard interaction (see [01-headline.md §1](#1-the-k-ceiling-is-workload-specific-not-hardware)) becomes destructive at K≥8.
- Don't ship n-gram as default. Make it opt-in (or remove it from the menu if you don't have code-heavy workloads).
- For 35B-A3B-class MoE targets where baseline tg is already ≥40 t/s, default to no spec-dec, or expose K explicitly.
