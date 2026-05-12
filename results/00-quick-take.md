# Quick Take

One-page summary of the headline numbers. Bench setup is in [02-context.md](02-context.md); the reasoning behind each finding is in [01-headline.md](01-headline.md).

## K sweep on Qwen3.5-27B-Q4_0 + Qwen3.5-0.8B-Q4_0 draft

3 real prompts × warmup + 3 measure runs, median tg t/s. Baseline = no spec-dec.

| K | P_code (code) | P_chat (conversation) | P_reason (math) | accept (worst) | variance (max/min) | kernel eff. (est.) |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | 13.32 (1.00×) | 13.27 (1.00×) | 13.29 (1.00×) | — | < 1.01× | — |
| 1  | 17.47 (1.31×) | 15.94 (1.20×) | 17.04 (1.28×) | 100% | < 1.01× | 75% |
| 2  | 22.36 (1.68×) | 18.28 (1.38×) | 21.07 (1.59×) | ~99% | < 1.01× | 73% |
| **4** ⭐ | **27.36 (2.05×)** | **19.77 (1.49×)** | **25.42 (1.91×)** | **96%** | **< 1.04×** | **68%** |
| 8  | 28.19 (2.12×) | 20.59 (1.57×) | 26.96 (2.03×) | 93% | < 1.03× | 55% |
| 16 | **32.56 (2.45× 🎉)** | 20.87 (1.59×) | 28.58 (2.15×) | 92% | < 1.04× | 47% |

**K=4 is the sweet spot.** K=8 gives +5% mean, K=16 gives +14% mean but kernel efficiency drops to 47% and accept rate slides to ~92%. K=16 is the right choice if you only generate code (P_code 2.45×, a 1.6 GB/s output stream on this hardware) and can tolerate the variance.

P_chat is the bottleneck on every K: 1.49× → 1.57× → 1.59× from K=4 to K=16 (only +7%) — conversational prompts have a lower "high-confidence run length" for the draft model. P_code and P_reason keep scaling.

## Draft size sweep on Qwen3.5-27B-Q4_0 + K=1 (real prompts)

K=1 isolates the per-step overhead of the draft model (only one draft token per round, so the comparison is "how much does draft inference cost vs. how much does target verification save").

| Draft | P_code | P_chat | P_reason | accept |
|---|---:|---:|---:|---:|
| 0.8B Q4_0 | **17.47 (1.31×)** | **15.94 (1.20×)** | **17.04 (1.28×)** | 100% |
| 2B Q4_0   | 16.90 (1.27×) | 15.44 (1.16×) | 16.81 (1.27×) | 100% |
| 4B Q4_0   | 14.80 (1.11×) | 14.04 (1.06×) | 14.99 (1.13×) | 100% |

All three drafts hit **accept=100%** on real prompts (greedy decoding, `temp=0` — the draft is essentially a smaller model running the same argmax at every step, and in practice it matches the target's argmax for these prompts). But the 2B and 4B drafts pay their own forward-pass cost per round, eating most of the spec-dec gain. **0.8B is optimum.**

This also rules in `Qwen3.6-family` targets: see the 35B-A3B section below.

## 35B-A3B (MoE active 3B) + 0.8B draft + K=4

Compatibility note: Qwen3.5-0.8B (vocab 248320) **is compatible** with Qwen3.6-35B-A3B (same vocab). `llama-server` log:

```
print_info: n_vocab               = 248320
srv    load_model: speculative decoding will use checkpoints
slot   load_model: id  0 | task -1 | speculative decoding context initialized
```

| cell | P_code | P_chat | P_reason | accept |
|---|---:|---:|---:|---:|
| 35B baseline | 58.52 | 58.53 | 58.51 | — |
| 35B + 0.8B + K=4 | 65.16 (**1.11×**) | 52.65 (**0.90× ⚠️**) | 64.38 (**1.10×**) | 96.7% / 92.6-96.3% / 97.5% |

The 35B-A3B baseline at 58 t/s already pushes ~56% of available memory bandwidth, and the draft's forward-pass cost per accepted round consumes almost all of the spec-dec savings. P_chat actually goes **negative** (0.90× slowdown). The K↑ scaling that works on 27B-Q4_0 (slow baseline, high bandwidth headroom) does *not* transfer here.

For 35B-A3B on this hardware, **n-gram-mod with code prompts (1.51×, from a separate evaluation)** is still the only spec-dec config that pays off — but with variance ≈ 1.76× it's the kind of speedup that vanishes on individual runs.

## Qwen3.6-27B-UD-Q4_K_XL + built-in MTP head (self-speculation)

Alternative target: Unsloth ships [`Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) with the MTP draft head inside the same file. The model card advertises **"MTP speculative decoding for ~1.5-2× faster generation"**; we measured the recipe (`--spec-type mtp --spec-draft-n-max=3`) and the K-sweep around it on the same Strix Halo + Vulkan + `-fa off` stack.

| K (`--spec-draft-n-max`) | P_code | P_chat | P_reason | avg | accept (min-max) |
|---:|---:|---:|---:|---:|---:|
| baseline | 11.86 (1.00×) | 11.81 (1.00×) | 11.84 (1.00×) | 1.00× | — |
| 1 | 19.48 (1.64×) | 18.59 (1.57×) | 19.68 (1.66×) | 1.63× | 84-95% |
| 2 | 25.63 (2.16×) | 22.34 (1.89×) | 25.28 (2.14×) | 2.06× | 73-93% |
| **3** (Unsloth recipe) | 26.98 (2.28×) | 21.75 (1.84×) | 27.05 (2.29×) | **2.13×** | 60-84% |
| **4** ⭐ | 27.65 (2.33×) | 21.55 (1.83×) | 27.05 (2.29×) | **2.15×** | 54-81% |
| 8 | 20.41 (1.72×) | 11.67 (**0.90× ⚠️**) | 18.49 (1.56×) | 1.42× | 27-59% |

**K=3 (Unsloth's recipe) reproduces the upper end of the claimed ~1.5-2× range — 2.13× avg.** K=4 nudges past it on the average (2.15×) and on P_code (2.33×); beyond K=4 the MTP head's joint-prediction confidence falls off fast and K=8 is a runtime regression on chat workloads.

Absolute tg at K=4 (21.55-27.65 t/s) lands in the same band as Phase 1-3's external-draft K=4 (19.77-27.36 t/s) — MTP doesn't save tokens/sec on this hardware, it saves operator overhead (no second GGUF, no `-md` flag, no draft-vocab compatibility check).

See [04-mtp.md](04-mtp.md) for the full claim-vs-measured comparison and the K=8 P_chat collapse analysis.

## What we rejected

| Approach | Why rejected | Result |
|---|---|---|
| `--spec-type ngram-simple` (default size-n=12) | Real prompts: 14-23% accept on code, 0% on chat/reason | 0.94× to 1.00× on 35B-A3B |
| `--spec-type ngram-simple` (size-n=3) | Match rate explodes (7-100× more drafts) but accept drops to 5-9% | 0.53× – 0.67× slowdown |
| `--spec-type ngram-cache` (dynamic or static) | Even per-prompt oracle corpus only hits 41-65% accept; draft validation cost > gain | 0.63× – 0.90× across configs |
| Q3.5-27B + draft 2B/4B at K=16 | Higher accept but draft forward-pass cost dominates | ≤1.19× |
| K=1 + min=1 vs min=0 | At K=1 with accept=100%, `min` never triggers `p_min`-skip | No measurable difference |

## What we didn't measure (yet)

- K=8/16 on 35B-A3B (predicted to be neutral or worse given K=4 already shows slowdown)
- `-fa on` for the MTP K-sweep (Unsloth's recipe default; we ran `-fa off` because of [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup> on gfx1151 — re-running this sweep on a backend where Vulkan flash-attn is correct would close that gap)
- MTP-GGUF on a MoE target (e.g. 35B-A3B class) — Unsloth has not released one as of 2026-05-12; structurally interesting because the 35B-A3B baseline at 58 t/s already eats most of the spec-dec margin
- Other model families (Llama 3.x): tokenizer compatibility is the gate, not measured
- Direct kernel profiling (RGP / rocprof / Tracy) to confirm the partial-inefficiency model
- AMD ROCm 7.x + gfx1151 (we tested mtp-rocm and got worse results; Vulkan path is preferred on this hardware)
