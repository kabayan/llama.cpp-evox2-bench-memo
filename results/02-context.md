# Context: hardware, software, methodology

This page documents what's specific to this configuration. If your hardware or stack differs in any of these dimensions, the numbers in [00-quick-take.md](00-quick-take.md) probably do not transfer.

## Hardware

- **APU**: AMD Strix Halo (Ryzen AI Max series), iGPU `gfx1151`
- **Memory**: 128 GB LPDDR5X, theoretical peak 256 GB/s shared between CPU and iGPU
- **OS**: Ubuntu 24.04, Linux 6.17 (oem)
- **Power profile**: default (no manual ppm tuning during benchmarks)

Measured baseline bandwidth utilization (`baseline tg × model size`):

| Model | tg (no spec) | size | bandwidth used | % of 256 GB/s |
|---|---:|---:|---:|---:|
| Qwen3.5-27B-Q4_0 (dense) | 13.32 t/s | 15.19 GB | 200 GB/s | **78%** |
| Qwen3.6-35B-A3B-UD-Q6_K (MoE, active 3B) | 58.52 t/s | 2.46 GB | 144 GB/s | **56%** |

The 27B target leaves only ~22% bandwidth headroom for spec-dec verify passes, but the per-token compute is dominated by memory reads of the full model. The 35B-A3B target uses less bandwidth in absolute terms (only ~3B of weights are read per token), but its faster baseline means spec-dec gain has less room to operate.

## Software stack

### llama.cpp build

We use **`am17an:mtp-clean`** at head SHA `5d5f1b46`, the source branch of upstream PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)<sup>[↘](../README.md#rel-pr-22673)</sup> ("llama + spec: MTP Support"). As of 2026-05-12 this PR is **open and unmerged**.

Why this build:
- The stable `master` branch (build 8763) rejects spec-dec on Qwen3.5 family targets with `common_speculative_is_compat: the target context does not support partial sequence removal`. Qwen3.5 uses hybrid linear+full attention, which the partial-sequence-removal-required code path doesn't handle.
- The `mtp-clean` branch adds a checkpoint-based spec-dec mechanism (`speculative decoding will use checkpoints`) that works around this. It was originally designed for Qwen3.6 MTP self-speculation but the same machinery enables external `-md` drafts for hybrid-attention targets.
- Important constraint inherited from this PR: `-np 1` is required, `--mmproj` is not supported.

### Patches we applied

The Phase 2 push will include two small patches on top of `5d5f1b46`:
- **Patch A** (`docker/patches/01_ngram_simple_continue_search.patch`, ~50 lines): fixes a one-shot backward-search bug in `common/ngram-map.cpp:77-103` where `ngram-simple` would stop after the first match window.
- **Patch B** (`docker/patches/02_ngram_simple_state_respect_cli.patch`, ~23 lines): fixes `common/speculative.cpp:797-805` to actually honor the `--spec-ngram-simple-size-n` CLI flag instead of using a hardcoded default.

With these patches, `ngram-simple` on raw `/completion` + `lorem ipsum` reaches 5.74×. **They do not affect the draft-model numbers in this Phase 1 results page.** The patches are useful only for the specific case of self-similar generation streams.

### Backends tested

| Backend container | Status | Notes |
|---|---|---|
| `vulkan` (stable) | spec-dec unsupported on Qwen3.5 | flash-attn off (per [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>) |
| `lemonade` (UD-Q4_K) | baseline reference only | stable build, no spec-dec |
| **`mtp-vulkan`** ⭐ | spec-dec works | what all our numbers use |
| `mtp-rocm` (ROCm 7.x + gfx1151) | spec-dec works but slower | accept rate similar to mtp-vulkan, but baseline ~12% lower (gfx1151 ROCm immature) |

The `vulkan` flash-attn-off rule comes from [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)<sup>[↘](../README.md#rel-vulkan-flash-attn)</sup>: on gfx1151, the Vulkan flash-attention path silently produces wrong outputs. We force `--flash-attn off` on Vulkan and `--flash-attn on` on ROCm/lemonade.

## Methodology

### Sequential, no GPU sharing

Every cell runs in isolation:
1. `pkill -9 llama-server` in the container
2. wait 2 seconds
3. Start `llama-server` with the cell's flags
4. Poll `GET /v1/models` until ready (max 600 s)
5. Run **warmup** measurement (results discarded but warm caches and JIT paths)
6. Run **3 measurement** runs; record per-run `tg`, `accept_rate`, `draft_n`, `draft_n_accepted`, `ttft`, prompt token count
7. Take the median tg as the cell's representative number

**No parallel runs** within a cell, across cells, or across configurations. GPU sharing or batching artifacts can swing tg by 10-30% on this hardware (we observed this early and rule it out as standard practice now).

### Prompts (held fixed across all evaluations)

Three single-turn user prompts, no system message, chat template applied by `--jinja --reasoning-format auto`:

- **P_code**: "Write a Python function `binary_search(arr: list[int], target: int) -> int` ..." (~115 tokens). Tests structured code generation with type hints, docstring, test block.
- **P_chat**: "I'm planning a 3-day trip to Kyoto in mid-November. Please suggest a concrete day-by-day itinerary..." (~85 tokens). Tests open-ended conversation.
- **P_reason**: "Two trains start at the same morning. Train A leaves Tokyo for Osaka at 9:00 AM..." (~95 tokens). Tests multi-step math reasoning.

`max_tokens=512`, `temp=0.0`, `cache_prompt=false`. Output is streamed via SSE; the bench script reads `timings.draft_n` / `timings.draft_n_accepted` from the per-chunk `timings` dict that llama-server emits.

### `--spec-draft-n-min` always less than `--spec-draft-n-max`

Setting them equal is an anti-pattern at K≥8 because of the `p_min` early-break + round-discard interaction (see [01-headline.md §1](01-headline.md#1-the-k-ceiling-is-workload-specific-not-hardware) for the gory details). The K=1 case is an exception (the buffer either has 1 token or 0, and `min=1` works fine), but we still set `min=1, max=K` everywhere for consistency.

## Pinning information

For reproducibility (Phase 2 will include the Dockerfile that pins all of this):

- llama.cpp ref: `am17an:mtp-clean@5d5f1b46`
- Vulkan SDK: bundled with `ubuntu:24.04` apt packages
- Qwen3.5 GGUF source: `unsloth/Qwen3.5-{27B,4B,2B,0.8B}-Q4_0.gguf` (HuggingFace)
- Qwen3.6 GGUF source: `unsloth/Qwen3.6-35B-A3B-UD-Q6_K.gguf` (HuggingFace)
- bench client: Python 3 + `httpx` (streaming SSE, sync API)

## Bench raw data

Phase 3 will publish:
- Per-run JSON for each bench (TG / PP / TTFT / accept / draft_n / gen_tokens)
- Server log excerpts where compatibility messages matter
- The derivation table for kernel efficiency (theoretical vs. measured speedup)
