# llama.cpp evox2 (Strix Halo) Speculative Decoding Bench Memo

Empirical notes from running `llama.cpp` Speculative Decoding on **AMD Strix Halo (gfx1151)** with **Vulkan**. The headline finding: with a **Qwen3.5-27B-Q4_0 target + Qwen3.5-0.8B-Q4_0 draft + `--spec-draft-n-max=4`**, real-world prompts (code/chat/reasoning) hit **1.49× – 2.05× speedup** while staying **stable across runs (variance < 1.04×)** — far better than any n-gram-based spec-dec on the same hardware.

This repo is a *lab notebook*, not a polished benchmark suite. Phases 1 (results), 2 (reproduction: Docker + scripts), and 3 (full per-cell tables + raw JSON) are all in place. Known open items are listed under "What we didn't measure (yet)" in [results/00-quick-take.md](results/00-quick-take.md).

## TL;DR

| Recommendation | Spec-dec config | Result on Qwen3.5-27B-Q4_0 |
|---|---|---|
| **Default for 27B-Q4_0** ⭐ | `--spec-type` (draft model), `-md Qwen3.5-0.8B-Q4_0.gguf`, `--spec-draft-n-max=4 --spec-draft-n-min=1` | **1.49× – 2.05×** (mean 1.82×), accept 96-98%, variance < 1.04× |
| Push for max P_code | Same, `--spec-draft-n-max=16 --spec-draft-n-min=1` | **2.45× on P_code**, but accept 92-97% (variance ↑, kernel efficiency ↓) |
| Avoid on Qwen3.6-35B-A3B | Same draft + K=4 | Only +11%, P_chat slows down to **0.90×** (MoE baseline 58 t/s already fast, spec-dec overhead eats gain) |
| All n-gram families | `--spec-type ngram-{simple,mod,cache}` | Rejected — best case (ngram-mod on P_code, 35B-A3B only) is 1.52× and **var 1.76×**; chat/reason flat or slower |

If you only remember one knob: **target = 27B-Q4_0, draft = 0.8B-Q4_0, K = 4, min = 1**.

## Why this is interesting

1. **Folklore says K=1 is the ceiling on memory-bound hardware.** That claim came from `lorem ipsum` micro-benchmarks where the draft model loses confidence on repetitive token streams and the spec-dec round gets `p_min`-skipped (see [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), `common/speculative.cpp:339`). On real prompts, draft 0.8B keeps 96-100% acceptance through K=4 — so the K↑ ceiling **never engages** and per-round token yield grows.
2. **Strix Halo's Vulkan batched-verify kernel is *partially* inefficient, not fully.** Kernel efficiency drops from 75% at K=1 to 47% at K=16 — but K=4 still hits 68% efficiency, which is why K=4 is the sweet spot rather than K=1 (DLS-045 line of reasoning) or K=16 (max raw speedup but accept and kernel both degrade).
3. **The 35B-A3B MoE pattern is structurally different.** Its baseline tg (~58 t/s) is already memory-bound at 56% bandwidth utilization with only 3B active parameters. Adding a 0.8B draft forward pass *per accepted round* eats most of the spec-dec gain. Conclusion: spec-dec speedup is strongly tied to baseline tg being slow.

## What's measured

- **Hardware**: AMD Strix Halo, gfx1151, 256 GB/s LPDDR5X (Vulkan path, ROCm flash-attn off — see [llama.cpp issue #12629](https://github.com/ggml-org/llama.cpp/issues/12629))
- **llama.cpp build**: `am17an:mtp-clean` head SHA `5d5f1b46` (PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) — needed for the checkpoint-based spec-dec path that supports Qwen3.5 hybrid linear+full attention models
- **Quants**: Qwen3.5-27B-Q4_0 (target, 15.7 GB), Qwen3.5-0.8B/2B/4B-Q4_0 (drafts), Qwen3.6-35B-A3B-UD-Q6_K (additional target, 28 GB)
- **Prompts**: 3 fixed real-world prompts (Python `binary_search`, 3-day Kyoto trip plan, 2-train relative-motion problem), `max_tokens=512`, `temp=0`, `ctx=16384`, chat-template via `--jinja --reasoning-format auto`
- **Methodology**: warmup + 3 measure runs per cell, server restarted between cells (`pkill llama-server` → wait `/v1/models`), sequential only (no GPU sharing during a run)

## Results

- **[results/00-quick-take.md](results/00-quick-take.md)** — single-page summary with the K-sweep and draft-size-sweep tables
- **[results/01-headline.md](results/01-headline.md)** — numbers behind the K=4 recommendation, kernel-efficiency curve, 35B-A3B contrast
- **[results/02-context.md](results/02-context.md)** — hardware/software stack and what's specific to this configuration
- **[results/03-full-tables.md](results/03-full-tables.md)** — full per-cell tables (every cell that contributes a number on 00/01) with raw-data links
- **[data/raw/](data/raw/)** — sanitized per-run JSON (8 files, one per bench session) — re-runnable with `scripts/run_bench.py`

## Phases (publishing roadmap)

| Phase | Status | Contents |
|---|---|---|
| 1. Results | ✅ done | README + `results/00..02` + LICENSE |
| 2. Reproduction | ✅ done | [`docker/`](docker/) (Dockerfile.mtp-vulkan + 2 patches + build/run doc), [`scripts/`](scripts/) (single-file `run_bench.py` with `httpx` only + sweep recipes) |
| **3. Per-cell tables + raw data** | ✅ this push | [`results/03-full-tables.md`](results/03-full-tables.md) (per-cell tg/accept/draft_n medians) + [`data/raw/`](data/raw/) (8 sanitized JSON files) |

## Reproducing the numbers

```bash
# 1. Build the image (~10 min on Strix Halo)
docker build -f docker/Dockerfile.mtp-vulkan -t llama-cpp-evox2-bench .

# 2. Start the container with your GGUF directory mounted
docker run -d --name llama-evox2 --device /dev/dri \
  -v /path/to/gguf:/gguf:ro -p 10001:10001 \
  llama-cpp-evox2-bench

# 3. Run the recommended config (~7 min)
pip install httpx
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.5-27B-Q4_0.gguf \
  --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --draft-n-max 4 --draft-n-min 1 \
  --output bench_k4.json
```

See [`docker/README.md`](docker/README.md) for the image internals, [`scripts/README.md`](scripts/README.md) for sweep recipes (K sweep, draft size sweep, 35B-A3B variant).

## Related

- llama.cpp Speculative Decoding upstream PR: https://github.com/ggml-org/llama.cpp/pull/22673 (`am17an:mtp-clean`, open as of 2026-05-12)
- Strix Halo Vulkan flash-attn note: https://github.com/ggml-org/llama.cpp/issues/12629
- `llama.cpp` `common/speculative.cpp` (`p_min` default 0.75, the early-break responsible for `lorem ipsum` K↑ regression)
- `tools/server/server-context.cpp:2480` (round-discard when `n_min > draft.size()`, why `--spec-draft-n-min == --spec-draft-n-max` is an anti-pattern at K≥8)

## License

MIT — see [LICENSE](LICENSE).
