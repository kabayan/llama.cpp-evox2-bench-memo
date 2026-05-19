# data/raw — sanitized bench JSON

Per-run JSON dumps from each bench session referenced in [`../../results/`](../../results/). Sanitization removes the container name and port number that the bench harness recorded (internal lab hints); every measurement value (`tg_ts`, `accept_rate`, `draft_n`, `draft_n_accepted`, `gen_tokens`, `actual_prompt`, `ttft`, `pp_ts`) is preserved verbatim.

## Files

| File | Source on | Contents |
|---|---|---|
| [`specdec_27b_k_sweep.json`](specdec_27b_k_sweep.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + Qwen3.5-{0.8B,2B,4B}-Q4_0 draft, K=1/2/4 + draft size sweep at K=1 (DLS-051 main). |
| [`specdec_27b_k_high.json`](specdec_27b_k_high.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + 0.8B-Q4_0 K=8/16 with fresh baseline (DLS-052). |
| [`specdec_27b_k1_min.json`](specdec_27b_k1_min.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + 0.8B-Q4_0 K=1 with `--spec-draft-n-min=0` vs `=1` (DLS-050, "no effect at K=1" confirmation). |
| [`specdec_35b_a3b_k4.json`](specdec_35b_a3b_k4.json) | 2026-05-12 | Qwen3.6-35B-A3B-UD-Q6_K + 0.8B-Q4_0 K=4 (DLS-052, MoE pattern). |
| [`specdec_qwen36_27b_mtp_sweep.json`](specdec_qwen36_27b_mtp_sweep.json) | 2026-05-12 | Qwen3.6-27B-UD-Q4_K_XL + built-in MTP head, K=1/2/3/4/5/8 (DLS-053). No external draft GGUF; `--spec-type mtp`. |
| [`specdec_qwen36_27b_mtp_length_sweep.json`](specdec_qwen36_27b_mtp_length_sweep.json) | 2026-05-13 | Qwen3.6-27B-UD-Q4_K_XL + MTP K=3, max_tokens ∈ {32, 64, 128, 256, 512} for baseline and MTP (DLS-055, length-dependence test). **Each run includes a `chunk_history: [[cumulative_gen_tokens, time_since_first_token], ...]` array** for post-hoc per-position cumulative tg computation. |
| [`specdec_qwen36_27b_mtp_length_sweep_long.json`](specdec_qwen36_27b_mtp_length_sweep_long.json) | 2026-05-13 | Same target + K=3 setup, max_tokens ∈ {1024, 2048} (DLS-056, extended length-dependence test). Same `chunk_history` schema. P_chat's cumulative tg traces a U-shape: ~2.00× at pos 100, 1.63× at pos 1000, 1.74× at pos 2000. |
| [`specdec_qwen36_27b_mtp_length_sweep_xlong.json`](specdec_qwen36_27b_mtp_length_sweep_xlong.json) | 2026-05-13 | Same target + K=3 setup, max_tokens=4096 only (DLS-057, U-shape recovery completion). The U-shape on P_chat reaches its minimum at pos 500 (1.64×) and recovers to 2.06× by pos 4000; windowed peak at pos 2500 = 2.48×. avg over three prompts = 2.17× = highest of the entire 32–4096 sweep. |
| [`specdec_qwen36_27b_mtp_length_sweep_xxlong.json`](specdec_qwen36_27b_mtp_length_sweep_xxlong.json) | 2026-05-13 | Same setup, max_tokens=8192 (DLS-058). U-shape continues but flattens; P_chat cumulative peak at pos 5000 = 2.05×, slight decline to 1.98× at pos 6500. avg 2.12×. |
| [`specdec_qwen36_27b_mtp_length_sweep_xxxlong.json`](specdec_qwen36_27b_mtp_length_sweep_xxxlong.json) | 2026-05-13 | Same setup, **ctx_size raised to 32768** for max_tokens=32767 (DLS-059). avg 2.15×, P_chat tg-speedup 2.04×. **Two findings**: (a) MTP terminates P_chat at gen=6758 (natural EoS) while baseline runs to gen=32688 (max), so wall-clock cost is **~10× lower for MTP** on this task (294s vs 2895s); (b) baseline tg drifts down 7% from pos 100 to pos 32000 — pure KV-cache long-tail decay, unrelated to spec-dec. |
| [`specdec_qwen36_27b_mtp_length_sweep_xxxxlong.json`](specdec_qwen36_27b_mtp_length_sweep_xxxxlong.json) | 2026-05-14 | Same setup, **ctx_size raised to 65536** for max_tokens=65535 (DLS-060, the practical Strix Halo ceiling: ~33 GB extra KV cache). avg 2.13×, P_chat tg-speedup 1.94×. **Three findings**: (a) baseline P_chat EoS becomes non-deterministic — only 1 of 3 runs went to the cap (gen 65456); the other two stopped at 6497 / 8031 near the natural EoS, so baseline's worst-case wall-clock is highly variable; (b) MTP P_chat gen also gains run-to-run variance (4715 / 6758 / 7256 vs the deterministic 6758 at T=32767); (c) baseline KV-cache long-tail decay extends linearly to pos 60000 (10.79 t/s = -11% from pos 100), same -0.7%/4K slope as observed at T=32767. |
| [`specdec_qwen36_35b_a3b_mtp_sweep.json`](specdec_qwen36_35b_a3b_mtp_sweep.json) | 2026-05-12 | Qwen3.6-35B-A3B-UD-Q4_K_XL + built-in MTP head (MoE variant), K=1/2/3/4/5/8 (DLS-054). Same `--spec-type mtp` setup as the 27B file. K=2 peak / K=3 (Unsloth recipe) suboptimal / K=8 all-prompt collapse. |
| [`specdec_qwen36_dls062_t65536_multiquant.json`](specdec_qwen36_dls062_t65536_multiquant.json) | 2026-05-18 to 2026-05-19 | **Mainline llama.cpp build 9211** (commit `053e01dff`, after [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) was merged). Multi-quant sweep at `max_tokens=65536` / `ctx_size=65536` covering **4 models × 4 K conditions × 3 prompts × (warmup + 3 measure runs)** = 16 cells. Models: `Qwen3.6-27B-Q4_0-MTP` (15.0 GB, dense, absolute throughput peak 28.49 t/s @ K=4), `Qwen3.6-27B-UD-Q4_K_XL` (17.9 GB, dense, balanced default — same target as the original Phase 4 sweep), `Qwen3.6-27B-UD-Q6_K_XL` (24.2 GB, dense, speedup ratio peak 2.34× @ K=4), `Qwen3.6-35B-A3B-UD-Q4_K_XL` (22.9 GB, MoE, K=3 peak 1.24× — K=4/5 regress on P_chat). K conditions: baseline (`spec_type=null`) and `--spec-type draft-mtp --spec-draft-n-max=K` for K ∈ {3, 4, 5}. New top-level fields: `schema_version`, `build`, `common_config`. Each cell preserves the per-prompt `warmup`, `runs`, `tg_med`/`tg_min`/`tg_max`/`pp_med`/`ttft_med`/`accept_med`/`draft_n_med`/`gen_tokens_med`/`actual_prompt_med` keys verbatim. (DLS-062) |
| [`ngram_35b_a3b_real.json`](ngram_35b_a3b_real.json) | 2026-05-11 | 35B-A3B + none / ngram-{simple, mod, cache} on real prompts (post-patch ngram-simple). |
| [`ngram_35b_a3b_cache_oracle.json`](ngram_35b_a3b_cache_oracle.json) | 2026-05-11 | 35B-A3B + ngram-cache static (oracle per-prompt corpus, project repo corpus) vs ngram-mod. |
| [`ngram_simple_size_n3.json`](ngram_simple_size_n3.json) | 2026-05-11 | 35B-A3B + ngram-simple `--spec-ngram-simple-size-n=3` vs `=12` (DLS-049 rejection). |
| [`ngram_27b_lorem.json`](ngram_27b_lorem.json) | 2026-05-11 | 27B-Q4_0 + ngram-simple / ngram-mod, lorem-ipsum 4096→256 legacy workload (DLS-041). |

## Schema

Three layouts coexist:

**Real-prompt layout** (most files):

```jsonc
{
  "cells": {
    "<cell_id>": {
      "use_draft": true | false,    // external-draft files
      "use_mtp"  : true | false,    // MTP files (specdec_qwen36_27b_mtp_sweep.json)
      "draft_max": <int>, "draft_min": <int>,
      "draft": "<model_path>" | null,
      "prompts": {
        "P_code"  : { "warmup": {...}, "runs": [<run>, <run>, <run>],
                      "tg_med": ..., "tg_min": ..., "tg_max": ...,
                      "accept_med": ..., "draft_n_med": ...,
                      "gen_tokens_med": ..., "actual_prompt_med": ...,
                      "pp_med": ..., "ttft_med": ... },
        "P_chat" : { ... same shape ... },
        "P_reason": { ... same shape ... }
      }
    },
    ...
  },
  "config": {                       // top-level, MTP file only
    "target": "<gguf_path>",
    "ctx": 16384, "reps": 3, "max_tokens": 512,
    "binary": "/app/build/bin/llama-server",
    "spec_type": "mtp"
  },
  "started_at": "<iso8601>", "finished_at": "<iso8601>"
}
```

Each cell carries `use_draft` (external-draft path) or `use_mtp` (MTP self-speculation), never both. The external-draft files lack a top-level `config` block.

Each `<run>` is:

```jsonc
{
  "ttft": <s>, "pp_ts": <t/s>, "tg_ts": <t/s>,
  "total": <s>, "decode": <s>,
  "gen_tokens": <int>, "actual_prompt": <int>,
  "draft_n": <int>, "draft_n_accepted": <int>,
  "accept_rate": <0..1>
}
```

**Multi-quant DLS-062 layout** (only `specdec_qwen36_dls062_t65536_multiquant.json`):

```jsonc
{
  "schema_version": "dls-062-multiquant-v1",
  "description": "<one-paragraph methodology>",
  "build": {
    "llama_cpp_commit": "053e01dff",
    "build_number": 9211,
    "branch": "mainline (ggml-org/llama.cpp)",
    "merged_prs": ["#22673", "#23198", "#23237"]
  },
  "common_config": {
    "container": "...", "binary": "...", "gpu_layers": 99,
    "ctx": 65536, "max_tokens": 65536,
    "spec_type_cli_flag": "draft-mtp",
    "prompts": ["P_code", "P_chat", "P_reason"],
    "reasoning_format": "auto", "runs_per_prompt": 3
  },
  "cells": {
    "<model_label>_<K_label>": {
      "model": "<gguf basename>",
      "model_quant_label": "<short tag, e.g. 27B-UD-Q6_K_XL>",
      "spec_type_internal": "mtp" | null,    // null on baseline cells
      "draft_n_max": <int> | null,            // null on baseline cells
      "draft_n_min": <int> | null,
      "started_at": "<iso8601>", "finished_at": "<iso8601>",
      "prompts": { ... same per-prompt shape as the real-prompt layout above ... }
    }
  }
}
```

`spec_type_internal` records the internal flag value (`"mtp"`) that the bench harness used; the actual CLI flag passed to `llama-server` is `--spec-type draft-mtp` (mainline build), translated by the harness. Baseline cells set `spec_type_internal: null` and pass no `--spec-type` flag.

**Legacy lorem-ipsum layout** (only `ngram_27b_lorem.json`):

```jsonc
{
  "cells": {
    "<cell_id>": {
      "target": "<gguf_path>",
      "spec_type": "ngram-simple" | "ngram-mod" | ...,
      "draft_max": <int>, "draft_min": <int>,
      "workloads": {
        "W1_pp4096_gen256": {
          "tg_med": ..., "pp_med": ..., "accept_med": ..., "draft_n_med": ...,
          "runs": [<run>, ...]
        }
      }
    }
  }
}
```

## Reproducing these numbers

Each file maps to a `scripts/run_bench.py` invocation. The mapping is documented in [`../../scripts/README.md`](../../scripts/README.md); the short version is:

- Spec-dec K-sweep (`specdec_27b_*.json`): run `scripts/run_bench.py --target Qwen3.5-27B-Q4_0.gguf --draft Qwen3.5-0.8B-Q4_0.gguf --draft-n-max K --draft-n-min 1` for K ∈ {1, 2, 4, 8, 16}.
- Draft size sweep: same invocation, vary `--draft` between `0.8B`, `2B`, `4B`.
- 35B-A3B (`specdec_35b_a3b_k4.json`): target `Qwen3.6-35B-A3B-UD-Q6_K.gguf` + draft `Qwen3.5-0.8B-Q4_0.gguf` (same vocab) + `--draft-n-max 4`.
- MTP self-speculation, dense 27B (`specdec_qwen36_27b_mtp_sweep.json`): target `Qwen3.6-27B-UD-Q4_K_XL.gguf` (downloaded directly from Unsloth — the MTP head is in the same GGUF), pass `--spec-type mtp --spec-draft-n-max K` to `llama-server`. No `--draft` flag. Server flags reused: `--jinja --reasoning-format auto --flash-attn off -np 1 -ngl 99`.
- MTP self-speculation, MoE 35B-A3B (`specdec_qwen36_35b_a3b_mtp_sweep.json`): same invocation as above but target `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.9 GB) from [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). K=2 was the empirical sweet spot here, not Unsloth's K=3 recipe.
- MTP self-speculation, multi-quant T=65536 (`specdec_qwen36_dls062_t65536_multiquant.json`): mainline `llama.cpp@053e01dff` (build 9211), pass `--spec-type draft-mtp --spec-draft-n-max=K --ctx-size=65536` and run with `--max-tokens=65536`. The `am17an:mtp-clean` flag `--spec-type mtp` no longer parses on mainline. Targets cycled across `Qwen3.6-27B-Q4_0-MTP.gguf`, `Qwen3.6-27B-UD-Q4_K_XL.gguf`, `Qwen3.6-27B-UD-Q6_K_XL.gguf` (all from [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)) and `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (from the MoE MTP-GGUF repo).
- n-gram cells (`ngram_*.json`): `--spec-type ngram-{simple, mod, cache}` (no `--draft`).

Hardware/software pinning is in [`../../results/02-context.md`](../../results/02-context.md). Differences in llama.cpp build SHA, GPU driver, or chat-template handling will all change the numbers.
