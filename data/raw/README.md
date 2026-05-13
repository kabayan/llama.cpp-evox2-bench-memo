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
| [`specdec_qwen36_35b_a3b_mtp_sweep.json`](specdec_qwen36_35b_a3b_mtp_sweep.json) | 2026-05-12 | Qwen3.6-35B-A3B-UD-Q4_K_XL + built-in MTP head (MoE variant), K=1/2/3/4/5/8 (DLS-054). Same `--spec-type mtp` setup as the 27B file. K=2 peak / K=3 (Unsloth recipe) suboptimal / K=8 all-prompt collapse. |
| [`ngram_35b_a3b_real.json`](ngram_35b_a3b_real.json) | 2026-05-11 | 35B-A3B + none / ngram-{simple, mod, cache} on real prompts (post-patch ngram-simple). |
| [`ngram_35b_a3b_cache_oracle.json`](ngram_35b_a3b_cache_oracle.json) | 2026-05-11 | 35B-A3B + ngram-cache static (oracle per-prompt corpus, project repo corpus) vs ngram-mod. |
| [`ngram_simple_size_n3.json`](ngram_simple_size_n3.json) | 2026-05-11 | 35B-A3B + ngram-simple `--spec-ngram-simple-size-n=3` vs `=12` (DLS-049 rejection). |
| [`ngram_27b_lorem.json`](ngram_27b_lorem.json) | 2026-05-11 | 27B-Q4_0 + ngram-simple / ngram-mod, lorem-ipsum 4096→256 legacy workload (DLS-041). |

## Schema

Two layouts coexist:

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
- n-gram cells (`ngram_*.json`): `--spec-type ngram-{simple, mod, cache}` (no `--draft`).

Hardware/software pinning is in [`../../results/02-context.md`](../../results/02-context.md). Differences in llama.cpp build SHA, GPU driver, or chat-template handling will all change the numbers.
