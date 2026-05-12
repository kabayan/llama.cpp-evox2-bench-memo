# data/raw — sanitized bench JSON

Per-run JSON dumps from each bench session referenced in [`../../results/`](../../results/). Sanitization removes the container name and port number that the bench harness recorded (internal lab hints); every measurement value (`tg_ts`, `accept_rate`, `draft_n`, `draft_n_accepted`, `gen_tokens`, `actual_prompt`, `ttft`, `pp_ts`) is preserved verbatim.

## Files

| File | Source on | Contents |
|---|---|---|
| [`specdec_27b_k_sweep.json`](specdec_27b_k_sweep.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + Qwen3.5-{0.8B,2B,4B}-Q4_0 draft, K=1/2/4 + draft size sweep at K=1 (DLS-051 main). |
| [`specdec_27b_k_high.json`](specdec_27b_k_high.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + 0.8B-Q4_0 K=8/16 with fresh baseline (DLS-052). |
| [`specdec_27b_k1_min.json`](specdec_27b_k1_min.json) | 2026-05-12 | Qwen3.5-27B-Q4_0 + 0.8B-Q4_0 K=1 with `--spec-draft-n-min=0` vs `=1` (DLS-050, "no effect at K=1" confirmation). |
| [`specdec_35b_a3b_k4.json`](specdec_35b_a3b_k4.json) | 2026-05-12 | Qwen3.6-35B-A3B-UD-Q6_K + 0.8B-Q4_0 K=4 (DLS-052, MoE pattern). |
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
      "use_draft": true | false,
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
  }
}
```

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
- n-gram cells (`ngram_*.json`): `--spec-type ngram-{simple, mod, cache}` (no `--draft`).

Hardware/software pinning is in [`../../results/02-context.md`](../../results/02-context.md). Differences in llama.cpp build SHA, GPU driver, or chat-template handling will all change the numbers.
