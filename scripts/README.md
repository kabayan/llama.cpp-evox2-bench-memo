# Bench scripts

Single-cell bench runner. Each invocation starts one `llama-server`, runs warmup + N measure runs over three real prompts, writes results to JSON, then tears down.

For multi-cell sweeps, drive this script from a shell loop. The script intentionally does **not** support parallel runs or in-process multi-cell sweeps — GPU sharing perturbs `tg` by 10-30% on Strix Halo, and we want every cell to start from a cold server.

## Dependencies

- Python 3.10+ (uses `from __future__ import annotations` and `X | None` syntax)
- `httpx` (`pip install httpx`)
- A running container built from [`../docker/`](../docker/) with the GGUFs mounted at `/gguf` (or wherever you configure)

## Quick start

```bash
# 1. build and start the container (one-time)
docker build -f docker/Dockerfile.mtp-vulkan -t llama-cpp-evox2-bench .
docker run -d --name llama-evox2 --device /dev/dri \
  -v /path/to/gguf:/gguf:ro -p 10001:10001 \
  llama-cpp-evox2-bench

# 2. baseline (no spec-dec)
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.5-27B-Q4_0.gguf \
  --output bench_baseline.json

# 3. draft 0.8B + K=4 (the recommended config — see results/00-quick-take.md)
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.5-27B-Q4_0.gguf \
  --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --draft-n-max 4 --draft-n-min 1 \
  --output bench_k4.json

# 4. push for max P_code speedup
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.5-27B-Q4_0.gguf \
  --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --draft-n-max 16 --draft-n-min 1 \
  --prompts P_code \
  --output bench_k16_code.json
```

## K sweep loop

```bash
for k in 0 1 2 4 8 16; do
  if [ "$k" = "0" ]; then
    python scripts/run_bench.py \
      --container llama-evox2 \
      --target /gguf/Qwen3.5-27B-Q4_0.gguf \
      --output bench_K${k}.json
  else
    python scripts/run_bench.py \
      --container llama-evox2 \
      --target /gguf/Qwen3.5-27B-Q4_0.gguf \
      --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
      --draft-n-max ${k} --draft-n-min 1 \
      --output bench_K${k}.json
  fi
done
```

Each cell takes ~5-9 min on Strix Halo (27B-Q4_0 target, 512 max_tokens, 1 warmup + 3 runs over 3 prompts). The whole sweep is ~30-50 min.

## Draft size sweep

```bash
for draft in 0.8B 2B 4B; do
  python scripts/run_bench.py \
    --container llama-evox2 \
    --target /gguf/Qwen3.5-27B-Q4_0.gguf \
    --draft /gguf/Qwen3.5-${draft}-Q4_0.gguf \
    --draft-n-max 1 --draft-n-min 1 \
    --output bench_draft_${draft}.json
done
```

(K=1 isolates the draft's per-step overhead from the verify-batch scaling.)

## 35B-A3B (MoE) variant

```bash
python scripts/run_bench.py \
  --container llama-evox2 \
  --target /gguf/Qwen3.6-35B-A3B-UD-Q6_K.gguf \
  --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --draft-n-max 4 --draft-n-min 1 \
  --reasoning-format auto \
  --output bench_35b_k4.json
```

(Expected: ~+11%, with P_chat going negative. See [`../results/00-quick-take.md`](../results/00-quick-take.md) for the comparison.)

## Output JSON shape

```jsonc
{
  "started_at": "2026-05-12T16:00:00",
  "finished_at": "2026-05-12T16:08:23",
  "config": { ... CLI args ... },
  "model_name": "Qwen3.5-27B-Q4_0.gguf",
  "prompts": {
    "P_code": {
      "warmup": { "tg_ts": 27.50, "accept_rate": 0.97, "draft_n": 349, ... },
      "tg_med": 27.36,
      "tg_min": 26.63,
      "tg_max": 27.36,
      "accept_med": 0.989,
      "draft_n_med": 349,
      "gen_tokens_med": 512,
      "runs": [ { ... per-run metrics ... }, ... ]
    },
    "P_chat": { ... },
    "P_reason": { ... }
  }
}
```

Key fields:
- **`tg_med`** — median tokens-generated-per-second. The headline number.
- **`accept_med`** — fraction of drafted tokens accepted by the target. At K=1 + real prompts you'll see 1.0 (greedy match); at K=4 expect 0.96-0.99; at K=16 expect 0.92-0.97.
- **`draft_n_med`** — total drafted tokens across the response. Divide by `gen_tokens_med` to get the fraction of output tokens produced by spec-dec rounds.

## Computing speedup

`speedup = tg_med(spec-dec) / tg_med(baseline)`. The baseline JSON has the same shape but with `accept_med = 0` and `draft_n_med = 0`.

A quick analysis loop:

```python
import json
from pathlib import Path

baseline = json.loads(Path("bench_baseline.json").read_text())
for k in [1, 2, 4, 8, 16]:
    cell = json.loads(Path(f"bench_K{k}.json").read_text())
    print(f"K={k}:")
    for p in ["P_code", "P_chat", "P_reason"]:
        b = baseline["prompts"][p]["tg_med"]
        c = cell["prompts"][p]["tg_med"]
        a = cell["prompts"][p]["accept_med"]
        print(f"  {p}: {c:.2f} t/s ({c/b:.3f}x, acc {a*100:.1f}%)")
```

## When the bench surprises you

The two failure modes to watch:

1. **`tg` is much lower than expected.** Check the run-to-run variance. If `tg_max / tg_min > 1.1`, something else was using the GPU (a desktop session, a leaked llama-server from a previous run, another container). The script does `pkill llama-server` at start, but external processes can still steal GPU time.
2. **`accept_rate` is near zero.** Either the draft and target tokenizers don't match (vocab size mismatch — check the `n_vocab` lines in server log) or the draft is a fundamentally different model (e.g. an old Qwen3 dense vs. Qwen3.5 family). The script doesn't validate compatibility upfront; the server log is your source of truth.

## Programmatic access

`run_bench.py` is single-file. Import from a Jupyter notebook or wrap it in your own analysis with `subprocess.run(["python", "scripts/run_bench.py", ...])`. The output JSON is the stable interface.
