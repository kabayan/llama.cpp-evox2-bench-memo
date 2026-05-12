#!/usr/bin/env python3
"""Single-cell bench runner for llama.cpp speculative decoding.

Starts a llama-server inside a running container, runs warmup + N measurement
runs over the three real prompts, and writes per-run metrics to a JSON file.

This is intentionally minimal:
- one cell per invocation (use a shell loop for sweeps)
- no parallel runs (GPU sharing perturbs tg by 10-30% on Strix Halo)
- depends only on `httpx` (`pip install httpx`)

Example
-------
    # baseline (no spec-dec)
    python scripts/run_bench.py \\
        --container llama-evox2 \\
        --target /gguf/Qwen3.5-27B-Q4_0.gguf \\
        --output baseline.json

    # draft 0.8B + K=4 (the recommended config)
    python scripts/run_bench.py \\
        --container llama-evox2 \\
        --target /gguf/Qwen3.5-27B-Q4_0.gguf \\
        --draft /gguf/Qwen3.5-0.8B-Q4_0.gguf \\
        --draft-n-max 4 \\
        --draft-n-min 1 \\
        --output k4.json

The output JSON has one entry per prompt with per-run and median values for
tg, pp, ttft, accept_rate, draft_n, gen_tokens. See `results/00-quick-take.md`
for how to interpret the numbers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

from prompts import PROMPTS, get as get_prompt


def quote(s: str) -> str:
    if " " in s or '"' in s:
        return "'" + s.replace("'", "'\\''") + "'"
    return s


def stop_server(container: str) -> None:
    subprocess.run(
        ["docker", "exec", container, "pkill", "-9", "-f", "llama-server"],
        capture_output=True,
    )
    time.sleep(2)


def start_server(
    container: str,
    target: str,
    draft: str | None,
    draft_n_max: int,
    draft_n_min: int,
    port: int,
    ctx: int,
    binary: str,
    gpu_layers: int,
    reasoning_format: str,
) -> None:
    cmd_parts = [
        binary,
        "-m", target,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-layers", str(gpu_layers),
        "--ctx-size", str(ctx),
        "--jinja",
        "--reasoning-format", reasoning_format,
        "--temp", "0.0",
        "-np", "1",
        "--no-warmup",
    ]
    if draft:
        cmd_parts.extend([
            "-md", draft,
            "-ngld", str(gpu_layers),
            "--spec-draft-n-max", str(draft_n_max),
            "--spec-draft-n-min", str(draft_n_min),
        ])
    inner = " ".join(quote(x) for x in cmd_parts) + " > /tmp/server.log 2>&1"
    subprocess.run(
        ["docker", "exec", "-d", container, "bash", "-c", inner],
        check=True,
    )


def wait_ready(port: int, timeout: float = 600.0) -> bool:
    start = time.time()
    url = f"http://localhost:{port}/v1/models"
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200 and "data" in r.json():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def get_model_name(port: int) -> str:
    try:
        r = httpx.get(f"http://localhost:{port}/v1/models", timeout=5)
        data = r.json().get("data", [])
        if data:
            return data[0].get("id", "model")
    except Exception:
        pass
    return "model"


def get_server_log_tail(container: str, lines: int = 80) -> str:
    out = subprocess.run(
        ["docker", "exec", container, "tail", f"-{lines}", "/tmp/server.log"],
        capture_output=True, text=True,
    )
    return out.stdout


def measure_one(
    port: int,
    model_name: str,
    prompt_text: str,
    max_tokens: int,
    read_timeout: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    ttft: float | None = None
    gen_tokens = 0
    actual_prompt = None
    draft_n = 0
    draft_n_accepted = 0
    http_timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0)
    with httpx.Client(timeout=http_timeout) as client:
        with client.stream(
            "POST",
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0.0,
                "cache_prompt": False,
            },
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.read()[:200]}")
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    actual_prompt = usage["prompt_tokens"]
                t = chunk.get("timings")
                if isinstance(t, dict):
                    if isinstance(t.get("draft_n"), int) and t["draft_n"] > draft_n:
                        draft_n = t["draft_n"]
                    if isinstance(t.get("draft_n_accepted"), int) and t["draft_n_accepted"] > draft_n_accepted:
                        draft_n_accepted = t["draft_n_accepted"]
                choices = chunk.get("choices") or []
                d = choices[0].get("delta", {}) if choices else {}
                delta = d.get("content") or d.get("reasoning_content") or ""
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    gen_tokens += 1
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    decode = total - ttft
    pp_ts = (actual_prompt / ttft) if (actual_prompt and ttft > 0) else 0.0
    tg_ts = (gen_tokens / decode) if decode > 0 else 0.0
    accept = (draft_n_accepted / draft_n) if draft_n > 0 else 0.0
    return {
        "ttft": ttft,
        "pp_ts": pp_ts,
        "tg_ts": tg_ts,
        "total": total,
        "decode": decode,
        "gen_tokens": gen_tokens,
        "actual_prompt": actual_prompt,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "accept_rate": accept,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--container", required=True,
                    help="Docker container name (running, idle, with /gguf mounted)")
    ap.add_argument("--target", required=True,
                    help="Path inside the container to the target GGUF")
    ap.add_argument("--draft", default=None,
                    help="Path inside the container to the draft GGUF (omit for baseline / no spec-dec)")
    ap.add_argument("--draft-n-max", type=int, default=4,
                    help="--spec-draft-n-max (default: 4, the recommended setting)")
    ap.add_argument("--draft-n-min", type=int, default=1,
                    help="--spec-draft-n-min (default: 1; do NOT set equal to --draft-n-max at K>=8)")
    ap.add_argument("--port", type=int, default=10001,
                    help="Port the llama-server listens on (default: 10001)")
    ap.add_argument("--ctx", type=int, default=16384,
                    help="--ctx-size (default: 16384)")
    ap.add_argument("--binary", default="/app/build/bin/llama-server",
                    help="Path inside the container to llama-server")
    ap.add_argument("--gpu-layers", type=int, default=99,
                    help="--gpu-layers / -ngld (default: 99 = all)")
    ap.add_argument("--reasoning-format", default="auto",
                    help="--reasoning-format (default: auto; some models need 'deepseek')")
    ap.add_argument("--prompts", default="P_code,P_chat,P_reason",
                    help="Comma-separated prompt names from prompts.py (default: all three)")
    ap.add_argument("--runs", type=int, default=3,
                    help="Number of measurement runs after warmup (default: 3)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="--max_tokens per request (default: 512)")
    ap.add_argument("--read-timeout", type=float, default=600.0,
                    help="HTTP read timeout in seconds (default: 600)")
    ap.add_argument("--output", default="bench_result.json",
                    help="Output JSON path (default: ./bench_result.json)")
    args = ap.parse_args()

    prompt_names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for pn in prompt_names:
        get_prompt(pn)  # raises KeyError if unknown

    output_path = Path(args.output)
    out: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "container": args.container,
            "target": args.target,
            "draft": args.draft,
            "draft_n_max": args.draft_n_max,
            "draft_n_min": args.draft_n_min,
            "port": args.port,
            "ctx": args.ctx,
            "binary": args.binary,
            "gpu_layers": args.gpu_layers,
            "reasoning_format": args.reasoning_format,
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "prompts": prompt_names,
        },
        "prompts": {},
    }
    output_path.write_text(json.dumps(out, indent=2))

    print(f"== {'baseline (no spec-dec)' if args.draft is None else f'draft={Path(args.draft).name} K={args.draft_n_max} min={args.draft_n_min}'} ==",
          flush=True)

    stop_server(args.container)
    try:
        start_server(
            args.container, args.target, args.draft,
            args.draft_n_max, args.draft_n_min,
            args.port, args.ctx, args.binary,
            args.gpu_layers, args.reasoning_format,
        )
    except subprocess.CalledProcessError as e:
        out["error"] = f"docker exec start_server failed: {e}"
        output_path.write_text(json.dumps(out, indent=2))
        print(out["error"], file=sys.stderr)
        sys.exit(1)

    print("  waiting for llama-server ready ...", flush=True)
    if not wait_ready(args.port, timeout=600):
        tail = get_server_log_tail(args.container)
        out["error"] = "llama-server ready timeout"
        out["log_tail"] = tail
        output_path.write_text(json.dumps(out, indent=2))
        print("TIMEOUT. Last server log lines:", file=sys.stderr)
        print(tail, file=sys.stderr)
        sys.exit(1)

    model_name = get_model_name(args.port)
    print(f"  ready. model_name={model_name}", flush=True)
    out["model_name"] = model_name

    for p_id in prompt_names:
        p_text = get_prompt(p_id)
        print(f"  [{p_id}]", flush=True)
        try:
            w = measure_one(args.port, model_name, p_text,
                            args.max_tokens, args.read_timeout)
            print(f"    warmup: tg={w['tg_ts']:.2f} t/s, gen={w['gen_tokens']},"
                  f" accept={w['accept_rate']*100:.1f}% (draft_n={w['draft_n']})",
                  flush=True)
        except Exception as e:
            out["prompts"][p_id] = {"error": f"warmup: {e}"}
            output_path.write_text(json.dumps(out, indent=2))
            continue
        runs = []
        try:
            for i in range(args.runs):
                r = measure_one(args.port, model_name, p_text,
                                args.max_tokens, args.read_timeout)
                print(f"    run{i+1}: tg={r['tg_ts']:.2f} t/s, pp={r['pp_ts']:.2f},"
                      f" ttft={r['ttft']*1000:.0f}ms, gen={r['gen_tokens']},"
                      f" accept={r['accept_rate']*100:.1f}% (draft_n={r['draft_n']})",
                      flush=True)
                runs.append(r)
        except Exception as e:
            out["prompts"][p_id] = {"error": f"run: {e}", "runs": runs}
            output_path.write_text(json.dumps(out, indent=2))
            continue

        def med(key: str) -> float:
            vs = sorted(r[key] for r in runs)
            return vs[len(vs) // 2]

        out["prompts"][p_id] = {
            "warmup": w,
            "tg_med": med("tg_ts"),
            "tg_min": min(r["tg_ts"] for r in runs),
            "tg_max": max(r["tg_ts"] for r in runs),
            "pp_med": med("pp_ts"),
            "ttft_med": med("ttft"),
            "accept_med": med("accept_rate"),
            "draft_n_med": med("draft_n"),
            "gen_tokens_med": med("gen_tokens"),
            "actual_prompt_med": med("actual_prompt") if all(r["actual_prompt"] for r in runs) else None,
            "runs": runs,
        }
        output_path.write_text(json.dumps(out, indent=2))

    stop_server(args.container)
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\n== DONE. results at {output_path} ==", flush=True)


if __name__ == "__main__":
    main()
