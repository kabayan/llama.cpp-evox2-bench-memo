# Docker setup

Minimum-reproduction Docker image for the bench in [../results/](../results/). Builds `llama.cpp` from [`am17an:mtp-clean@5d5f1b46`](https://github.com/ggml-org/llama.cpp/pull/22673) with two small patches applied on top, targeting Vulkan on AMD Strix Halo (`gfx1151`).

## Build

```bash
# from repository root
docker build -f docker/Dockerfile.mtp-vulkan -t llama-cpp-evox2-bench .
```

The build downloads ~1 GB of llama.cpp source and compiles Vulkan kernels (~5-10 min on Strix Halo).

## Run

Container starts idle (`ENTRYPOINT ["tail", "-f", "/dev/null"]`). The bench scripts in [`../scripts/`](../scripts/) use `docker exec` to start `llama-server` per cell.

```bash
# adjust GGUF host path to where your *.gguf files live
docker run -d \
  --name llama-evox2 \
  --device /dev/dri \
  -v /path/to/gguf:/gguf:ro \
  -p 10001:10001 \
  llama-cpp-evox2-bench

# verify the binary is present
docker exec llama-evox2 /app/build/bin/llama-server --version
# expected: version: 9032 (5d5f1b46e) ... built with cc ...
```

## Inside the image

| Path | Notes |
|---|---|
| `/app/` | llama.cpp source tree (cloned at build time) |
| `/app/build/bin/llama-server` | server binary (also symlinked as `/app/llama-server`) |
| `/app/build/bin/llama-bench` | offline bench binary (used by `llama-bench` mode, not by our scripts) |
| `/tmp/patches/` | the two `.patch` files applied during build |

## Patches applied

1. **`01_ngram_simple_continue_search.patch`** (`common/ngram-map.cpp`): the upstream `ngram-simple` algorithm stops at the first backward match. For periodic-text sources the first match is often closer to the cursor than `n_draft_min`, so the entire draft gets rejected. This patch continues searching past matches that are too close.
2. **`02_ngram_simple_state_respect_cli.patch`** (`common/speculative.cpp`): the upstream `common_speculative_state_ngram_simple` class hardcodes `n_max == n_min == size_mgram` (24 by default), ignoring `--spec-draft-n-max` / `--spec-draft-n-min`. This patch makes both methods read from `params.draft.n_max` / `n_min` (with `min_clamp = 1`).

With both patches, raw `/completion` + `lorem ipsum` reaches **5.74×** speedup with `ngram-simple`. **They do not affect the draft-model results** in [`../results/`](../results/) — the K-sweep numbers were measured with the draft-model spec-dec path, which doesn't use this code.

Patches are not (yet) submitted upstream; they are minimal fixes to make `ngram-simple` work as advertised in PR #22673. See the headers in each `.patch` file for details on the original bug.

## Pin / SHA stability

The Dockerfile pins:
- llama.cpp ref: `am17an:mtp-clean` at SHA `5d5f1b46e4f56885801c86363d4677a5f72f83af`
- Base image: `ubuntu:24.04`
- Vulkan: apt-installed packages from Ubuntu Noble main repo

The `am17an:mtp-clean` branch is the source of upstream PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673). The PR is open / unmerged as of 2026-05-12. If you change `LLAMA_CPP_REF` in the Dockerfile, the patches need to be re-validated with `git apply --check` — the patch hunks are anchored to specific line numbers that can shift.

## Running on other hardware

This Dockerfile is gfx1151-specific in that it disables ROCm flash-attention (see [llama.cpp #12629](https://github.com/ggml-org/llama.cpp/issues/12629)). On NVIDIA hardware, swap `-DGGML_VULKAN=ON` for `-DGGML_CUDA=ON` and adjust the base image. On other AMD GPUs (gfx1100, etc.) the Vulkan build should work but the flash-attention rule may not apply — test with a small target first.
