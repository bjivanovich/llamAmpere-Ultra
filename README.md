# llamAmpere-Ultra: Autonomous Multi-Agent Runtime & Sub-2.3s CUDA VMM Model Swapping

> **llamAmpere-Ultra**: An integrated high-performance `llama.cpp` runtime tuned for the NVIDIA GeForce RTX 3090 / 3090 Ti (Ampere, SM86).
>
> **Flagship Breakthrough — `cuda-vmm-swap`**: Built-in native **CUDA Driver Virtual Memory Management (VMM)** that hot-swaps 14 to 20 GB models between host RAM and GPU VRAM in **under 2.3 seconds** (`<60 ms` unmap eviction). Enables running sequential multi-agent pipelines (Architect -> Coder -> Debugger) on a single 24 GB GPU without restarting `llama-server` or recompiling CUDA graphs.
>
> Combines JakeATX's Ampere tensor-core kernels (Q8/Q4 MMA, FlashAttention I8QK, SJ-KVaRN / turbo4-5 KV cache up to 200k context), native **Gyro-Coder TQ2_0 MoE engine (AgentionAI)**, an **OpenAI-compatible fast-swap router**, and an **MCP server** for AI coding assistants.

---

### Core Innovations in llamAmpere-Ultra:

1. **Native CUDA Driver VMM Fast Model Swapping (<2.3s)**:
   - Eviction (`Sleep` via `cuMemUnmap`): **<60 ms** to release 18-22 GB of physical VRAM back to baseline (540 MB).
   - Restoration (`Wake` via `cuMemMap` + PCIe DMA): **1.71s - 2.29s** at ~8.5 GB/s effective DMA throughput.
   - Solves the Windows 11 WDDM 24 GB pinned host memory cap via the **Sliding Window Invariant**.
   - Zero CUDA context teardown, zero graph recompilation.
2. **Modular External PLE Table Support (`--ple` / `LLAMA_PLE_GGUF`)**:
   - Dynamic lazy mapping of external `per_layer_token_embd.weight` GGUF tables via NVMe `mmap` (`TENSOR_READ_LAZY`).
   - Enables decoupled inference: a single 54 GB PLE table (`Qwen3.8-Flash-Next-PLE-Q8_0.gguf`) is shared across multiple pruned model variants (`-noPLE` Gyro-Coder TQ2_0 and pruned 128..192 expert models) without consuming GPU VRAM or host memory.
   - Built-in `[PLEPF]` prefetching threadpool for anticipatory page loading ahead of token generation.
3. **Gyro-Coder TQ2_0 & MoE Architecture on RTX 3090 (24 GB)**:
   - MoE 70B (3.8B active) running 100% in GPU VRAM (22.6 GB VRAM at 64k - 96k context) querying the 54 GB PLE table on NVMe via `mmap` at **44.86 tok/s sustained decode** (TTFT 1.6s).
   - Comparative 64k Context Benchmark:
     - **Gyro-Coder TQ2_0 (64K context, Q8_0 KV Cache)**: **44.86 tok/s** (100% GPU VRAM residency via native TQ2_T engine).
     - **Qwen3.8-176Exp Q3_K_S (with LLAMA_MOE_ARENA=off)**: **36.33 - 44.78 tok/s** (native GPU execution with PLE N-gram prefetch; bypassing the Linux-specific host arena fallback).
4. **Enhanced CLI & Model Tuning Infrastructure**:
   - Universal `--override-kv` type parser supporting standard GGUF keys (`u32:`, `uint:`, `i32:`, `i64:`, `f32:`, `int:`, `str:`, `bool:`).
   - Native `--jinja` template engine integration for Qwen4Exp official chat templates.
5. **Multi-Agent Infrastructure (Zero-Config)**:
   - `master_harness.py`: Supervisor CLI orchestrator with `--swap [on|off]`, `--check-memory`, and `--open-ui`.
   - `router/vmm_router.py`: OpenAI-compatible reverse proxy (`http://127.0.0.1:8000/v1`) with mutual exclusion (`asyncio.Lock`), strict VRAM eviction barrier (`<3000 MB`), and dynamic swap toggle.
   - `router/memory_advisor.py`: Engine-Aware Memory Budget Calculator & Windows WDDM 50% pinned RAM cap guardian.
   - `router/dashboard.html`: Integrated Web Control Studio dashboard served at `http://127.0.0.1:8000/ui`.
   - `pipeline/autonomous_pipeline.py`: Closed-loop 4-stage pipeline runner (Architect -> Coder -> Sandbox -> Debugger -> Coder Fix) with 502/503 retry tolerance.
   - `mcp/vmm_mcp_server.py`: Native Model Context Protocol server exposing `vmm_get_memory_budget`, `vmm_list_models`, `vmm_swap_model`, and `vmm_query_agent`.
   - `extensions/`: Native desktop integrations including Paseo plugin (`extensions/paseo-vmm-swap/`) and Pi-coding-agent skill (`extensions/pi-coding-agent/`).
   - `scripts/sync_upstream.py`: Automated tool to monitor and sync updates from upstream JakeATX/llamAmpere, Strata, and AgentionAI.

> **Standalone VMM Swap Repository:**
> If you only need the engine-agnostic VMM hot-swapping patch, router, and MCP server to integrate into upstream `llama.cpp` or another custom runtime without cloning the full engine, see:
> **[cuda-vmm-swap](https://github.com/bjivanovich/cuda-vmm-swap)**.

---

> **llamAmpere v0.5.4**: a llama.cpp fork tuned for the RTX 3090 / 3090 Ti (Ampere, SM86), running Qwen3.8-27B with its MTP draft head.

llamAmpere is a fast version of llama.cpp (the open-source program that runs AI language models on your own computer),
tuned for NVIDIA RTX 30-series "Ampere" graphics cards, especially the RTX 3090 and 3090 Ti. It runs Qwen3.8 models,
including Qwen3.8-27B with its built-in MTP draft head (a small helper that guesses several words ahead so the model
writes faster) and Qwen3.8 Flash-Next.

## What's new in v0.6.3.3

- Small cards use memory-bounded prompt passes and shorter prompt-tail chunks by default. Normally free 24 GB cards keep their previous plan.
- Fixes Windows private model mappings and MSVC atomics (#94, #93, #82). Linux launchers count physical performance cores on hybrid Intel CPUs.
- Adds CPU expert-split diagnostics and `--moe-cpu-share` for troubleshooting (#88). [Release notes and hardware limits](docs/llamampere-v0.6.3.3/RELEASE_NOTES.md).

## What's new in v0.6.3.2

- Fixes a Flash-Next crash when a follow-up request reuses a cached prompt with shared experts on a side stream (#88).
- Fixes Windows model loading, Windows build errors, and reads from large expert overlays (#82). [Release notes](docs/llamampere-v0.6.3.2/RELEASE_NOTES.md).

## What's new in v0.6.3.1

- Flash-Next leaves more temporary card memory on 24 GB cards and handles `--parallel` startup. Unsupported Flash-Next tensor split now gives a clear error.
- Small cards keep 32K prompt passes, with a 4096-token microbatch to avoid a measured prompt fallback. Adaptive prompt passes remain opt-in. [Release notes](docs/llamampere-v0.6.3.1/RELEASE_NOTES.md).

## What's new in v0.6.3

- **New Flash-Next model file, [qwen3.8-flash-next-Swift1.5-ATX-AI3](https://huggingface.co/jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3):** 80.11 GB (74.61 GiB), compared with 75.97 GB for IQ3_XXS. On an RTX 3090 with ample system RAM, this file wrote 121–125 tokens/s across coding, agentic and RAG tasks. These are observations on the new file, not a same-engine comparison with the standard file. [Measured results and limits](docs/llamampere-v0.6.3/RELEASE_NOTES.md).
- The expert kernels our speed tests use are now on by default; whole-file page read-ahead when the model fits in RAM.

## What's new in v0.6.1

- **Faster Flash-Next:** about 3% faster writing and prompt reading than v0.6 on an RTX 3090 Ti (113.15 vs 109.65 writing tokens/s, 2,751 vs 2,672 prompt tokens/s, 100K prompt; one-seed observation), and room for more experts on the card. Fixes the 12 GB startup crash. [v0.6.1 release notes](docs/llamampere-v0.6.1/RELEASE_NOTES.md).

## What's new in v0.6

- **mb-MTP is on by default** with the MTP draft head: more guessed words are kept, so writing is faster (`--no-spec-mb-mtp` gives exact sampling).
- **Faster prompt reading on Swift 1.5 27B** (+5.3% at 100K tokens) with the same output.
- **AI+ model file for Swift 1.5 27B**, our own weight format for Ampere and newer cards: 14.4 GB, 10.4% lower KL (a measure of drift from the full-precision model) than IQ4_XS-M. Older AI+ files no longer load.
- **Qwen3.8 Flash-Next** support: experts split between the card and RAM, with `--kv-gpu-tokens N` to keep only part of the context memory on the card.

Details, speed tables and everything older: [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).


## Quick start

**Build and run (v0.6.3.3, Qwen3.8 Flash-Next, Linux).** For one RTX 30-series card with enough system RAM for the model and prompt workspace. Small-card tests used a 3090 limited to 12 GB with about 90 GiB of system RAM; native 12/16 GB cards remain unmeasured, and 64 GB of system RAM is not a fit promise. Small cards use memory-bounded prompt passes; see the [hardware limits](docs/llamampere-v0.6.3.3/RELEASE_NOTES.md). The launcher keeps the tested `--fit off` setting; its own card planner still chooses the memory plan and keeps at least 5 GiB of RAM free. Automatic `--fit on` has not been measured on this file. Needs the CUDA 12 or newer toolkit (`nvcc`), CMake, git, a C++
compiler, the `hf` download tool (`pip install -U huggingface_hub`) and 81.63 GB for the model and drafter, plus space for the build, prompt scratch and saved prompts. The launcher uses a 5 GiB free-RAM floor. This
one paste clones or updates llamAmpere at v0.6.3.3, builds `llama-server`, downloads the model and its drafter and starts the server:

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.6.3.3
else
  git clone -b v0.6.3.3 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
hf download jakeatx/qwen3.8-flash-next-Swift1.5-ATX-AI3 --include '*.gguf' --local-dir models/flash-next && \
hf download jakeatx/Swift-1.5-Qwen3.8-Flash-Next-MTP-Drafter-GGUF mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf --local-dir models/flash-next && \
LLAMA_SERVER_BIN=build-sm86/bin/llama-server \
FLASH_NEXT_DRAFTER=models/flash-next/mtp-Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2S-B.gguf \
scripts/flash-next-server.sh -m models/flash-next/qwen3.8-flash-next-Swift1.5-ATX-AI3-00001-of-00002.gguf
```

To start it again later, run the last three lines from the `llamAmpere` folder. The server listens on http://127.0.0.1:8080 (OpenAI-compatible API).

**Build and run (v0.6.3.3, Swift 1.5 27B at 2.3 bpw, 200K context, one 24 GB RTX 30-series card, Linux).** Needs the CUDA 12 or newer
toolkit (`nvcc`), CMake, git and a C++ compiler. The model (about 9.1 GB) downloads on first start. This one paste clones or
updates llamAmpere at v0.6.3.3, builds `llama-server` and starts it:

```bash
if [ -d llamAmpere ]; then
  git -C llamAmpere fetch --tags --force && git -C llamAmpere checkout v0.6.3.3
else
  git clone -b v0.6.3.3 https://github.com/JakeATX/llamAmpere.git
fi && cd llamAmpere && \
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_BORINGSSL=ON && \
cmake --build build-sm86 -j8 --target llama-server && \
./build-sm86/bin/llama-server -hf jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2 \
  -c 204800 -ngl 99 -fa on -b 4096 -ub 1024 --parallel 1 --jinja --no-context-shift \
  -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 4 \
  --spec-draft-p-min 0 --spec-mb-mtp
```

Commands for 12 and 10 GB cards (not yet run on this model) are in the [v0.6.2 release notes](https://github.com/JakeATX/llamAmpere/releases/tag/v0.6.2).

The server listens on http://127.0.0.1:8080 (OpenAI-compatible API). Add `--no-spec-mb-mtp` for exact drafting.
To use the IQ4_XS-M file (15.6 GB, also runs in mainline llama.cpp) instead, use
`--hf-repo jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF --hf-file ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.gguf`.

## Headline results

Numbers for this release (Swift 1.5 27B, v0.6 against v0.5) are in the [v0.6 release notes](docs/llamampere-v0.6/RELEASE_NOTES.md).

## Links

- Release notes: [v0.6.3.3](docs/llamampere-v0.6.3.3/RELEASE_NOTES.md), [v0.6.3.2](docs/llamampere-v0.6.3.2/RELEASE_NOTES.md), [v0.6.3.1](docs/llamampere-v0.6.3.1/RELEASE_NOTES.md), [v0.6.3](docs/llamampere-v0.6.3/RELEASE_NOTES.md), [v0.6.2](docs/llamampere-v0.6.2/RELEASE_NOTES.md), [v0.6.1](docs/llamampere-v0.6.1/RELEASE_NOTES.md), [v0.6](docs/llamampere-v0.6/RELEASE_NOTES.md), [v0.5](docs/llamampere-v0.5/RELEASE_NOTES.md), [v0.4](docs/llamampere-v0.4/RELEASE_NOTES.md), [v0.3.1](docs/llamampere-v0.3.1/RELEASE_NOTES.md); recommended settings for each card size: [v0.5 RECOMMENDED.md](docs/llamampere-v0.5/RECOMMENDED.md)
- Models: [huggingface.co/jakeatx](https://huggingface.co/jakeatx)
- Background: [QWEN_AMPERE.md](QWEN_AMPERE.md), [write-up](docs/llamampere-v0.3/ARTICLE.md), flags in [docs/speculative.md](docs/speculative.md) and [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md)
- SJ-KVaRN paper: [paper](https://huggingface.co/spaces/jakeatx/sj-kvarn-paper), codec [JakeATX/sj-kvarn-codec](https://github.com/JakeATX/sj-kvarn-codec) (MIT)
- Upstream llama.cpp README (quick start, backends, docs index): [docs/llama.cpp-upstream-README.md](docs/llama.cpp-upstream-README.md)

## License and credits

llamAmpere is MIT licensed (see [LICENSE](LICENSE)). It is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
by the ggml authors and includes work from TheTom's TurboQuant fork. It is the successor of
[llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

Third-party libraries (from upstream llama.cpp):

## Contributing

- Contributors can open PRs
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Upstream Synchronization

Keep llamAmpere-Ultra up to date with upstream ecosystem developments:

`ash
# Monitor upstream releases from JakeATX, Strata, and AgentionAI
python scripts/sync_upstream.py

# Or one-click on Windows PowerShell
.\scripts\sync-upstream.ps1
`

## Credits & Acknowledgements

This project builds upon the groundbreaking work of the open-source AI and CUDA systems community:

- **[JakeATX / llamAmpere](https://github.com/JakeATX/llamAmpere)**: Core foundation of the SM86 Ampere execution engine, custom fused MMA kernels (Q8/Q4), FlashAttention I8QK integration, and SJ-KVaRN / turbo4-5 compressed KV cache architectures enabling 96k - 200k+ context windows.
- **[AgentionAI](https://huggingface.co/agentionai)**: Creators of Gyro-Coder TQ2_0 (MoE 70B with 3.8B active) and the A8 PLE N-Gram speculative table methodology, delivering near-zero syntax error code synthesis.
- **[Niko1221 / Strata](https://github.com/Niko1221/Strata)**: Conceptual and architectural pioneer in local LLM offloading, memory tiering, and domain expert cache optimization for consumer hardware.
- **u/enndeeee**: Original author of the CUDA Driver VMM SleepableBuffer concept in [Infernix PR #43](https://github.com/infernix/llama.cpp/pull/43), which inspired our Windows WDDM sliding-window VMM model swapper.
- **[ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp)**: The underlying LLM inference framework.
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
