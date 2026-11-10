# Changelog

All notable changes to `llamAmpere-Ultra` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.2] - 2026-11-10

### Added
- **Modular External PLE Table Support (`--ple` / `LLAMA_PLE_GGUF`)**:
  - Added CLI flag `--ple <path>` and environment variable `LLAMA_PLE_GGUF` (and `LLAMA_ARG_PLE`) across `llama-server`, `llama-cli`, and `llama-bench`.
  - Dynamic on-demand mapping of external `per_layer_token_embd.weight` GGUF files in `src/llama-model-loader.cpp`.
  - Enables sharing a single 54 GB PLE table (`Qwen3.8-Flash-Next-PLE-Q8_0.gguf`) across multiple decoupled models (`-noPLE` Gyro-Coder TQ2_0 and pruned 128, 160, 176, 192 Expert models) via lazy NVMe mmap (`TENSOR_READ_LAZY`) without consuming VRAM or host memory.

---

## [0.7.1] - 2026-11-09

### Added
- **Full Synchronization with `cuda-vmm-swap` v2.3.0+**:
  - Engine-Aware Memory Budget Calculator (`router/memory_advisor.py`) supporting Windows WDDM 50% pinned RAM cap and TurboQuant/SJ-KVaRN KV cache sizing.
  - Web Control Studio Dashboard (`router/dashboard.html`) served at `/` and `/ui` with real-time VRAM/RAM gauges and hot-swap controls.
  - MCP tool `vmm_get_memory_budget` added to `mcp/vmm_mcp_server.py`.
  - Master Orchestrator Harness (`master_harness.py`) with `--swap [on|off]`, `--check-memory`, and `--open-ui`.
  - Desktop extensions in `extensions/` including native Paseo plugin (`extensions/paseo-vmm-swap/`) and Pi-coding-agent skill (`extensions/pi-coding-agent/SKILL.md`).
  - Closed-loop autonomous pipeline upgrades (`pipeline/autonomous_pipeline.py`) with 5-stage retry backoff on 502/503 and deep reasoning extraction.

### Changed
- `router/vmm_router.py`: Upgraded to v2.3.0 with strict VRAM eviction barrier (`<3000 MB` physical check via `nvidia-smi`), dynamic swap toggle, and integrated dashboard serving.
- `router/router_config.json`: Added `swap_enabled`, `engine_type`, and `--reasoning-budget` parameters.

---

## [0.7.0] - 2026-11-04

### Added
- **Native CUDA Driver Virtual Memory Management (VMM)**:
  - Low-level driver API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemUnmap`, `cudaHostAlloc`).
  - Sub-60ms physical VRAM unmapping and <2.3s PCIe DMA wake from pinned RAM mirror.
  - Windows WDDM Sliding Window Invariant enforcing strict 2-model residency (<= 24 GB pinned memory).
- **Production Pipeline Suites**:
  - `pipeline_complete_e2e_verified.py`: 4-stage sequential software engineering pipeline with live Python sandbox verification.
  - `pipeline_fast_swap_3models.py`: Automated VMM transition latency and throughput benchmark.
  - `run-gyro-coder.ps1`: 96k context launcher for Gyro-Coder TQ2_0 MoE (19.9 GB VRAM expert layer, 54 GB NVMe PLE).
  - `run-cyber-atx-27b.ps1`: 96k/160k context launcher for Cyber-ATX 27B with MTP speculative draft chain.
- **High-Context 96K Verification**:
  - Validated 98,304 tokens context with FlashAttention (`-fa on`), `--kv-unified`, and `-ctk turbo5 -ctv turbo4`.
  - Added `--load-mode none` to prevent Windows pagefile thrashing during weight offload.

### Changed
- Synchronized with `cuda-vmm-swap` v2.2.0 for OpenAI-compatible routing and IDE MCP integration.

---

## [0.6.2] - 2026-10-30

### Added
- Ampere SM 8.6 MMA Tensor Core dequantization kernels (`GGML_Q8_TURBO3_MMA_FUSED`).
- FlashAttention with 8-bit quantized keys and queries (`GGML_CUDA_FA_I8QK`).
- Multi-Token Prediction (MTP) draft chain with temperature-scaled sampling (`LLAMA_SPEC_CHAIN_SAMPLED`).
- Gyro-Coder TQ2_0 baseline support with A8 PLE N-Gram table memory-mapping.

---

## [0.6.1] - 2026-10-18

### Added
- Context-checkpointing in host RAM (`--cache-ram 16384`, `--ctx-checkpoints 24`).
- Support for `turbo5` / `turbo4` compressed KV cache formats.
- Static CUDA graph optimization for decode passes (`GGML_CUDA_GRAPH_OPT=1`).
