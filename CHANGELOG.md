# Changelog

All notable changes to `llamAmpere-Ultra` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
