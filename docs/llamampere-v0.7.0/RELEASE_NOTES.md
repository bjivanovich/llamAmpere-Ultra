# llamAmpere-Ultra v0.7.0 Release Notes

**Release Date:** 2026-11-04  
**Engine Scope:** NVIDIA GeForce RTX 3090 / 3090 Ti (Ampere SM 8.6)  
**Primary Breakthrough:** Integrated Native CUDA Driver VMM Fast Swapping (<2.3s) & 96k Multi-Agent Runtime

---

## Highlights

1. **Native CUDA Driver VMM Hot Swapping (<2.3s Transition)**:
   - Full integration of Virtual Memory Management driver primitives (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemUnmap`, `cudaHostAlloc`).
   - Eviction of resident models in under 60 ms releasing physical VRAM back to baseline (540 MB).
   - PCIe 3.0 x16 DMA wake in 1.71s - 2.29s for 14.3 GB to 18.8 GB models at ~8.5 GB/s effective throughput.
   - Preserves constant virtual addresses and execution graphs, eliminating CUDA context rebuilds and graph recompilations.

2. **Windows 11 WDDM Sliding Window Resolution**:
   - Resolves the Windows 24 GB pinned host memory cap (`cudaHostAlloc`) via the Sliding Window Invariant: strictly at most 1 model active in VRAM and 1 model staged in pinned host RAM.
   - Added `--load-mode none` to bypass OS page cache pollution during weight ingestion.

3. **Multi-Agent 96K Pipeline Execution (Architect -> Coder -> Debugger -> Coder Fix)**:
   - Verified 98,304 tokens context window support across both dense and MoE architectures.
   - Integrated `pipeline_complete_e2e_verified.py` and `pipeline_fast_swap_3models.py` demonstrating zero CUDA contention and sustained performance:
     - Stage 1 (Architect - `ATX-Swift-1.5 Q5_K_M`): ~52.5 tok/s decode, 226 tok/s prefill.
     - Stage 2 (Coder - `Gyro-Coder TQ2_0`): 19.9 GB expert layer in VRAM, 54 GB PLE N-gram table on NVMe mmap, sustained 44.0 - 53.5 tok/s decode, 300 tok/s prefill.
     - Stage 3 (Debugger - `Cyber-ATX Q5_K_M`): Forensic analysis and defect detection, 41.6 tok/s decode, up to 750-1010 tok/s prefill.
     - Stage 4 (Coder Fix): Hot-swapped back into clean VRAM to resolve identified defects.

4. **JakeATX Ampere Optimizations**:
   - Fused MMA Q8/Q4 Tensor Core dequantization kernels (`GGML_Q8_TURBO3_MMA_FUSED=1`).
   - 8-bit quantized FlashAttention keys and queries (`GGML_CUDA_FA_I8QK=1`).
   - Static CUDA graph capture and reuse (`GGML_CUDA_GRAPH_OPT=1`).
   - Unified KV cache compression with `turbo5` / `turbo4` keeping resident KV memory under 3.8 GB at 96k context.

---

## File Changes in this Release
- `docs/vmm-swap/`: Complete architecture and benchmark documentation.
- `pipeline_complete_e2e_verified.py`: End-to-end verified autonomous multi-agent pipeline with execution sandbox.
- `pipeline_fast_swap_3models.py`: Rapid sub-2.3s multi-model VMM swap test harness.
- `run-gyro-coder.ps1`: Production launcher for Gyro-Coder TQ2_0 at 96k context with clean VRAM allocation.
- `run-cyber-atx-27b.ps1`: Production launcher for Cyber-ATX 27B Q5_K_M with MTP drafting and VMM sleep.
