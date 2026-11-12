# TurboQuant TQ3_1S and TQ4_1S dp4a Acceleration & Pruned MoE Benchmarks

## 1. Overview

This document reports the implementation details, hardware benchmarks, and microarchitectural analysis of native INT8 `dp4a` execution for TurboQuant weight types (`TQ4_1S` and `TQ3_1S`) on the NVIDIA Ampere architecture (RTX 3090, SM86, 24 GB VRAM).

We evaluated pruned expert variants derived from the **Qwen3.8-Flash-Next base model (512 experts, FP8 source)** specialized for programming workloads, comparing high-precision TurboQuant quantization against the baseline **Gyro-Coder TQ2_0 (256 experts, 2.0 bpw)**.

---

## 2. Kernel Implementation: TQ3_1S dp4a

### INT8 Centroid Quantization Table
`TQ3_1S` uses an 8-entry centroid codebook mapped to INT8 values:
`{-127, -82, -47, -16, 15, 46, 81, 127}`
Rescaling factor: `1.996684 / 127.0`.

### Register-Level Look-Up Table (LUT)
To maximize instruction throughput on CUDA cores, centroids are packed into two 32-bit registers:
- `lut0 = 0xF0D1AE81u` (centroids for indices 0 to 3)
- `lut1 = 0x7F512E0Fu` (centroids for indices 4 to 7)

Extraction of 8 centroids from 3 packed bytes (24 bits = 8 x 3-bit indices) is performed via hardware byte permutation (`__byte_perm` on CUDA / `__builtin_amdgcn_perm` on HIP):

```cpp
__device__ __forceinline__ void tq3_cents8_reg(uint32_t three_bytes, int &c0, int &c1) {
    constexpr uint32_t lut0 = 0xF0D1AE81u;
    constexpr uint32_t lut1 = 0x7F512E0Fu;

    const uint32_t s0 = (three_bytes & 7u)
                      | (((three_bytes >> 3)  & 7u) << 4)
                      | (((three_bytes >> 6)  & 7u) << 8)
                      | (((three_bytes >> 9)  & 7u) << 12);

    const uint32_t p_hi = three_bytes >> 12;
    const uint32_t s1 = (p_hi & 7u)
                      | (((p_hi >> 3) & 7u) << 4)
                      | (((p_hi >> 6) & 7u) << 8)
                      | (((p_hi >> 9) & 7u) << 12);

#if defined(GGML_USE_HIP)
    c0 = (int) __builtin_amdgcn_perm(lut1, lut0, s0);
    c1 = (int) __builtin_amdgcn_perm(lut1, lut0, s1);
#else
    c0 = __byte_perm(lut0, lut1, s0);
    c1 = __byte_perm(lut0, lut1, s1);
#endif
}
```

The resulting 4-byte INT8 vectors are then fed into the `__dp4a` instruction together with dynamically quantized `q8_1` activations.

---

## 3. Comprehensive Benchmark Results on RTX 3090 (24 GB)

Evaluated with `--temp 0.6 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.1` and modular PLE table:

| Model Variant | Experts | Quant Type | Real BPW | Model Size | VRAM Usage | VRAM Free (24GB) | Decode Speed | Status / Assessment |
|---|---|---|---|---|---|---|---|---|
| **108Exp-TQ4_1S** | 108 | TQ4_1S | 4.0 bpw | 19.60 GB | 19.60 GB | **4.97 GB** | **43.0 tok/s** | Verified. Large headroom for >96K context. |
| **120Exp-TQ4_1S** | 120 | TQ4_1S | 4.0 bpw | 21.42 GB | 20.72 GB | **3.85 GB** | **45.2 tok/s** | **Winner / Optimal Sweet Spot**. Double the bits per weight (+5.3 dB SNR) matching Gyro-Coder speed. |
| **144Exp-TQ4_1S** | 144 | TQ4_1S | 4.0 bpw | 25.04 GB | >24.5 GB | 0 GB (OOM) | - | Exceeds 24 GB single-card VRAM ceiling. |
| **160Exp-TQ3_1S** | 160 | TQ3_1S | 3.25 bpw | 22.11 GB | 21.44 GB | **3.13 GB** | **10.1 tok/s** | Verified (>1k tokens continuous). ALU compute-bound. |
| **176Exp-TQ3_1S** | 176 | TQ3_1S | 3.25 bpw | 24.04 GB | ~23.30 GB | **1.27 GB** | **~9.9 tok/s** | Functional; narrow headroom for extended KV cache. |
| **192Exp-TQ3_1S** | 192 | TQ3_1S | 3.25 bpw | 25.98 GB | >24.5 GB | 0 GB (OOM) | - | Exceeds 24 GB single-card VRAM ceiling. |
| **Gyro-Coder** | 256 | TQ2_0 | 2.0 bpw | 18.20 GB | 18.20 GB | **6.37 GB** | **45.0 tok/s** | Baseline. Extremely fast, but aggressive 2-bit quantization. |

---

## 4. Microarchitectural Analysis: Why TQ4_1S Outperforms TQ3_1S

A critical architectural divergence was observed between `TQ4_1S` (45.2 tok/s) and `TQ3_1S` (10.1 tok/s) despite both using `dp4a`:

### TQ4_1S: Hardware-Aligned Nibbles (Memory Bandwidth Bound)
1. **Nibble Alignment:** In `TQ4_1S`, every weight is 4 bits (1 nibble). Exactly two weights fit in a byte; 8 weights fit in a 32-bit word.
2. **Single-Cycle Unpacking:** NVIDIA GPUs have hardware permute instructions (`PRMT`) capable of extracting 8 centroids simultaneously from a register LUT in a single cycle.
3. **Execution Profile:** The GPU Streaming Multiprocessors (SMs) spend minimal cycles unpacking. The execution pipeline is 98% memory bandwidth bound, completely saturating the 936 GB/s GDDR6X bus of the RTX 3090 at **45.2 tok/s**.

### TQ3_1S: Non-Aligned 3-Bit Unpacking (ALU Compute Bound)
1. **Non-Power-of-Two Bitwidth:** 3 bits do not divide evenly into 8, 16, or 32 bits. 32 weights take 12 bytes (4 groups of 3 bytes = 24 bits each).
2. **ALU Inflation:** To extract 8 indices from 24 bits into two 16-bit selector registers, the thread must perform multiple arithmetic shifts (`>>`), bitwise masks (`& 7u`), and bitwise OR (`|`) operations. Per 32-weight block, this requires **88 to 100 integer ALU instructions**.
3. **Execution Profile:** The integer ALU pipeline becomes 100% saturated with unpacking arithmetic before the `dp4a` arithmetic can proceed. The bottleneck shifts from memory bandwidth to ALU instruction issue rate, restricting throughput to **~10.1 tok/s**.

---

## 5. Memory Budget for 96K Context on RTX 3090 (120Exp-TQ4_1S)

| Component | Format | VRAM Allocation | Notes |
|---|---|---|---|
| Model Weights | TQ4_1S (120 Experts) | 18.62 GB | 100% GPU offload (60 layers) |
| Modular PLE Table | Q8_0 | 1.53 GB | Pinned shared embeddings |
| KV Cache (96K context) | `turbo4` | 1.84 GB | 4.25 bits/value with WHT rotation |
| Compute & CUDA Graphs | FP16/FP32 | 0.85 GB | Scratch buffers and graph nodes |
| **Total VRAM** | - | **22.84 GB** | **Fits inside 24.57 GB physical VRAM** |
| **Headroom** | - | **~1.73 GB** | Zero risk of OOM or system RAM spill |

---

## 6. How to Run

### Optimal Environment Configuration
```powershell
$env:LLAMA_MOE_ARENA = "off"
$env:GGML_CUDA_GRAPH_OPT = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"
$env:GGML_TQ_NATIVE = "1"
$env:GGML_CUDA_Q8CACHE = "1"
$env:GGML_CUDA_FUSE_CHAIN = "1"
```

### Execution Example (llama-cli)
```powershell
.\bin\Release\llama-cli.exe `
  -m "D:\models\Qwen3.8-Flash-Next-120Exp-TQ4_1S.gguf" `
  --ple "D:\models\Qwen3.8-Flash-Next-PLE-Q8_0.gguf" `
  -c 32768 `
  -ngl 99 `
  -ctk turbo4 -ctv turbo4 `
  -fa `
  --jinja `
  --temp 0.6 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.1 `
  -p "<|im_start|>user`nEscribe una funcion en C++ para calcular el WHT de 128 elementos.<|im_end|>`n<|im_start|>assistant`n"
```
