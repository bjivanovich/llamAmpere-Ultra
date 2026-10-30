import os
import sys

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"
agent_root = r"C:\Users\bj_iv\Desktop\agentionai-llama"

def patch_file(rel_path, search_str, replace_str):
    target = os.path.join(ultra_root, rel_path)
    with open(target, 'r', encoding='utf-8') as fp:
        content = fp.read()
    if search_str not in content:
        print(f"FAILED to find target in {rel_path}: '{search_str[:50]}...'")
        return False
    content = content.replace(search_str, replace_str, 1)
    with open(target, 'w', encoding='utf-8') as fp:
        fp.write(content)
    print(f"Successfully patched {rel_path}")
    return True

print("=== 1. Patching ggml/include/ggml.h ===")
patch_file(
    r"ggml\include\ggml.h",
    """        GGML_TYPE_PTQ1_0  = 143, // Prism packed group-128 ternary weights
        GGML_TYPE_COUNT   = 144,
    };""",
    """        GGML_TYPE_PTQ1_0  = 143, // Prism packed group-128 ternary weights
        GGML_TYPE_TQ2_T   = 144,
        GGML_TYPE_TQK6    = 145,
        GGML_TYPE_TQK7    = 146,
        GGML_TYPE_TQP5    = 147,
        GGML_TYPE_TQP6    = 148,
        GGML_TYPE_TQP7    = 149,
        GGML_TYPE_COUNT   = 150,
    };"""
)

patch_file(
    r"ggml\include\ggml.h",
    """        GGML_FTYPE_MOSTLY_PTQ1_0 = 129, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_CR   = 29, // except 1d tensors""",
    """        GGML_FTYPE_MOSTLY_PTQ1_0 = 129, // except 1d tensors
        GGML_FTYPE_MOSTLY_TQ2_T  = 130, // except 1d tensors (trellis-coded, group 128)
        GGML_FTYPE_MOSTLY_TQK6   = 131, // except 1d tensors (trellis-coded K=6, group 128)
        GGML_FTYPE_MOSTLY_TQK7   = 132, // except 1d tensors (trellis-coded K=7, group 128)
        GGML_FTYPE_MOSTLY_TQP5   = 133, // except 1d tensors (pair trellis K=5, group 128)
        GGML_FTYPE_MOSTLY_TQP6   = 134, // except 1d tensors (pair trellis K=6, group 128)
        GGML_FTYPE_MOSTLY_TQP7   = 135, // except 1d tensors (pair trellis K=7, group 128)
        GGML_FTYPE_MOSTLY_Q8_CR   = 29, // except 1d tensors"""
)

print("=== 2. Patching include/llama.h ===")
patch_file(
    r"include\llama.h",
    """        LLAMA_FTYPE_MOSTLY_PTQ1_0       = 143, // except 1d tensors (Prism group-128 ternary, 1.75 bpw)

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file""",
    """        LLAMA_FTYPE_MOSTLY_PTQ1_0       = 143, // except 1d tensors (Prism group-128 ternary, 1.75 bpw)
        LLAMA_FTYPE_MOSTLY_TQ2_T        = 144, // except 1d tensors (trellis-coded, group 128, 2.125 bpw)
        LLAMA_FTYPE_MOSTLY_TQK6         = 145, // except 1d tensors (trellis-coded K=6, group 128, 1.625 bpw)
        LLAMA_FTYPE_MOSTLY_TQK7         = 146, // except 1d tensors (trellis-coded K=7, group 128, 1.875 bpw)
        LLAMA_FTYPE_MOSTLY_TQP5         = 147, // except 1d tensors (pair trellis K=5, group 128, 2.625 bpw)
        LLAMA_FTYPE_MOSTLY_TQP6         = 148, // except 1d tensors (pair trellis K=6, group 128, 3.125 bpw)
        LLAMA_FTYPE_MOSTLY_TQP7         = 149, // except 1d tensors (pair trellis K=7, group 128, 3.625 bpw)

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file"""
)

print("=== 3. Patching ggml/src/ggml-common.h ===")
trellis_common_h = """
// TQ2_T: trellis-coded 2.125 bpw (QTIP-style bitshift trellis, L=16 K=8 V=4, group 128).
// The 32 qs bytes are the low bytes of a tail-biting 16-bit state path: step t
// (weights 4t..4t+3) has state s = qs[(t+31) % 32] << 8 | qs[t], and weight 4t+j is
// d * tq2t_lut_f16[2*tq2t_hyb_index(2*s + (j>>1)) + (j&1)] (one multiply per state,
// see tq2t_hyb_index). Every byte string is a
// valid path. Encoding (Viterbi) is done offline; see quantize_row_tq2_t_ref.
#define QK_TQ2_T 128
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[QK_TQ2_T/4]; // one trellis byte per 4 weights
} block_tq2_t;
static_assert(sizeof(block_tq2_t) == sizeof(ggml_half) + QK_TQ2_T/4, "wrong tq2_t block size/padding");

// TQK6 / TQK7: bit-packed siblings of TQ2_T (L=16, K=6 / 7 new bits per step, V=4,
// group 128). Same codebook and value formula as TQ2_T; only the state differs.
// qs is a circular 32*K-bit stream, stream bit i = (qs[i >> 3] >> (i & 7)) & 1.
// Step t (weights 4t..4t+3) has 16-bit state s = the 16 stream bits starting at bit
// (31 - t)*K, read LSB-first and circularly modulo 32*K (tail-biting: the first steps'
// windows wrap into bit 0..), and weight 4t+j is
// d * tq2t_lut_f16[2*tq2t_hyb_index(2*s + (j>>1)) + (j&1)]. See tqk_state.
#define QK_TQK 128
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[4*6];        // 32 steps x 6 bits
} block_tqk6;
static_assert(sizeof(block_tqk6) == sizeof(ggml_half) + 4*6, "wrong tqk6 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[4*7];        // 32 steps x 7 bits
} block_tqk7;
static_assert(sizeof(block_tqk7) == sizeof(ggml_half) + 4*7, "wrong tqk7 block size/padding");

// TQP5 / TQP6 / TQP7: pair trellis (L=16, K = 5 / 6 / 7 new bits per step, V=2, group 128).
// qs is a circular 64*K-bit stream, stream bit i = (qs[i >> 3] >> (i & 7)) & 1.
// Step t (weights 2t, 2t+1) has state s = the 16 stream bits starting at bit (63 - t)*K,
// read LSB-first and circularly modulo 64*K, and weight 2t+j is
// d * tq2t_lut_f16[2*tq2t_hyb_index(2*s) + j] (the first pair of TQ2_T state s). See tqp_state.
#define QK_TQP 128
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*5];        // 64 steps x 5 bits
} block_tqp5;
static_assert(sizeof(block_tqp5) == sizeof(ggml_half) + 8*5, "wrong tqp5 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*6];        // 64 steps x 6 bits
} block_tqp6;
static_assert(sizeof(block_tqp6) == sizeof(ggml_half) + 8*6, "wrong tqp6 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*7];        // 64 steps x 7 bits
} block_tqp7;
static_assert(sizeof(block_tqp7) == sizeof(ggml_half) + 8*7, "wrong tqp7 block size/padding");
"""

patch_file(
    r"ggml\src\ggml-common.h",
    """static_assert(sizeof(block_ptq1_0) == sizeof(ggml_half) + QK_PTQ1_0/64 + (QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5, "wrong ptq1_0 block size/padding");

#define QK4_0 32""",
    """static_assert(sizeof(block_ptq1_0) == sizeof(ggml_half) + QK_PTQ1_0/64 + (QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5, "wrong ptq1_0 block size/padding");
""" + trellis_common_h + """
#define QK4_0 32"""
)

print("=== 4. Patching ggml/src/ggml-cuda/moe-cache.cu ===")
patch_file(
    r"ggml\src\ggml-cuda\moe-cache.cu",
    """        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;""",
    """        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ2_T:
        case GGML_TYPE_TQK6:
        case GGML_TYPE_TQK7:
        case GGML_TYPE_TQP5:
        case GGML_TYPE_TQP6:
        case GGML_TYPE_TQP7:
            return true;"""
)
