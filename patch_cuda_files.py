import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"

def patch(rel_path, target, replacement):
    f = os.path.join(ultra_root, rel_path)
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    if target not in c:
        print(f"FAILED {rel_path}: target not found: {repr(target[:40])}")
        return False
    c = c.replace(target, replacement, 1)
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write(c)
    print(f"SUCCESS {rel_path}")
    return True

# 1. ggml-cuda/common.cuh
trellis_cuda_traits = """
// Trellis types (TQ2_T, TQK6, TQK7, TQP5/6/7): 128 weights per block, 32 ints of 4 weights
#define QR_TQ 4
#define QI_TQ (QK_TQK/QR_TQ)
static_assert(QK_TQ2_T == QK_TQK, "trellis block sizes differ");

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQ2_T> {
    static constexpr int qk = QK_TQ2_T;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tq2_t);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK6> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqk6);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK7> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqk7);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP5> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp5);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP6> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp6);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP7> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp7);
};
"""

patch(
    r"ggml\src\ggml-cuda\common.cuh",
    "template<>\nstruct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {",
    trellis_cuda_traits + "\ntemplate<>\nstruct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {"
)

# 2. ggml-cuda/convert.cu
patch(
    r"ggml\src\ggml-cuda\convert.cu",
    '#include "convert.cuh"\n#include "dequantize.cuh"',
    '#include "convert.cuh"\n#include "dequantize.cuh"\n#include "tq.cuh"'
)

tq_convert_cases = """        case GGML_TYPE_TQ2_T:
            return dequantize_tq_cuda<GGML_TYPE_TQ2_T>;
        case GGML_TYPE_TQK6:
            return dequantize_tq_cuda<GGML_TYPE_TQK6>;
        case GGML_TYPE_TQK7:
            return dequantize_tq_cuda<GGML_TYPE_TQK7>;
        case GGML_TYPE_TQP5:
            return dequantize_tq_cuda<GGML_TYPE_TQP5>;
        case GGML_TYPE_TQP6:
            return dequantize_tq_cuda<GGML_TYPE_TQP6>;
        case GGML_TYPE_TQP7:
            return dequantize_tq_cuda<GGML_TYPE_TQP7>;"""

# There are several get_to_fp32_cuda, get_to_fp16_cuda, etc. in convert.cu
with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\convert.cu"), 'r', encoding='utf-8') as fp:
    c_cu = fp.read()

# Add tq_convert_cases right before default: or case GGML_TYPE_F32: in get_dequantize_row_cuda
c_cu = c_cu.replace(
    "        case GGML_TYPE_IQ4_XS:\n            return dequantize_row_iq4_xs_cuda;",
    "        case GGML_TYPE_IQ4_XS:\n            return dequantize_row_iq4_xs_cuda;\n" + tq_convert_cases
)

with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\convert.cu"), 'w', encoding='utf-8') as fp:
    fp.write(c_cu)
print("SUCCESS ggml-cuda/convert.cu cases")

# 3. ggml-cuda/getrows.cu
patch(
    r"ggml\src\ggml-cuda\getrows.cu",
    '#include "dequantize.cuh"\n#include "convert.cuh"',
    '#include "dequantize.cuh"\n#include "convert.cuh"\n#include "tq.cuh"'
)

tq_getrows = """        case GGML_TYPE_TQ2_T:
            get_rows_cuda_tq<GGML_TYPE_TQ2_T, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQK6:
            get_rows_cuda_tq<GGML_TYPE_TQK6, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQK7:
            get_rows_cuda_tq<GGML_TYPE_TQK7, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP5:
            get_rows_cuda_tq<GGML_TYPE_TQP5, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP6:
            get_rows_cuda_tq<GGML_TYPE_TQP6, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP7:
            get_rows_cuda_tq<GGML_TYPE_TQP7, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;"""

patch(
    r"ggml\src\ggml-cuda\getrows.cu",
    "        case GGML_TYPE_IQ4_XS:\n            get_rows_cuda<GGML_TYPE_IQ4_XS, dst_t>(src0_d, src1_d, dst_d,",
    tq_getrows + "\n        case GGML_TYPE_IQ4_XS:\n            get_rows_cuda<GGML_TYPE_IQ4_XS, dst_t>(src0_d, src1_d, dst_d,"
)

# 4. ggml-cuda/mmq.cu
patch(
    r"ggml\src\ggml-cuda\mmq.cu",
    "        case GGML_TYPE_IQ4_XS:\n            return mmq_cuda<GGML_TYPE_IQ4_XS>(src0, src1, dst);",
    """        case GGML_TYPE_IQ4_XS:
            return mmq_cuda<GGML_TYPE_IQ4_XS>(src0, src1, dst);
        case GGML_TYPE_TQ2_T:
            return mmq_cuda<GGML_TYPE_TQ2_T>(src0, src1, dst);
        case GGML_TYPE_TQK6:
            return mmq_cuda<GGML_TYPE_TQK6>(src0, src1, dst);
        case GGML_TYPE_TQK7:
            return mmq_cuda<GGML_TYPE_TQK7>(src0, src1, dst);
        case GGML_TYPE_TQP5:
            return mmq_cuda<GGML_TYPE_TQP5>(src0, src1, dst);
        case GGML_TYPE_TQP6:
            return mmq_cuda<GGML_TYPE_TQP6>(src0, src1, dst);
        case GGML_TYPE_TQP7:
            return mmq_cuda<GGML_TYPE_TQP7>(src0, src1, dst);"""
)

# 5. ggml-cuda/mmq.cuh
patch(
    r"ggml\src\ggml-cuda\mmq.cuh",
    "        case GGML_TYPE_IQ4_XS:\n            return true;",
    """        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ2_T:
        case GGML_TYPE_TQK6:
        case GGML_TYPE_TQK7:
        case GGML_TYPE_TQP5:
        case GGML_TYPE_TQP6:
        case GGML_TYPE_TQP7:
            return true;"""
)

# 6. ggml-cuda/mmvq.cu
patch(
    r"ggml\src\ggml-cuda\mmvq.cu",
    '#include "mmvq.cuh"\n#include "vecdotq.cuh"',
    '#include "mmvq.cuh"\n#include "vecdotq.cuh"\n#include "tq.cuh"'
)

tq_mmvq = """        case GGML_TYPE_TQ2_T:    return vec_dot_tq_q8_1_s<GGML_TYPE_TQ2_T>;
        case GGML_TYPE_TQK6:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQK6>;
        case GGML_TYPE_TQK7:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQK7>;
        case GGML_TYPE_TQP5:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP5>;
        case GGML_TYPE_TQP6:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP6>;
        case GGML_TYPE_TQP7:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP7>;"""

patch(
    r"ggml\src\ggml-cuda\mmvq.cu",
    "        case GGML_TYPE_IQ4_XS:\n            return vec_dot_q_cuda<GGML_TYPE_IQ4_XS>;",
    "        case GGML_TYPE_IQ4_XS:\n            return vec_dot_q_cuda<GGML_TYPE_IQ4_XS>;\n" + tq_mmvq
)

# 7. ggml-cuda/mmq-config-ampere.cuh
tq_ampere_cfg = """    CASE(GGML_TYPE_TQ2_T, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_TQK6,  256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_TQK7,  256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_TQP5,  256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_TQP6,  256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_TQP7,  256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, true, true);
"""

patch(
    r"ggml\src\ggml-cuda\mmq-config-ampere.cuh",
    "    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q4_K, MMQ_ITER_K, true, true);",
    "    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q4_K, MMQ_ITER_K, true, true);\n" + tq_ampere_cfg
)
