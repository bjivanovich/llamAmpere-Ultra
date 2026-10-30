import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"

def patch(rel_path, target, replacement):
    f = os.path.join(ultra_root, rel_path)
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    if target not in c:
        print(f"FAILED {rel_path}: target not found: {target[:40]}")
        return False
    c = c.replace(target, replacement, 1)
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write(c)
    print(f"SUCCESS {rel_path}")
    return True

# 1. include/llama.h
patch(
    r"include\llama.h",
    "        LLAMA_FTYPE_MOSTLY_PTQ1_0    = 143, // except 1d tensors (Prism group-128 ternary, 1.75 bpw)\n        LLAMA_FTYPE_MOSTLY_Q8_CR         = 42, // except 1d tensors",
    """        LLAMA_FTYPE_MOSTLY_PTQ1_0    = 143, // except 1d tensors (Prism group-128 ternary, 1.75 bpw)
        LLAMA_FTYPE_MOSTLY_TQ2_T        = 144, // except 1d tensors (trellis-coded, group 128, 2.125 bpw)
        LLAMA_FTYPE_MOSTLY_TQK6         = 145, // except 1d tensors (trellis-coded K=6, group 128, 1.625 bpw)
        LLAMA_FTYPE_MOSTLY_TQK7         = 146, // except 1d tensors (trellis-coded K=7, group 128, 1.875 bpw)
        LLAMA_FTYPE_MOSTLY_TQP5         = 147, // except 1d tensors (pair trellis K=5, group 128, 2.625 bpw)
        LLAMA_FTYPE_MOSTLY_TQP6         = 148, // except 1d tensors (pair trellis K=6, group 128, 3.125 bpw)
        LLAMA_FTYPE_MOSTLY_TQP7         = 149, // except 1d tensors (pair trellis K=7, group 128, 3.625 bpw)
        LLAMA_FTYPE_MOSTLY_Q8_CR         = 42, // except 1d tensors"""
)

# 2. ggml/src/ggml-quants.h
patch(
    r"ggml\src\ggml-quants.h",
    "GGML_API void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k);",
    """GGML_API void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_t_ref(const float * GGML_RESTRICT x, block_tq2_t * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqk6_ref(const float * GGML_RESTRICT x, block_tqk6 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqk7_ref(const float * GGML_RESTRICT x, block_tqk7 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp5_ref(const float * GGML_RESTRICT x, block_tqp5 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp6_ref(const float * GGML_RESTRICT x, block_tqp6 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp7_ref(const float * GGML_RESTRICT x, block_tqp7 * GGML_RESTRICT y, int64_t k);"""
)

patch(
    r"ggml\src\ggml-quants.h",
    "GGML_API void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);",
    """GGML_API void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_t(const block_tq2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqk6(const block_tqk6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqk7(const block_tqk7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp5(const block_tqp5 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp6(const block_tqp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp7(const block_tqp7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
)

patch(
    r"ggml\src\ggml-quants.h",
    "GGML_API size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);",
    """GGML_API size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_t(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqk6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqk7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp5(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
)

# 3. ggml/src/ggml.c
patch(
    r"ggml\src\ggml.c",
    """    [GGML_TYPE_PTQ1_0] = {
        .type_name                = "ptq1_0",
        .blck_size                = QK_PTQ1_0,
        .type_size                = sizeof(block_ptq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_ptq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_ptq1_0_ref,
    },""",
    """    [GGML_TYPE_PTQ1_0] = {
        .type_name                = "ptq1_0",
        .blck_size                = QK_PTQ1_0,
        .type_size                = sizeof(block_ptq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_ptq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_ptq1_0_ref,
    },
    [GGML_TYPE_TQ2_T] = {
        .type_name                = "tq2_t",
        .blck_size                = QK_TQ2_T,
        .type_size                = sizeof(block_tq2_t),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_t,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_t_ref,
    },
    [GGML_TYPE_TQK6] = {
        .type_name                = "tqk6",
        .blck_size                = QK_TQK,
        .type_size                = sizeof(block_tqk6),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tqk6,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tqk6_ref,
    },
    [GGML_TYPE_TQK7] = {
        .type_name                = "tqk7",
        .blck_size                = QK_TQK,
        .type_size                = sizeof(block_tqk7),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tqk7,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tqk7_ref,
    },
    [GGML_TYPE_TQP5] = {
        .type_name                = "tqp5",
        .blck_size                = QK_TQP,
        .type_size                = sizeof(block_tqp5),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tqp5,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tqp5_ref,
    },
    [GGML_TYPE_TQP6] = {
        .type_name                = "tqp6",
        .blck_size                = QK_TQP,
        .type_size                = sizeof(block_tqp6),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tqp6,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tqp6_ref,
    },
    [GGML_TYPE_TQP7] = {
        .type_name                = "tqp7",
        .blck_size                = QK_TQP,
        .type_size                = sizeof(block_tqp7),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tqp7,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tqp7_ref,
    },"""
)

patch(
    r"ggml\src\ggml.c",
    """        case GGML_FTYPE_MOSTLY_PTQ1_0: wtype = GGML_TYPE_PTQ1_0; break;
        case GGML_FTYPE_MOSTLY_Q8_CR:""",
    """        case GGML_FTYPE_MOSTLY_PTQ1_0: wtype = GGML_TYPE_PTQ1_0; break;
        case GGML_FTYPE_MOSTLY_TQ2_T:  wtype = GGML_TYPE_TQ2_T;  break;
        case GGML_FTYPE_MOSTLY_TQK6:   wtype = GGML_TYPE_TQK6;   break;
        case GGML_FTYPE_MOSTLY_TQK7:   wtype = GGML_TYPE_TQK7;   break;
        case GGML_FTYPE_MOSTLY_TQP5:   wtype = GGML_TYPE_TQP5;   break;
        case GGML_FTYPE_MOSTLY_TQP6:   wtype = GGML_TYPE_TQP6;   break;
        case GGML_FTYPE_MOSTLY_TQP7:   wtype = GGML_TYPE_TQP7;   break;
        case GGML_FTYPE_MOSTLY_Q8_CR:"""
)

patch(
    r"ggml\src\ggml.c",
    """        case GGML_TYPE_PTQ1_0:  result = quantize_ptq1_0 (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ2_XXS:""",
    """        case GGML_TYPE_PTQ1_0:  result = quantize_ptq1_0 (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQ2_T:   result = quantize_tq2_t  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQK6:   result = quantize_tqk6  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQK7:   result = quantize_tqk7  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP5:   result = quantize_tqp5  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP6:   result = quantize_tqp6  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP7:   result = quantize_tqp7  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ2_XXS:"""
)
