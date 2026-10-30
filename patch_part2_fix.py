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

patch(
    r"ggml\src\ggml.c",
    "        case GGML_FTYPE_MOSTLY_PTQ1_0:    wtype = GGML_TYPE_PTQ1_0; break;\n        case GGML_FTYPE_MOSTLY_Q5_0:",
    """        case GGML_FTYPE_MOSTLY_PTQ1_0:    wtype = GGML_TYPE_PTQ1_0; break;
        case GGML_FTYPE_MOSTLY_TQ2_T:     wtype = GGML_TYPE_TQ2_T;  break;
        case GGML_FTYPE_MOSTLY_TQK6:      wtype = GGML_TYPE_TQK6;   break;
        case GGML_FTYPE_MOSTLY_TQK7:      wtype = GGML_TYPE_TQK7;   break;
        case GGML_FTYPE_MOSTLY_TQP5:      wtype = GGML_TYPE_TQP5;   break;
        case GGML_FTYPE_MOSTLY_TQP6:      wtype = GGML_TYPE_TQP6;   break;
        case GGML_FTYPE_MOSTLY_TQP7:      wtype = GGML_TYPE_TQP7;   break;
        case GGML_FTYPE_MOSTLY_Q5_0:"""
)

patch(
    r"ggml\src\ggml.c",
    "        case GGML_TYPE_PTQ1_0: result = quantize_ptq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n        case GGML_TYPE_Q4_0:",
    """        case GGML_TYPE_PTQ1_0: result = quantize_ptq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQ2_T:  result = quantize_tq2_t (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQK6:   result = quantize_tqk6  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQK7:   result = quantize_tqk7  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP5:   result = quantize_tqp5  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP6:   result = quantize_tqp6  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQP7:   result = quantize_tqp7  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q4_0:"""
)
