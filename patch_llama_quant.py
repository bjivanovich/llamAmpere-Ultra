import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"
f_quant = os.path.join(ultra_root, r"src\llama-quant.cpp")

with open(f_quant, 'r', encoding='utf-8') as fp:
    c = fp.read()

c = c.replace(
    "            case GGML_TYPE_PTQ1_0:\n            case GGML_TYPE_PQ2_0:",
    """            case GGML_TYPE_PTQ1_0:
            case GGML_TYPE_PQ2_0:
            case GGML_TYPE_TQ2_T:
            case GGML_TYPE_TQK6:
            case GGML_TYPE_TQK7:
            case GGML_TYPE_TQP5:
            case GGML_TYPE_TQP6:
            case GGML_TYPE_TQP7:""",
    1
)

c = c.replace(
    "        case LLAMA_FTYPE_MOSTLY_PTQ1_0: return GGML_TYPE_PTQ1_0;\n\n        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE:",
    """        case LLAMA_FTYPE_MOSTLY_PTQ1_0: return GGML_TYPE_PTQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_T:  return GGML_TYPE_TQ2_T;
        case LLAMA_FTYPE_MOSTLY_TQK6:   return GGML_TYPE_TQK6;
        case LLAMA_FTYPE_MOSTLY_TQK7:   return GGML_TYPE_TQK7;
        case LLAMA_FTYPE_MOSTLY_TQP5:   return GGML_TYPE_TQP5;
        case LLAMA_FTYPE_MOSTLY_TQP6:   return GGML_TYPE_TQP6;
        case LLAMA_FTYPE_MOSTLY_TQP7:   return GGML_TYPE_TQP7;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE:""",
    1
)

with open(f_quant, 'w', encoding='utf-8') as fp:
    fp.write(c)

print("SUCCESS: llama-quant.cpp patched")
