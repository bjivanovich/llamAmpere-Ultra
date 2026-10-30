import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"
agent_root = r"C:\Users\bj_iv\Desktop\agentionai-llama"

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

# 1. mmq.cu
tq_mmq_cases = """        case GGML_TYPE_TQ2_T:
            mul_mat_q_case<GGML_TYPE_TQ2_T>(ctx, args, stream);
            break;
        case GGML_TYPE_TQK6:
            mul_mat_q_case<GGML_TYPE_TQK6>(ctx, args, stream);
            break;
        case GGML_TYPE_TQK7:
            mul_mat_q_case<GGML_TYPE_TQK7>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP5:
            mul_mat_q_case<GGML_TYPE_TQP5>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP6:
            mul_mat_q_case<GGML_TYPE_TQP6>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP7:
            mul_mat_q_case<GGML_TYPE_TQP7>(ctx, args, stream);
            break;"""

patch(
    r"ggml\src\ggml-cuda\mmq.cu",
    "        case GGML_TYPE_IQ4_NL:\n            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);\n            break;",
    "        case GGML_TYPE_IQ4_NL:\n            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);\n            break;\n" + tq_mmq_cases
)

# 2. mmq.cuh
tq_mmq_supp = """        case GGML_TYPE_TQ2_T:
        case GGML_TYPE_TQK6:
        case GGML_TYPE_TQK7:
        case GGML_TYPE_TQP5:
        case GGML_TYPE_TQP6:
        case GGML_TYPE_TQP7:"""

patch(
    r"ggml\src\ggml-cuda\mmq.cuh",
    "        case GGML_TYPE_MXFP4:\n        case GGML_TYPE_NVFP4:\n            mmq_supported = true;",
    "        case GGML_TYPE_MXFP4:\n        case GGML_TYPE_NVFP4:\n" + tq_mmq_supp + "\n            mmq_supported = true;"
)

# 3. mmvq.cu
# Check include in mmvq.cu
with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\mmvq.cu"), 'r', encoding='utf-8') as fp:
    c_mmvq = fp.read()

if '#include "tq.cuh"' not in c_mmvq:
    c_mmvq = '#include "tq.cuh"\n' + c_mmvq

tq_mmvq_switch = """        case GGML_TYPE_TQ2_T:    return vec_dot_tq_q8_1_s<GGML_TYPE_TQ2_T>;
        case GGML_TYPE_TQK6:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQK6>;
        case GGML_TYPE_TQK7:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQK7>;
        case GGML_TYPE_TQP5:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP5>;
        case GGML_TYPE_TQP6:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP6>;
        case GGML_TYPE_TQP7:     return vec_dot_tq_q8_1_s<GGML_TYPE_TQP7>;"""

# In mmvq.cu, find default: in get_vec_dot_q_cuda
c_mmvq = c_mmvq.replace(
    "        default:\n            return nullptr;",
    tq_mmvq_switch + "\n        default:\n            return nullptr;"
)
with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\mmvq.cu"), 'w', encoding='utf-8') as fp:
    fp.write(c_mmvq)
print("SUCCESS mmvq.cu")

# 4. mmq-config-ampere.cuh
# Extract all CASE(GGML_TYPE_TQ... from agent_root mmq-config-ampere.cuh
with open(os.path.join(agent_root, r"ggml\src\ggml-cuda\mmq-config-ampere.cuh"), 'r', encoding='utf-8') as fp:
    agent_cfg = fp.readlines()

tq_cfg_lines = [l for l in agent_cfg if any(k in l for k in ["GGML_TYPE_TQ2_T", "GGML_TYPE_TQK6", "GGML_TYPE_TQK7", "GGML_TYPE_TQP5", "GGML_TYPE_TQP6", "GGML_TYPE_TQP7"])]
tq_cfg_block = "\n    // Trellis types on Ampere SM86\n" + "".join(tq_cfg_lines) + "\n"

with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\mmq-config-ampere.cuh"), 'r', encoding='utf-8') as fp:
    c_amp = fp.read()

# Append right before default: or return false
c_amp = c_amp.replace(
    "    return false;\n}",
    tq_cfg_block + "    return false;\n}"
)
with open(os.path.join(ultra_root, r"ggml\src\ggml-cuda\mmq-config-ampere.cuh"), 'w', encoding='utf-8') as fp:
    fp.write(c_amp)
print("SUCCESS mmq-config-ampere.cuh")
