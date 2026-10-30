import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"
f_loader = os.path.join(ultra_root, r"src\llama-model-loader.cpp")

with open(f_loader, 'r', encoding='utf-8') as fp:
    c = fp.read()

# 1. llama_ftype_name
tq_ftype_names = """        case LLAMA_FTYPE_MOSTLY_PTQ1_0:    name = LLAMA_FTYPE_PREFIX "PTQ1_0 - 1.75 bpw ternary (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_T:     name = LLAMA_FTYPE_PREFIX "TQ2_T - 2.125 bpw trellis (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQK6:      name = LLAMA_FTYPE_PREFIX "TQK6 - 1.625 bpw trellis (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQK7:      name = LLAMA_FTYPE_PREFIX "TQK7 - 1.875 bpw trellis (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQP5:      name = LLAMA_FTYPE_PREFIX "TQP5 - 2.625 bpw pair trellis (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQP6:      name = LLAMA_FTYPE_PREFIX "TQP6 - 3.125 bpw pair trellis (group 128)"; break;
        case LLAMA_FTYPE_MOSTLY_TQP7:      name = LLAMA_FTYPE_PREFIX "TQP7 - 3.625 bpw pair trellis (group 128)"; break;"""

c = c.replace(
    '        case LLAMA_FTYPE_MOSTLY_PTQ1_0:    name = LLAMA_FTYPE_PREFIX "PTQ1_0 - 1.75 bpw ternary (group 128)"; break;',
    tq_ftype_names,
    1
)

# 2. Trellis state bits check
tq_state_check = """
        // Trellis types decode with TQ_STATE_BITS-bit states; a file encoded with any other
        // width would load and silently produce garbage, so require the declared width.
        if (n_type[GGML_TYPE_TQ2_T] + n_type[GGML_TYPE_TQK6] + n_type[GGML_TYPE_TQK7] +
            n_type[GGML_TYPE_TQP5] + n_type[GGML_TYPE_TQP6] + n_type[GGML_TYPE_TQP7] > 0) {
            const int64_t kid = gguf_find_key(metadata, "agention.trellis.state_bits");
            const uint32_t bits = kid >= 0 ? gguf_get_val_u32(metadata, kid) : 16u;
            if (bits != 15u) {
                throw std::runtime_error(format(
                    "trellis tensors were encoded with %u-bit states (%s); this build decodes 15-bit states. "
                    "Re-encode the file with a current gguf-hadamard.",
                    bits, kid >= 0 ? "agention.trellis.state_bits" : "no agention.trellis.state_bits key: pre-2026-09-26 format"));
            }
        }
"""

c = c.replace(
    "        switch (type_max) {",
    tq_state_check + "\n        switch (type_max) {",
    1
)

# 3. switch (type_max) cases
tq_type_cases = """            case GGML_TYPE_PTQ1_0:  ftype = LLAMA_FTYPE_MOSTLY_PTQ1_0;  break;
            case GGML_TYPE_TQ2_T:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_T;   break;
            case GGML_TYPE_TQK6:    ftype = LLAMA_FTYPE_MOSTLY_TQK6;    break;
            case GGML_TYPE_TQK7:    ftype = LLAMA_FTYPE_MOSTLY_TQK7;    break;
            case GGML_TYPE_TQP5:    ftype = LLAMA_FTYPE_MOSTLY_TQP5;    break;
            case GGML_TYPE_TQP6:    ftype = LLAMA_FTYPE_MOSTLY_TQP6;    break;
            case GGML_TYPE_TQP7:    ftype = LLAMA_FTYPE_MOSTLY_TQP7;    break;"""

c = c.replace(
    "            case GGML_TYPE_PTQ1_0:  ftype = LLAMA_FTYPE_MOSTLY_PTQ1_0;  break;",
    tq_type_cases,
    1
)

with open(f_loader, 'w', encoding='utf-8') as fp:
    fp.write(c)

print("SUCCESS: llama-model-loader.cpp patched")
