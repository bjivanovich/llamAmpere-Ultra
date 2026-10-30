import os
ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"

def find_context(rel_path, pattern):
    f = os.path.join(ultra_root, rel_path)
    if not os.path.exists(f):
        print(f"NOT FOUND: {rel_path}")
        return
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    print(f"=== {rel_path} (search: {pattern}) ===")
    for idx, l in enumerate(lines):
        if pattern in l:
            start = max(0, idx - 2)
            end = min(len(lines), idx + 4)
            for j in range(start, end):
                print(f"{j+1}: {repr(lines[j])}")

find_context(r"ggml\src\ggml-cuda\getrows.cu", "case GGML_TYPE_")
find_context(r"ggml\src\ggml-cuda\mmq.cu", "case GGML_TYPE_")
find_context(r"ggml\src\ggml-cuda\mmq.cuh", "case GGML_TYPE_")
find_context(r"ggml\src\ggml-cuda\mmvq.cu", "return vec_dot")
find_context(r"ggml\src\ggml-cuda\mmq-config-ampere.cuh", "CASE(GGML_TYPE_")
