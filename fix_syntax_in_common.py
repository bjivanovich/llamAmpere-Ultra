import os

ultra_common = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\ggml\src\ggml-common.h"
agent_common = r"C:\Users\bj_iv\Desktop\agentionai-llama\ggml\src\ggml-common.h"

with open(agent_common, 'r', encoding='utf-8') as fp:
    agent_lines = fp.readlines()

# Extract from line 1985 (index 1984) to line 2298 (index 2297)
full_lut_block = "".join(agent_lines[1984:2298])

with open(ultra_common, 'r', encoding='utf-8') as fp:
    ultra_content = fp.read()

# Replace the broken block
broken_marker = "    defined(GGML_COMMON_IMPL_CUDA) || defined(GGML_COMMON_IMPL_HIP) || defined(GGML_COMMON_IMPL_MUSA)"
start_idx = ultra_content.find(broken_marker)
if start_idx != -1:
    end_marker = "#endif // GGML_COMMON_IMPL\n#endif // GGML_COMMON_IMPL"
    end_idx = ultra_content.find(end_marker, start_idx)
    if end_idx != -1:
        ultra_content = ultra_content[:start_idx] + full_lut_block + "\n" + ultra_content[end_idx:]
        with open(ultra_common, 'w', encoding='utf-8') as fp:
            fp.write(ultra_content)
        print("SUCCESS: Fixed #if header in ggml-common.h")
    else:
        print("FAILED: end_marker not found")
else:
    print("FAILED: broken_marker not found")
