import os

agent_common = r"C:\Users\bj_iv\Desktop\agentionai-llama\ggml\src\ggml-common.h"
ultra_common = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\ggml\src\ggml-common.h"

agent_quants = r"C:\Users\bj_iv\Desktop\agentionai-llama\ggml\src\ggml-quants.c"
ultra_quants = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\ggml\src\ggml-quants.c"

# 1. Extract lines 1986 to 2297 from agent_common
with open(agent_common, 'r', encoding='utf-8') as fp:
    agent_common_lines = fp.readlines()

lut_block = "".join(agent_common_lines[1985:2298])

# Insert into ultra_common before the final #endif // GGML_COMMON_IMPL
with open(ultra_common, 'r', encoding='utf-8') as fp:
    ultra_common_content = fp.read()

# Find the end of table section or end of file
target_end = "#endif // GGML_COMMON_IMPL"
idx = ultra_common_content.rfind(target_end)
if idx != -1:
    ultra_common_content = ultra_common_content[:idx] + "\n" + lut_block + "\n" + ultra_common_content[idx:]
    with open(ultra_common, 'w', encoding='utf-8') as fp:
        fp.write(ultra_common_content)
    print("SUCCESS: ggml-common.h updated with tq2t_lut_f16 and state functions")
else:
    print("FAILED: could not find end of ggml-common.h")

# 2. Extract tq2t_lut_val and tq2t_step_err from agent_quants (lines 2238 to 2255)
with open(agent_quants, 'r', encoding='utf-8') as fp:
    agent_quants_lines = fp.readlines()

err_block = "".join(agent_quants_lines[2237:2255])

with open(ultra_quants, 'r', encoding='utf-8') as fp:
    ultra_quants_content = fp.read()

target_anchor = "void quantize_row_tq2_t_ref"
if target_anchor in ultra_quants_content:
    ultra_quants_content = ultra_quants_content.replace(
        target_anchor,
        err_block + "\n" + target_anchor,
        1
    )
    with open(ultra_quants, 'w', encoding='utf-8') as fp:
        fp.write(ultra_quants_content)
    print("SUCCESS: ggml-quants.c updated with tq2t_lut_val and tq2t_step_err")
else:
    print("FAILED: could not find quantize_row_tq2_t_ref in ggml-quants.c")
