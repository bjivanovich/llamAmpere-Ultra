import os

agent_quants = r"C:\Users\bj_iv\Desktop\agentionai-llama\ggml\src\ggml-quants.c"
ultra_quants = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\ggml\src\ggml-quants.c"

with open(agent_quants, 'r', encoding='utf-8') as fp:
    agent_lines = fp.readlines()

with open(ultra_quants, 'r', encoding='utf-8') as fp:
    ultra_lines = fp.readlines()

# In agent_quants, find start and end of Trellis block
start_idx = None
end_idx = None
for i, l in enumerate(agent_lines):
    if "void quantize_row_tq2_t_ref" in l:
        start_idx = i - 1
        break

for i in range(start_idx, len(agent_lines)):
    if "void quantize_row_q4_0_ref" in agent_lines[i] or "size_t quantize_q4_0" in agent_lines[i] or "void quantize_row_q4_0" in agent_lines[i]:
        end_idx = i
        break

print(f"Agent Trellis block lines: {start_idx+1} to {end_idx} (total {end_idx - start_idx} lines)")
trellis_code = "".join(agent_lines[start_idx:end_idx])

# In ultra_quants, find where to insert: right before quantize_row_q4_0
insert_idx = None
for i, l in enumerate(ultra_lines):
    if "void quantize_row_q4_0_ref" in l or "void quantize_row_q4_0(" in l:
        insert_idx = i
        break

print(f"Ultra insertion line: {insert_idx+1}")
ultra_lines.insert(insert_idx, "\n// ==================== Trellis (TQ2_T, TQK6, TQK7, TQP5/6/7) ====================\n" + trellis_code + "\n")

# Also find switch (type) in validate_row_data in ultra_quants
val_idx = None
for i, l in enumerate(ultra_lines):
    if "case GGML_TYPE_IQ1_S:" in l:
        val_idx = i
        break

if val_idx:
    print(f"Adding validation cases before line {val_idx+1}")
    cases = """        case GGML_TYPE_TQ2_T:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_t, data, nb);
            } break;
        case GGML_TYPE_TQK6:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqk6, data, nb);
            } break;
        case GGML_TYPE_TQK7:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqk7, data, nb);
            } break;
        case GGML_TYPE_TQP5:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp5, data, nb);
            } break;
        case GGML_TYPE_TQP6:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp6, data, nb);
            } break;
        case GGML_TYPE_TQP7:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp7, data, nb);
            } break;
"""
    ultra_lines.insert(val_idx, cases)

with open(ultra_quants, 'w', encoding='utf-8') as fp:
    fp.writelines(ultra_lines)

print("Successfully patched ggml-quants.c")
