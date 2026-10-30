import os
import re

agent_dir = r"C:\Users\bj_iv\Desktop\agentionai-llama"

def dump_file_matches(filepath):
    full = os.path.join(agent_dir, filepath)
    with open(full, 'r', encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
    print(f"\n=================== {filepath} ===================")
    matches = []
    for idx, l in enumerate(lines):
        if re.search(r'TQ2_T|TQK6|TQK7|TQP5|TQP6|TQP7|quantize_row_tq|quantize_tq', l):
            matches.append(idx)
    blocks = []
    curr_block = []
    for m in matches:
        if not curr_block:
            curr_block = [m]
        elif m - curr_block[-1] <= 3:
            curr_block.append(m)
        else:
            blocks.append((curr_block[0], curr_block[-1]))
            curr_block = [m]
    if curr_block:
        blocks.append((curr_block[0], curr_block[-1]))
    for start, end in blocks:
        s = max(0, start - 2)
        e = min(len(lines), end + 3)
        print(f"--- Lines {s+1} to {e} ---")
        for j in range(s, e):
            print(f"{j+1}: {lines[j]}", end="")

dump_file_matches(r"ggml\src\ggml.c")
dump_file_matches(r"ggml\src\ggml-quants.h")
