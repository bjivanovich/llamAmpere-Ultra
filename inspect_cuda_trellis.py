import os
import re

agent_dir = r"C:\Users\bj_iv\Desktop\agentionai-llama"

def extract_trellis_sections(filepath):
    full = os.path.join(agent_dir, filepath)
    if not os.path.exists(full):
        print(f"NOT FOUND: {filepath}")
        return
    with open(full, 'r', encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
    
    print(f"\n=================== {filepath} ===================")
    matches = []
    for idx, l in enumerate(lines):
        if re.search(r'TQ2_T|TQK6|TQK7|TQP5|TQP6|TQP7|tq\.cuh|tq-lut-i8', l):
            matches.append(idx)
            
    # Group contiguous matches into blocks with 2 lines context
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

extract_trellis_sections(r"ggml\src\ggml-cuda\common.cuh")
extract_trellis_sections(r"ggml\src\ggml-cuda\convert.cu")
extract_trellis_sections(r"ggml\src\ggml-cuda\getrows.cu")
extract_trellis_sections(r"ggml\src\ggml-cuda\mmq.cu")
extract_trellis_sections(r"ggml\src\ggml-cuda\mmq.cuh")
extract_trellis_sections(r"ggml\src\ggml-cuda\mmvq.cu")
extract_trellis_sections(r"ggml\src\ggml-cuda\mmq-config-ampere.cuh")
extract_trellis_sections(r"ggml\src\ggml-cuda\mmq-load-tiles.cuh")
extract_trellis_sections(r"ggml\src\ggml-cuda\fattn-common.cuh")
