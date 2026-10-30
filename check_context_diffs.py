import os
import difflib

agent_dir = r"C:\Users\bj_iv\Desktop\agentionai-llama"
ampere_dir = r"C:\Users\bj_iv\Desktop\llamAmpere"

files_to_check = [
    r"src\llama-context.cpp",
    r"src\llama-context.h",
    r"src\llama-cparams.h",
    r"src\llama-graph.h",
    r"ggml\src\ggml-cuda\moe-cache.cu",
    r"ggml\src\ggml-cuda\fattn.cu",
    r"ggml\src\ggml-cuda\ggml-cuda.cu",
]

for rel in files_to_check:
    f_agent = os.path.join(agent_dir, rel)
    f_amp = os.path.join(ampere_dir, rel)
    
    if not os.path.exists(f_agent):
        print(f"Missing in agent: {rel}")
        continue
    if not os.path.exists(f_amp):
        print(f"Missing in ampere: {rel}")
        continue
        
    with open(f_agent, 'r', encoding='utf-8', errors='ignore') as fp1:
        lines_agent = fp1.readlines()
    with open(f_amp, 'r', encoding='utf-8', errors='ignore') as fp2:
        lines_amp = fp2.readlines()
        
    diff = list(difflib.unified_diff(lines_agent, lines_amp, fromfile='agent/'+rel, tofile='ampere/'+rel))
    added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
    print(f"{rel}: diff +{added} -{removed}")
