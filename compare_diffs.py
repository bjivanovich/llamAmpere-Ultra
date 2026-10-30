import os
import difflib

agent_dir = r"C:\Users\bj_iv\Desktop\agentionai-llama"
ultra_dir = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"

key_files = [
    r"ggml\include\ggml.h",
    r"ggml\src\ggml.c",
    r"ggml\src\ggml-quants.c",
    r"ggml\src\ggml-cpu\ggml-cpu.c",
    r"ggml\src\ggml-cpu\ops.cpp",
    r"ggml\src\ggml-cuda\common.cuh",
    r"ggml\src\ggml-cuda\convert.cu",
    r"ggml\src\ggml-cuda\fattn-common.cuh",
    r"ggml\src\ggml-cuda\getrows.cu",
    r"ggml\src\ggml-cuda\ggml-cuda.cu",
    r"ggml\src\ggml-cuda\mmq.cu",
    r"ggml\src\ggml-cuda\mmq.cuh",
    r"ggml\src\ggml-cuda\mmvq.cu",
    r"ggml\src\ggml-cuda\mmq-load-tiles.cuh",
    r"ggml\src\ggml-cuda\mmq-vec-dot.cuh",
    r"ggml\src\ggml-cuda\mmq-config-ampere.cuh",
    r"ggml\src\ggml-cuda\CMakeLists.txt",
    r"src\llama-model-loader.cpp",
    r"src\llama-quant.cpp",
    r"src\models\qwen4exp.cpp",
    r"common\arg.cpp",
    r"common\speculative.cpp"
]

report = []
for rel in key_files:
    f_agent = os.path.join(agent_dir, rel)
    f_ultra = os.path.join(ultra_dir, rel)
    
    if not os.path.exists(f_ultra):
        report.append(f"[NEW IN AGENT] {rel}")
        continue
    if not os.path.exists(f_agent):
        report.append(f"[ONLY IN ULTRA] {rel}")
        continue
        
    with open(f_agent, 'r', encoding='utf-8', errors='ignore') as fp1:
        lines_agent = fp1.readlines()
    with open(f_ultra, 'r', encoding='utf-8', errors='ignore') as fp2:
        lines_ultra = fp2.readlines()
        
    diff = list(difflib.unified_diff(lines_ultra, lines_agent, fromfile='ultra/'+rel, tofile='agent/'+rel))
    added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
    report.append(f"[DIFF] {rel}: +{added} -{removed} lines")

with open(r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\diff_summary.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(report))

print("\n".join(report))
