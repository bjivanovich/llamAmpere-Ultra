import os
import re

agent_dir = r"C:\Users\bj_iv\Desktop\agentionai-llama"
patterns = [
    re.compile(r"GGML_TYPE_TQK7"),
    re.compile(r"draft-mtp"),
    re.compile(r"tq-lut-i8"),
    re.compile(r"tq\.cuh"),
    re.compile(r"GGML_TYPE_TQ2_T")
]

matching_files = []
for root, dirs, files in os.walk(agent_dir):
    if ".git" in root or "build" in root:
        continue
    for f in files:
        if f.endswith(('.h', '.c', '.cpp', '.cu', '.cuh')):
            full_path = os.path.join(root, f)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    for p in patterns:
                        if p.search(content):
                            rel = os.path.relpath(full_path, agent_dir)
                            matching_files.append((rel, p.pattern))
                            break
            except Exception as e:
                pass

print(f"Total matching files: {len(matching_files)}")
for mf, pat in sorted(matching_files):
    print(f"  {mf} (matched: {pat})")
