import os

agent_root = r"C:\Users\bj_iv\Desktop\agentionai-llama"
ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"

missing_files = []
for root, dirs, files in os.walk(agent_root):
    if ".git" in root or "build" in root:
        continue
    rel = os.path.relpath(root, agent_root)
    ultra_dir = os.path.join(ultra_root, rel)
    for f in files:
        if f.endswith(('.h', '.hpp', '.c', '.cpp', '.cu', '.cuh')):
            ultra_file = os.path.join(ultra_dir, f)
            if not os.path.exists(ultra_file):
                missing_files.append(os.path.join(rel, f))

print(f"Total missing source files: {len(missing_files)}")
for mf in missing_files:
    print(f"  {mf}")
