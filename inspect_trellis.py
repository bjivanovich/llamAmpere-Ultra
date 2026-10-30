import os

agent = r"C:\Users\bj_iv\Desktop\agentionai-llama"

def search_in_file(rel_path, pattern):
    p = os.path.join(agent, rel_path)
    if not os.path.exists(p):
        print(f"File not found: {rel_path}")
        return
    with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
    print(f"=== {rel_path} ===")
    for idx, line in enumerate(lines):
        if pattern in line:
            start = max(0, idx - 2)
            end = min(len(lines), idx + 3)
            for j in range(start, end):
                print(f"{j+1}: {lines[j]}", end="")
            print("---")

search_in_file(r"ggml\src\ggml-cuda\CMakeLists.txt", "tq")
search_in_file(r"ggml\src\ggml.c", "GGML_TYPE_TQK7")
search_in_file(r"src\llama-model-loader.cpp", "GGML_TYPE_TQK7")
