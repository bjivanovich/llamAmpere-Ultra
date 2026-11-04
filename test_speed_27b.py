import os
import sys
import time
import subprocess
import requests
import json

SERVER_EXE = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
MODEL_PATH = r"D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-IQ4_XS-M.gguf"
VOCAB_MAP = r"D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"
PORT = 8089

try:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
except Exception:
    pass
time.sleep(1)

env = os.environ.copy()
env["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env["GGML_CUDA_GRAPH_OPT"] = "1"
env["GGML_CUDA_FA_I8QK"] = "1"
env["LLAMA_SPEC_CHAIN"] = "1"
env["LLAMA_SPEC_CHAIN_SAMPLED"] = "1"
env["LLAMA_SPEC_PQ"] = "1"
env["LLAMA_SPEC_ADAPT_COST"] = "1"

cmd = [
    SERVER_EXE,
    "-m", MODEL_PATH,
    "-ngl", "99",
    "-c", "8192",
    "-b", "4096",
    "-ub", "1024",
    "-t", "20",
    "-tb", "8",
    "-fa", "on",
    "-ctk", "turbo5",
    "-ctv", "turbo4",
    "--kv-unified",
    "--prio", "3",
    "--parallel", "1",
    "--spec-draft-type-k", "q8_0",
    "--spec-draft-type-v", "q8_0",
    "--spec-draft-vocab-map", VOCAB_MAP,
    "--reasoning-format", "none",
    "--temp", "0.2",
    "--top-p", "0.90",
    "--top-k", "40",
    "--min-p", "0.05",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

print("Starting server...")
t_start = time.time()
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

url_health = f"http://127.0.0.1:{PORT}/health"
url_chat = f"http://127.0.0.1:{PORT}/v1/chat/completions"

ready = False
for i in range(60):
    try:
        r = requests.get(url_health, timeout=1)
        if r.status_code == 200:
            ready = True
            print(f"Server ready in {time.time() - t_start:.2f}s")
            break
    except Exception:
        pass
    time.sleep(1)

if not ready:
    print("FAILED to start server")
    proc.kill()
    sys.exit(1)

prompt = "Implement a high-performance thread-safe concurrent queue in C++20 with ring buffer and lock-free atomic head/tail."
payload = {
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 512,
    "temperature": 0.2
}

# Warmup request
print("Warming up graphs...")
requests.post(url_chat, json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}, timeout=30)

print("Running benchmark generation...")
t0 = time.time()
res = requests.post(url_chat, json=payload, timeout=60).json()
t1 = time.time()

usage = res.get("usage", {})
comp_tokens = usage.get("completion_tokens", 0)
duration = t1 - t0
tok_s = comp_tokens / duration if duration > 0 else 0

print(f"\n==========================================")
print(f"Results for Cyber-ATX iQ4_XS 27B:")
print(f"Generated tokens: {comp_tokens}")
print(f"Generation time:  {duration:.2f}s")
print(f"SPEED:            {tok_s:.2f} tok/s")
print(f"==========================================")

proc.kill()
