import os
import sys
import time
import subprocess
import requests

SERVER_EXE = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
MODEL_PATH = r"D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-IQ4_XS-M.gguf"
VOCAB_MAP = r"D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"
PORT = 8088

try:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
except Exception:
    pass
time.sleep(1)

env = os.environ.copy()
env["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env["GGML_CUDA_FA_I8QK"] = "1"
env["GGML_CUDA_GRAPH_OPT"] = "1"
env["LLAMA_SPEC_CHAIN"] = "1"
env["LLAMA_SPEC_CHAIN_SAMPLED"] = "1"
env["LLAMA_SPEC_PQ"] = "1"
env["LLAMA_SPEC_ADAPT_COST"] = "1"

cmd = [
    SERVER_EXE,
    "-m", MODEL_PATH,
    "-ngl", "99",
    "-c", "98304",
    "-b", "4096",
    "-ub", "1024",
    "-t", "12",
    "-tb", "8",
    "-fa", "on",
    "-ctk", "turbo5",
    "-ctv", "turbo4",
    "--kv-unified",
    "--prio", "3",
    "--parallel", "1",
    "--jinja",
    "--fit", "off",
    "--cache-prompt",
    "--cache-ram", "16384",
    "--ctx-checkpoints", "16",
    "--checkpoint-min-step", "10240",
    "--spec-draft-vocab-map", VOCAB_MAP,
    "--spec-draft-vocab-hot", "256",
    "--spec-draft-type-k", "q8_0",
    "--spec-draft-type-v", "q8_0",
    "--spec-draft-backend-sampling",
    "--reasoning-format", "none",
    "--reasoning-effort", "medium",
    "--temp", "0.2",
    "--top-p", "0.95",
    "--top-k", "40",
    "--min-p", "0.05",
    "--repeat-penalty", "1.0",
    "--repeat-last-n", "256",
    "--alias", "Swift-1.5-Cyber-ATX-IQ4_XS",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

print("Launching 96k tuned server...")
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def logger():
    for line in iter(proc.stdout.readline, ''):
        l = line.strip()
        if any(k in l for k in ["listening", "loaded", "eval time", "graphs reused", "draft acceptance"]):
            print(f"  [srv] {l}", flush=True)

import threading
t = threading.Thread(target=logger, daemon=True)
t.start()

url_health = f"http://127.0.0.1:{PORT}/health"
url_chat = f"http://127.0.0.1:{PORT}/v1/chat/completions"

ready = False
for i in range(60):
    try:
        r = requests.get(url_health, timeout=1)
        if r.status_code == 200:
            ready = True
            print(f"Server ready in {i+1}s!")
            break
    except Exception:
        pass
    time.sleep(1)

if not ready:
    print("FAILED TO START SERVER")
    proc.kill()
    sys.exit(1)

# Warmup request
requests.post(url_chat, json={"messages": [{"role": "user", "content": "Warmup request"}], "max_tokens": 32}, timeout=30)

print("\n--- BENCHMARK 1 (256 tokens) ---")
payload1 = {
    "messages": [{"role": "user", "content": "Write a high-performance memory allocator in C++ with slab and arena allocation strategies."}],
    "max_tokens": 256,
    "temperature": 0.2
}
t0 = time.time()
r1 = requests.post(url_chat, json=payload1, timeout=60).json()
t1 = time.time()

u1 = r1.get("usage", {})
c1 = u1.get("completion_tokens", 0)
s1 = c1 / (t1 - t0) if (t1 - t0) > 0 else 0
print(f"Tokens: {c1} in {t1-t0:.2f}s -> {s1:.2f} tok/s")

print("\n--- BENCHMARK 2 (512 tokens with warmed CUDA graphs) ---")
payload2 = {
    "messages": [{"role": "user", "content": "Implement a high-throughput lock-free FIFO queue with hazard pointers in modern C++."}],
    "max_tokens": 512,
    "temperature": 0.2
}
t0 = time.time()
r2 = requests.post(url_chat, json=payload2, timeout=60).json()
t1 = time.time()

u2 = r2.get("usage", {})
c2 = u2.get("completion_tokens", 0)
s2 = c2 / (t1 - t0) if (t1 - t0) > 0 else 0
print(f"Tokens: {c2} in {t1-t0:.2f}s -> {s2:.2f} tok/s")

proc.kill()
