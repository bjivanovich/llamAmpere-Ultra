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
env["GGML_CUDA_EXL3_GEMM"] = "1"
env["GGML_CUDA_EXL3_DEC"] = "1"
env["GGML_CUDA_EXL3_VW"] = "1"

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
    "--fit", "off",
    "--cache-prompt",
    "--cache-ram", "16384",
    "--ctx-checkpoints", "16",
    "--checkpoint-min-step", "10240",
    "--spec-draft-vocab-map", VOCAB_MAP,
    "--spec-chain", "8",
    "--spec-draft-n-min-adaptive", "3",
    "--spec-draft-vocab-hot", "256",
    "--spec-draft-backend-sampling",
    "--spec-draft-type-k", "q8_0",
    "--spec-draft-type-v", "q8_0",
    "--gdn-replay",
    "--reasoning-format", "none",
    "--temp", "0.2",
    "--top-p", "0.95",
    "--top-k", "40",
    "--min-p", "0.05",
    "--repeat-penalty", "1.0",
    "--repeat-last-n", "256",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

print("Launching tuned server...")
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def logger():
    for line in iter(proc.stdout.readline, ''):
        l = line.strip()
        if any(k in l for k in ["listening", "loaded", "spec", "chain", "turbo", "error", "failed", "eval time"]):
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
            print(f"\nServer ready in {i+1}s!")
            break
    except Exception:
        pass
    time.sleep(1)

if not ready:
    print("FAILED TO START SERVER")
    proc.kill()
    sys.exit(1)

# Warmup request to lock in CUDA graphs
print("Running graph warmup...")
requests.post(url_chat, json={"messages": [{"role": "user", "content": "Warmup"}], "max_tokens": 16}, timeout=30)

# Main generation benchmark
print("Running generation benchmark (512 tokens)...")
payload = {
    "messages": [{"role": "user", "content": "Write a high-performance concurrent hashmap in C++20 with hopscotch hashing and SIMD lookup."}],
    "max_tokens": 512,
    "temperature": 0.2
}

t0 = time.time()
r = requests.post(url_chat, json=payload, timeout=60).json()
t1 = time.time()

usage = r.get("usage", {})
comp_tokens = usage.get("completion_tokens", 0)
gen_sec = t1 - t0
tok_s = comp_tokens / gen_sec if gen_sec > 0 else 0

print(f"\n==========================================")
print(f"BENCHMARK RESULTS (96k Context + MTP Chain 8):")
print(f"Completion Tokens: {comp_tokens}")
print(f"Total Time:        {gen_sec:.2f}s")
print(f"SPEED:             {tok_s:.2f} tok/s")
print(f"==========================================")

proc.kill()
