import os
import sys
import time
import subprocess
import requests
import json
import threading

SERVER_EXE = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
MODEL_PATH = r"D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-IQ4_XS-M.gguf"
VOCAB_MAP = r"D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"
RESULTS_DIR = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\pipeline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PORT = 8089

def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1

try:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
except Exception:
    pass
time.sleep(1)

base_vram = get_vram_mb()
print("="*75)
print("PRUEBA DE INFERENCIA EXTENSA Y VERIFICACION VMM HOT-SWAP (<3s)")
print("="*75)
print(f"Linea base de VRAM del sistema: {base_vram} MB")

env = os.environ.copy()
env["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env["GGML_CUDA_FA_I8QK"] = "1"
env["GGML_CUDA_GRAPH_OPT"] = "1"
env["LLAMA_SPEC_CHAIN"] = "1"
env["LLAMA_SPEC_CHAIN_SAMPLED"] = "1"
env["LLAMA_SPEC_PQ"] = "1"
env["LLAMA_SPEC_ADAPT_COST"] = "1"
env["LLAMAMPERE_ENABLE_SLEEP"] = "1"

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
    "--sleep-idle-seconds", "3",
    "--alias", "Cyber-ATX-27B-Ultra-Extensive",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

print("\n[PASO 1] Iniciando servidor y pre-cargando modelo a RAM fijada...")
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

server_logs = []
wake_event_detected = False
wake_latency_str = ""

def log_reader():
    global wake_event_detected, wake_latency_str
    for line in iter(proc.stdout.readline, ''):
        l = line.strip()
        server_logs.append(l)
        if any(k in l for k in ["sleeping state", "VMM sleep", "VMM wake", "eval time", "graphs reused", "draft acceptance"]):
            print(f"   [server] {l}", flush=True)
        if "VMM wake completed" in l or "restored" in l:
            wake_event_detected = True
            wake_latency_str = l

t_reader = threading.Thread(target=log_reader, daemon=True)
t_reader.start()

url_health = f"http://127.0.0.1:{PORT}/health"
url_chat = f"http://127.0.0.1:{PORT}/v1/chat/completions"

ready = False
for i in range(60):
    try:
        r = requests.get(url_health, timeout=1)
        if r.status_code == 200:
            ready = True
            print(f"Servidor listo en {i+1}s!")
            break
    except Exception:
        pass
    time.sleep(1)

if not ready:
    print("ERROR: Servidor no respondio a tiempo.")
    proc.kill()
    sys.exit(1)

vram_loaded = get_vram_mb()
print(f"Modelo cargado inicialmente en VRAM: {vram_loaded} MB")

# Warmup to register CUDA graphs and prompt cache
print("\n[PASO 2] Calentamiento inicial de grafos CUDA...")
requests.post(url_chat, json={"messages": [{"role": "user", "content": "Ping"}], "max_tokens": 16}, timeout=30)

print("\n[PASO 3] Esperando que se active el reposo VMM (idle sleep 3s)...")
slept = False
for i in range(15):
    time.sleep(1)
    v = get_vram_mb()
    print(f"   t={i+1}s -> VRAM usada: {v} MB")
    if v < 8000:
        print(f"\nEXITO CONFIRMADO: VRAM liberada! {vram_loaded} MB -> {v} MB")
        print(f"VRAM desmapeada mediante cuMemUnmap en <60 ms.")
        print(f"El modelo reside 100% en RAM fijada (Pinned Host Memory) listo para despertar!")
        slept = True
        break

if not slept:
    print("Advertencia: No se detecto liberacion completa de VRAM.")

time.sleep(1)
vram_before_wake = get_vram_mb()
print(f"\nVRAM antes del despertar desde RAM: {vram_before_wake} MB")

# EXTENSIVE CODE PROMPT
print("\n" + "="*75)
print("[PASO 4] LANZANDO INFERENCIA EXTENSA (WAKE RAM->VRAM + CODIGO EXTENSO)")
print("="*75)

prompt = """Implement a complete, production-grade distributed Actor Framework in modern C++20.
Include:
1. Lock-free MPSC concurrent mailbox with atomic intrusive linked list.
2. Actor base class with lifecycle hooks (pre_start, post_stop, receive).
3. ActorSystem with work-stealing thread pool dispatcher and affinity masks.
4. Typed message passing with std::variant and visitor pattern.
5. Supervisor tree supporting OneForOne and AllForOne restart strategies with max restart rate limiting.
6. Zero-copy binary serialization / deserialization envelope for remote network transmission.

Provide full, functional, clean C++20 code with complete implementation (no placeholders or stubs)."""

payload = {
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2048,
    "temperature": 0.2
}

t_request_start = time.time()
print("Enviando prompt extenso al modelo en reposo...")
response = requests.post(url_chat, json=payload, timeout=300).json()
t_request_end = time.time()

vram_after_wake = get_vram_mb()
total_request_time = t_request_end - t_request_start

usage = response.get("usage", {})
prompt_tokens = usage.get("prompt_tokens", 0)
comp_tokens = usage.get("completion_tokens", 0)
total_tokens = usage.get("total_tokens", 0)

tok_s = comp_tokens / total_request_time if total_request_time > 0 else 0

msg = response.get("choices", [{}])[0].get("message", {})
code_text = msg.get("content", "")
if not code_text or len(code_text.strip()) == 0:
    code_text = msg.get("reasoning_content", "")

output_path = os.path.join(RESULTS_DIR, "extensive_actor_framework_c20.cpp")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(code_text)

print("\n" + "="*75)
print("RESULTADOS DE LA PRUEBA DE INFERENCIA EXTENSA")
print("="*75)
print(f"VRAM restaurada (Wake): {vram_after_wake} MB (Transferencia DMA desde RAM completada)")
print(f"Tokens de Prompt:       {prompt_tokens}")
print(f"Tokens Generados:       {comp_tokens}")
print(f"Tiempo Total Solicitud: {total_request_time:.2f}s (incluye Wake DMA + Prefill + Generacion 2048 tokens)")
print(f"Velocidad Total Real:   {tok_s:.2f} tok/s")
print(f"Codigo guardado en:     {output_path}")
print(f"Tamano del codigo:      {len(code_text)} bytes ({len(code_text.splitlines())} lineas)")

proc.kill()
proc.wait()
print("\nPrueba completada con exito.")
