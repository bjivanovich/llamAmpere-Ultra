import os
import sys
import time
import json
import subprocess
import threading
import requests

def get_vram():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(res.strip().split("\n")[0])
    except Exception:
        return -1

server_bin = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
model_path = r"D:\models\Qwen3.8-Flash-Next-160Exp-TQ3_1S.gguf"
ple_path   = r"D:\models\Qwen3.8-Flash-Next-PLE-Q8_0.gguf"
port = 8088

env = os.environ.copy()
env["LLAMA_MOE_ARENA"] = "off"
env["GGML_CUDA_GRAPH_OPT"] = "1"
env["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env["GGML_CUDA_FA_I8QK"] = "1"
env["GGML_TQ_NATIVE"] = "1"
env["GGML_CUDA_Q8CACHE"] = "1"
env["GGML_CUDA_FUSE_CHAIN"] = "1"
env["LLAMAMPERE_ENABLE_SLEEP"] = "1"

cmd = [
    server_bin,
    "-m", model_path,
    "--ple", ple_path,
    "--port", str(port),
    "-ngl", "999",
    "-c", "32768",
    "-ctk", "turbo4",
    "-ctv", "turbo4",
    "-b", "2048",
    "-ub", "512",
    "-t", "8",
    "-tb", "8",
    "-fa", "on",
    "--jinja",
    "--ctx-checkpoints", "0",
    "--parallel", "1",
    "--prio", "3",
    "--moe-profile", "python",
    "--ignore-eos",
    "--alias", "Qwen3.8-Flash-Next-160Exp-TQ3_1S"
]

print("=" * 65)
print("  llamAmpere-Ultra: Test 1,500 Tokens con TQ3_1S + DP4A")
print(f"  Modelo: 160Exp-TQ3_1S (3.25 bpw) | PLE: {ple_path}")
print(f"  VRAM Inicial: {get_vram()} MB")
print("=" * 65)

proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def print_server_logs():
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        l = line.strip()
        print(f"  [SERVER] {l}", flush=True)

t_log = threading.Thread(target=print_server_logs, daemon=True)
t_log.start()

print("\nEsperando inicializacion de llamAmpere-Ultra en puerto 8088...")
t0 = time.time()
ready = False
for _ in range(120):
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
        if r.status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(1.0)

if not ready:
    print("ERROR: El servidor llamAmpere-Ultra no inicio dentro de los 120s.")
    proc.terminate()
    try:
        proc.kill()
    except Exception:
        pass
    sys.exit(1)

vram_peak = get_vram()
print(f"\n>>> Servidor listo en {time.time() - t0:.2f}s! VRAM cargada: {vram_peak} MB <<<\n")

prompt = (
    "Por favor escribe un articulo exhaustivo y un modulo completo de programacion en C++ de alta eficiencia "
    "para procesamiento de senales digitales con la Transformada Walsh-Hadamard (WHT) de 128 elementos. "
    "Incluye teoria microarquitectural, uso de vectorizacion AVX2/AVX-512, implementacion de bit-reversal, "
    "un bucle butterfly sin recursividad, benchmarks con std::chrono y casos de prueba unitarios detallados. "
    "Explayate extensamente con explicaciones paso a paso, codigo completo y comentarios en cada seccion."
)

payload = {
    "model": "Qwen3.8-Flash-Next-160Exp-TQ3_1S",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1500,
    "temperature": 0.6,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "stream": True,
    "ignore_eos": True
}

print("=" * 65)
print("  Iniciando streaming en vivo de 1,500 tokens...")
print("=" * 65)

t_req_start = time.perf_counter()
t_first_token = None
token_count = 0
last_report = 0
tokens_generated = []

try:
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=900
    )
    print(f"HTTP Response Status: {resp.status_code}")

    if resp.status_code != 200:
        print("ERROR RESPONSE:", resp.text)
    else:
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            line_str = line_str.strip()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    token_count += 1
                    tokens_generated.append(content)
                    now = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = now
                        ttft = (t_first_token - t_req_start) * 1000.0
                        print(f"  [TTFT] Primer token generado en {ttft:.2f} ms")

                    if token_count - last_report >= 100 or token_count == 1500:
                        last_report = token_count
                        cur_vram = get_vram()
                        cur_speed = (token_count - 1) / (now - t_first_token) if now > t_first_token else 0
                        elapsed = now - t_first_token
                        print(f"  Tokens: {token_count:4d}/1500 | Tiempo: {elapsed:5.1f}s | Velocidad: {cur_speed:5.2f} tok/s | VRAM: {cur_vram} MB", flush=True)

            except Exception as e:
                pass

except Exception as e:
    print(f"Request exception: {e}")

t_end = time.perf_counter()

# Cleanup server
try:
    proc.terminate()
    proc.kill()
except Exception:
    pass

total_gen_time = t_end - (t_first_token if t_first_token else t_req_start)
sustained_speed = token_count / total_gen_time if total_gen_time > 0 and token_count > 0 else 0

print("\n" + "=" * 65)
print("  RESULTADOS DEL TEST: 1,500 TOKENS TQ3_1S EN LLAMAMPERE-ULTRA")
print("=" * 65)
print(f"  Tokens Totales Generados: {token_count}")
if t_first_token:
    print(f"  TTFT (Time To First Token): {(t_first_token - t_req_start)*1000.0:.2f} ms")
print(f"  Tiempo Total de Generacion: {total_gen_time:.2f} segundos")
print(f"  Velocidad Sostenida Real:   {sustained_speed:.2f} tokens/segundo")
print(f"  VRAM Pico Ocupada:          {vram_peak} MB")
print(f"  VRAM Final Liberada:        {get_vram()} MB")
print("=" * 65)

output_file = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\test_1500_tq3_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Tokens: {token_count}\n")
    if t_first_token:
        f.write(f"TTFT_ms: {(t_first_token - t_req_start)*1000.0:.2f}\n")
    f.write(f"Generation_Time_s: {total_gen_time:.2f}\n")
    f.write(f"Speed_tok_per_s: {sustained_speed:.2f}\n")
    f.write(f"Peak_VRAM_MB: {vram_peak}\n")
    f.write("\n--- Texto Generado (Extracto) ---\n")
    f.write("".join(tokens_generated[:500]))
print(f"\nResultados guardados en {output_file}")
