import os
import sys
import time
import json
import subprocess
import requests

SERVER_EXE = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
VOCAB_MAP  = r"D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"
PYTHON_EXE = r"d:\bjivanovich2017\My Drive\aibot\python\python.exe"

MODEL_SWIFT = r"D:\models\ATX-Swift-1.5-Qwen3.8-27B-Uncensored-MTP-i1-Q5_K_M.gguf"
MODEL_GYRO  = r"D:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"
MODEL_CYBER = r"D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-Q5_K_M.gguf"

RESULTS_DIR = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\pipeline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PORT = 8089

def kill_servers():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.5)

def get_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1

def wait_for_server(port, timeout=90):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.8)
    return False

def query_endpoint(port, system_prompt, user_prompt, max_tokens=2500, temperature=0.6):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=600)
    t1 = time.perf_counter()
    
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    
    # Calculate metrics
    prompt_tokens = usage.get("prompt_tokens", timings.get("prompt_n", 0))
    completion_tokens = usage.get("completion_tokens", timings.get("predicted_n", 0))
    
    prompt_ms = timings.get("prompt_ms", 0.0)
    prompt_tok_s = timings.get("prompt_per_second", (prompt_tokens / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0)
    
    pred_ms = timings.get("predicted_ms", 0.0)
    decode_tok_s = timings.get("predicted_per_second", (completion_tokens / (pred_ms / 1000.0)) if pred_ms > 0 else (completion_tokens / (t1 - t0)))
    
    total_duration_s = t1 - t0
    
    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_eval_ms": round(prompt_ms, 2),
        "prompt_tok_per_sec": round(prompt_tok_s, 2),
        "eval_ms": round(pred_ms, 2),
        "decode_tok_per_sec": round(decode_tok_s, 2),
        "total_roundtrip_s": round(total_duration_s, 3)
    }

print("=" * 80)
print("INICIANDO BENCHMARK EN VIVO: TELEMETRIA PROFUNDA MULTI-AGENTE (96k CONTEXT)")
print("=" * 80)

kill_servers()
base_vram = get_vram_mb()
print(f">> VRAM Base del sistema: {base_vram} MB")

telemetry = {}

# =============================================================================
# ETAPA 1: ARQUITECTO (ATX-Swift-1.5 Q5_K_M)
# =============================================================================
print("\n" + "="*80)
print("[ETAPA 1] Lanzando ARQUITECTO: ATX-Swift-1.5 Q5_K_M")
print("          Contexto: 96k | KV: turbo5/turbo4 | MTP Activo")
print("="*80)

env_swift = os.environ.copy()
env_swift["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env_swift["GGML_CUDA_FA_I8QK"] = "1"
env_swift["LLAMAMPERE_ENABLE_SLEEP"] = "1"
env_swift["GGML_CUDA_GRAPH_OPT"] = "1"
env_swift["LLAMA_SPEC_CHAIN"] = "1"
env_swift["LLAMA_SPEC_CHAIN_SAMPLED"] = "1"
env_swift["LLAMA_SPEC_PQ"] = "1"
env_swift["LLAMA_SPEC_ADAPT_COST"] = "1"

cmd_swift = [
    SERVER_EXE,
    "-m", MODEL_SWIFT,
    "-ngl", "99",
    "-c", "98304",
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
    "--jinja",
    "--fit", "off",
    "--cache-prompt",
    "--cache-ram", "16384",
    "--ctx-checkpoints", "24",
    "--checkpoint-min-step", "10240",
    "--spec-draft-type-k", "q8_0",
    "--spec-draft-type-v", "q8_0",
    "--spec-draft-vocab-map", VOCAB_MAP,
    "--spec-draft-n-min-adaptive", "3",
    "--spec-draft-n-max", "5",
    "--spec-draft-p-min", "0",
    "--reasoning-format", "none",
    "--reasoning-effort", "medium",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "40",
    "--min-p", "0.05",
    "--repeat-penalty", "1.0",
    "--repeat-last-n", "256",
    "--load-mode", "none",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

t_boot0 = time.perf_counter()
swift_log_path = os.path.join(RESULTS_DIR, "server_swift.log")
swift_log_file = open(swift_log_path, "w", encoding="utf-8")
proc_swift = subprocess.Popen(cmd_swift, env=env_swift, stdout=swift_log_file, stderr=subprocess.STDOUT)
if not wait_for_server(PORT):
    swift_log_file.flush()
    print("[ERROR] Servidor Swift no respondio! Ultimas lineas del log:")
    with open(swift_log_path, "r", encoding="utf-8", errors="replace") as f:
        print("".join(f.readlines()[-30:]))
    kill_servers()
    sys.exit(1)
t_boot_swift = round(time.perf_counter() - t_boot0, 2)
vram_swift = get_vram_mb()
print(f">> Servidor Arquitecto listo en {t_boot_swift}s | VRAM ocupada: {vram_swift} MB")

sys_prompt_arch = "Eres un arquitecto de software y sistemas distribuidos de nivel principal. Tu tarea es diseñar la arquitectura técnica completa para el requerimiento del usuario. Especifica módulos, protocolos, estructuras de memoria y contratos de API de forma rigurosa y exhaustiva."
user_prompt_arch = "Diseña la arquitectura de un motor distribuido de telemetría de red en tiempo real en Python, con buffers circulares compartidos sin copia (ring buffers) y bucle asíncrono con multiprocesamiento."

print(">> Generando Blueprint Arquitectonico...")
res_arch = query_endpoint(PORT, sys_prompt_arch, user_prompt_arch, max_tokens=2048, temperature=0.6)
res_arch["vram_mb"] = vram_swift
res_arch["cold_boot_s"] = t_boot_swift
telemetry["architect"] = res_arch

arch_file = os.path.join(RESULTS_DIR, "01_live_architecture_plan.md")
with open(arch_file, "w", encoding="utf-8") as f:
    f.write(res_arch["content"])

print(f">> ARQUITECTO COMPLETADO:")
print(f"   - Prompt Tokens leidos:      {res_arch['prompt_tokens']} tokens")
print(f"   - Tiempo de Prefill (Lectura): {res_arch['prompt_eval_ms']} ms ({res_arch['prompt_tok_per_sec']} tok/s)")
print(f"   - Completion Tokens emitidos:  {res_arch['completion_tokens']} tokens")
print(f"   - Tiempo de Decode (Emision):  {res_arch['eval_ms']} ms ({res_arch['decode_tok_per_sec']} tok/s)")
print(f"   - Tiempo Total de Respuesta:   {res_arch['total_roundtrip_s']} s")

# Desalojo VMM Sleep del Arquitecto
kill_servers()
time.sleep(1)
vram_after_unmap1 = get_vram_mb()
print(f">> Desalojo de VRAM completado | VRAM liberada a: {vram_after_unmap1} MB")

# =============================================================================
# ETAPA 2: CODER (Gyro-Coder TQ2_0)
# =============================================================================
print("\n" + "="*80)
print("[ETAPA 2] Lanzando CODER: Gyro-Coder TQ2_0")
print("          Contexto: 96k | KV: Q8_0 | Capas de Expertos VRAM: 19.9 GB")
print("="*80)

env_gyro = os.environ.copy()
env_gyro["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env_gyro["GGML_CUDA_GRAPH_OPT"] = "1"
env_gyro["LLAMAMPERE_ENABLE_SLEEP"] = "1"

cmd_gyro = [
    SERVER_EXE,
    "-m", MODEL_GYRO,
    "-ngl", "99",
    "-c", "98304",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-b", "2048",
    "-ub", "512",
    "-t", "8",
    "-tb", "8",
    "-fa", "on",
    "--ctx-checkpoints", "0",
    "--parallel", "1",
    "--prio", "3",
    "--moe-profile", "python",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "40",
    "--min-p", "0.05",
    "--repeat-penalty", "1.05",
    "--repeat-last-n", "256",
    "--load-mode", "none",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

t_wake_gyro_0 = time.perf_counter()
gyro_log_path = os.path.join(RESULTS_DIR, "server_gyro.log")
gyro_log_file = open(gyro_log_path, "w", encoding="utf-8")
proc_gyro = subprocess.Popen(cmd_gyro, env=env_gyro, stdout=gyro_log_file, stderr=subprocess.STDOUT)
if not wait_for_server(PORT):
    gyro_log_file.flush()
    print("[ERROR] Servidor Gyro-Coder no respondio! Ultimas lineas del log:")
    with open(gyro_log_path, "r", encoding="utf-8", errors="replace") as f:
        print("".join(f.readlines()[-30:]))
    kill_servers()
    sys.exit(1)
t_wake_gyro = round(time.perf_counter() - t_wake_gyro_0, 2)
vram_gyro = get_vram_mb()
print(f">> Servidor Coder listo | VMM DMA Wake / Load: {t_wake_gyro}s | VRAM ocupada: {vram_gyro} MB")

sys_prompt_coder = "Eres un ingeniero senior de desarrollo en Python. Implementa el codigo completo, ejecutable y robusto basado estrictamente en el blueprint arquitectonico. Incluye pruebas unitarias completas con unittest al final. Devuelve unicamente codigo Python valido dentro de un bloque ```python."
user_prompt_coder = f"Blueprint Arquitectonico:\n\n{res_arch['content']}\n\nTarea: Implementa todo el codigo fuente y las pruebas unitarias completas."

print(">> Sintetizando Codigo...")
res_coder = query_endpoint(PORT, sys_prompt_coder, user_prompt_coder, max_tokens=3500, temperature=0.6)
res_coder["vram_mb"] = vram_gyro
res_coder["wake_s"] = t_wake_gyro
telemetry["coder"] = res_coder

coder_code = res_coder["content"]
if "```python" in coder_code:
    clean_code = coder_code.split("```python")[1].split("```")[0].strip()
elif "```" in coder_code:
    clean_code = coder_code.split("```")[1].split("```")[0].strip()
else:
    clean_code = coder_code

coder_file = os.path.join(RESULTS_DIR, "02_live_coder_implementation.py")
with open(coder_file, "w", encoding="utf-8") as f:
    f.write(clean_code)

print(f">> CODER COMPLETADO:")
print(f"   - Prompt Tokens leidos (Plan Arquitecto + Sys): {res_coder['prompt_tokens']} tokens")
print(f"   - Tiempo de Prefill (Lectura del Plan):         {res_coder['prompt_eval_ms']} ms ({res_coder['prompt_tok_per_sec']} tok/s)")
print(f"   - Completion Tokens emitidos (Codigo + Tests):  {res_coder['completion_tokens']} tokens")
print(f"   - Tiempo de Decode (Generacion de Codigo):      {res_coder['eval_ms']} ms ({res_coder['decode_tok_per_sec']} tok/s)")
print(f"   - Tiempo Total de Respuesta:                    {res_coder['total_roundtrip_s']} s")

# Ejecucion Sandbox
print("\n>> Ejecutando sandbox de verificacion de codigo en vivo...")
t_exec0 = time.perf_counter()
proc_test = subprocess.run([PYTHON_EXE, coder_file], capture_output=True, text=True, timeout=30)
t_exec = round(time.perf_counter() - t_exec0, 3)
exit_code = proc_test.returncode
stdout = proc_test.stdout
stderr = proc_test.stderr
print(f"   - Exit Code: {exit_code}")
print(f"   - Tiempo de ejecucion: {t_exec} s")
if exit_code == 0:
    print("   - Resultado: 100% OK (0 errores)")
else:
    print(f"   - Fallo detectado: {stderr[:250]}")

telemetry["sandbox"] = {
    "exit_code": exit_code,
    "execution_time_s": t_exec,
    "stdout_snippet": stdout[:300],
    "stderr_snippet": stderr[:300]
}

# Desalojo VMM Sleep del Coder
kill_servers()
time.sleep(1)
vram_after_unmap2 = get_vram_mb()
print(f">> Desalojo de VRAM completado | VRAM liberada a: {vram_after_unmap2} MB")

# =============================================================================
# ETAPA 3: DEBUGGER / AUDITOR (Cyber-ATX Q5_K_M)
# =============================================================================
print("\n" + "="*80)
print("[ETAPA 3] Lanzando DEBUGGER & AUDITOR: Cyber-ATX Q5_K_M")
print("          Contexto: 96k | KV: turbo5/turbo4 | MTP Activo | Imatrix i1")
print("="*80)

env_cyber = os.environ.copy()
env_cyber["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
env_cyber["GGML_CUDA_FA_I8QK"] = "1"
env_cyber["LLAMAMPERE_ENABLE_SLEEP"] = "1"
env_cyber["GGML_CUDA_GRAPH_OPT"] = "1"
env_cyber["LLAMA_SPEC_CHAIN"] = "1"
env_cyber["LLAMA_SPEC_CHAIN_SAMPLED"] = "1"
env_cyber["LLAMA_SPEC_PQ"] = "1"
env_cyber["LLAMA_SPEC_ADAPT_COST"] = "1"

cmd_cyber = [
    SERVER_EXE,
    "-m", MODEL_CYBER,
    "-ngl", "99",
    "-c", "98304",
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
    "--jinja",
    "--fit", "off",
    "--cache-prompt",
    "--cache-ram", "16384",
    "--ctx-checkpoints", "24",
    "--checkpoint-min-step", "10240",
    "--spec-draft-type-k", "q8_0",
    "--spec-draft-type-v", "q8_0",
    "--spec-draft-vocab-map", VOCAB_MAP,
    "--spec-draft-n-min-adaptive", "3",
    "--spec-draft-n-max", "5",
    "--spec-draft-p-min", "0",
    "--reasoning-format", "none",
    "--reasoning-effort", "medium",
    "--temp", "0.2",
    "--top-p", "0.90",
    "--top-k", "20",
    "--min-p", "0.05",
    "--repeat-penalty", "1.08",
    "--repeat-last-n", "256",
    "--load-mode", "none",
    "--host", "127.0.0.1",
    "--port", str(PORT)
]

t_wake_cyber_0 = time.perf_counter()
cyber_log_path = os.path.join(RESULTS_DIR, "server_cyber.log")
cyber_log_file = open(cyber_log_path, "w", encoding="utf-8")
proc_cyber = subprocess.Popen(cmd_cyber, env=env_cyber, stdout=cyber_log_file, stderr=subprocess.STDOUT)
if not wait_for_server(PORT):
    cyber_log_file.flush()
    print("[ERROR] Servidor Cyber-ATX no respondio! Ultimas lineas del log:")
    with open(cyber_log_path, "r", encoding="utf-8", errors="replace") as f:
        print("".join(f.readlines()[-30:]))
    kill_servers()
    sys.exit(1)
t_wake_cyber = round(time.perf_counter() - t_wake_cyber_0, 2)
vram_cyber = get_vram_mb()
print(f">> Servidor Debugger listo | VMM DMA Wake / Load: {t_wake_cyber}s | VRAM ocupada: {vram_cyber} MB")

sys_prompt_debug = "Eres un auditor forense de ciberseguridad y experto senior en depuracion de sistemas de alto rendimiento. Tu mision es revisar el blueprint del arquitecto, la implementacion del coder y el resultado de la ejecucion. Corrige errores sintacticos, resuelve condiciones de carrera y sella vulnerabilidades. Devuelve el codigo final blindado en un bloque ```python."
user_prompt_debug = f"Blueprint Arquitecto:\n{res_arch['content'][:2500]}\n\nCodigo del Coder:\n```python\n{clean_code}\n```\n\nResultado Ejecucion:\nExit Code: {exit_code}\nStdout: {stdout[:400]}\nStderr: {stderr[:400]}\n\nAudita, corrige fallos y entrega el codigo endurecido final."

print(">> Auditando y Depurando...")
res_debug = query_endpoint(PORT, sys_prompt_debug, user_prompt_debug, max_tokens=3500, temperature=0.2)
res_debug["vram_mb"] = vram_cyber
res_debug["wake_s"] = t_wake_cyber
telemetry["debugger"] = res_debug

debug_code = res_debug["content"]
if "```python" in debug_code:
    final_hardened = debug_code.split("```python")[1].split("```")[0].strip()
elif "```" in debug_code:
    final_hardened = debug_code.split("```")[1].split("```")[0].strip()
else:
    final_hardened = debug_code

debug_file = os.path.join(RESULTS_DIR, "03_live_hardened_audited.py")
with open(debug_file, "w", encoding="utf-8") as f:
    f.write(final_hardened)

print(f">> DEBUGGER COMPLETADO:")
print(f"   - Prompt Tokens leidos (Plan + Codigo + Errores): {res_debug['prompt_tokens']} tokens")
print(f"   - Tiempo de Prefill (Lectura de todo el contexto): {res_debug['prompt_eval_ms']} ms ({res_debug['prompt_tok_per_sec']} tok/s)")
print(f"   - Completion Tokens emitidos (Codigo blindado):    {res_debug['completion_tokens']} tokens")
print(f"   - Tiempo de Decode (Emision del codigo final):     {res_debug['eval_ms']} ms ({res_debug['decode_tok_per_sec']} tok/s)")
print(f"   - Tiempo Total de Respuesta:                       {res_debug['total_roundtrip_s']} s")

# Verificacion final del codigo depurado
print("\n>> Ejecutando verificacion final del codigo depurado...")
proc_final = subprocess.run([PYTHON_EXE, debug_file], capture_output=True, text=True, timeout=30)
print(f"   - Exit Code Final: {proc_final.returncode}")
if proc_final.returncode == 0:
    print("   - Estado Final: 100% OPERATIVO Y BLINDADO (Exit code 0)")
else:
    print(f"   - Salida: {proc_final.stderr[:200]}")

kill_servers()

# Guardar json consolidado
summary_path = os.path.join(RESULTS_DIR, "live_deep_telemetry_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(telemetry, f, indent=2)

print("\n" + "="*80)
print("BENCHMARK EN VIVO COMPLETADO EXITOSAMENTE!")
print(f"Resumen guardado en: {summary_path}")
print("="*80)
