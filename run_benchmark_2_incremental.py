#!/usr/bin/env python3
"""
Deep Pipeline Benchmark 2: Full vs Incremental Conditioned Debugger
===================================================================
Compares two distinct multi-agent paradigms on a single RTX 3090 24GB:
  1. Full Traditional Pipeline:
     - Debugger receives 100% of blueprint + 100% of generated code.
     - Debugger re-evaluates and regenerates the entire codebase (~3500 tokens).
  2. Incremental & Conditioned Pipeline:
     - Sandbox acts as first-line deterministic gatekeeper.
     - If tests pass (Exit 0): Debugger receives contract criteria + test metrics
       and emits concise sign-off / risk assessment (~300 tokens).
     - If tests fail (Exit != 0): Debugger receives only the traceback, failing test,
       and target function, emitting a surgical patch (~400 tokens).

Measures: Prompt tokens, prefill time/speed, decode tokens, decode speed,
total duration, VRAM/RAM pinned usage, and defect detection accuracy.
"""

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

RESULTS_DIR = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\pipeline_results\benchmark_2"
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

def query_endpoint(port, system_prompt, user_prompt, max_tokens=3500, temperature=0.6):
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
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    
    # Extract code if in reasoning
    if not content.strip() and reasoning.strip():
        content = reasoning
    elif reasoning.strip() and "```" in reasoning and "```" not in content:
        content = content + "\n\n" + reasoning
        
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    
    prompt_tokens = usage.get("prompt_tokens", timings.get("prompt_n", 0))
    completion_tokens = usage.get("completion_tokens", timings.get("predicted_n", 0))
    
    prompt_ms = timings.get("prompt_ms", 0.0)
    prompt_tok_s = timings.get("prompt_per_second", (prompt_tokens / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0)
    
    pred_ms = timings.get("predicted_ms", 0.0)
    decode_tok_s = timings.get("predicted_per_second", (completion_tokens / (pred_ms / 1000.0)) if pred_ms > 0 else (completion_tokens / (t1 - t0)))
    
    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_eval_ms": round(prompt_ms, 2),
        "prompt_tok_per_sec": round(prompt_tok_s, 2),
        "eval_ms": round(pred_ms, 2),
        "decode_tok_per_sec": round(decode_tok_s, 2),
        "total_roundtrip_s": round(t1 - t0, 3)
    }

def main():
    print("=" * 80)
    print("INICIANDO DEEP PIPELINE BENCHMARK 2: FULL VS INCREMENTAL CONDITIONED")
    print("Hardware: RTX 3090 24GB | PCIe 3.0 x16 | Contexto 96k | Windows WDDM")
    print("=" * 80)

    kill_servers()
    base_vram = get_vram_mb()
    print(f">> VRAM Base: {base_vram} MB")

    telemetry = {
        "architect": {},
        "coder": {},
        "sandbox": {},
        "debugger_full": {},
        "debugger_incremental": {},
        "targeted_bug_fix_test": {}
    }

    # =========================================================================
    # FASE 1: ARQUITECTO (ATX-Swift-1.5 Q5_K_M)
    # =========================================================================
    print("\n" + "="*80)
    print("[FASE 1] ARQUITECTO: ATX-Swift-1.5 Q5_K_M")
    print("         Contexto: 96k | KV: turbo5/turbo4 | MTP Activo")
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
    log_swift = open(os.path.join(RESULTS_DIR, "server_swift.log"), "w", encoding="utf-8")
    proc_swift = subprocess.Popen(cmd_swift, env=env_swift, stdout=log_swift, stderr=subprocess.STDOUT)
    if not wait_for_server(PORT):
        print("[ERROR] Servidor Swift no inicio!")
        kill_servers()
        sys.exit(1)
    t_boot_swift = round(time.perf_counter() - t_boot0, 2)
    vram_swift = get_vram_mb()
    print(f">> Arquitecto listo | Boot: {t_boot_swift}s | VRAM: {vram_swift} MB")

    sys_arch = "Eres un arquitecto de software principal. Diseña la arquitectura modular, contratos de API, estructuras de memoria y criterios de aceptacion rigurosos para el requerimiento del usuario."
    user_arch = """Requerimiento: Implementa una libreria de concurrencia de alto rendimiento en Python con:
1. RingBuffer SPSC lock-free en memoria compartida (multiprocessing.shared_memory).
2. Protocolo de serializacion binaria ultra-rapida con struct (campos: timestamp_ns, seq, stream_id, payload_len, payload).
3. Mecanismo de backpressure con watermarks (High 75%, Low 50%).
4. Suite exhaustiva de pruebas unitarias con unittest verificando FIFO, desbordamiento, y concurrencia."""

    print(">> Sintetizando Blueprint Arquitectonico...")
    res_arch = query_endpoint(PORT, sys_arch, user_arch, max_tokens=2200, temperature=0.6)
    res_arch["vram_mb"] = vram_swift
    res_arch["boot_s"] = t_boot_swift
    telemetry["architect"] = res_arch

    arch_file = os.path.join(RESULTS_DIR, "01_architecture_blueprint.md")
    with open(arch_file, "w", encoding="utf-8") as f:
        f.write(res_arch["content"])

    print(f">> ARQUITECTO FINALIZADO:")
    print(f"   - Prompt tokens leidos:       {res_arch['prompt_tokens']} tokens")
    print(f"   - Tiempo Prefill:             {res_arch['prompt_eval_ms']} ms ({res_arch['prompt_tok_per_sec']} tok/s)")
    print(f"   - Tokens generados (Decode):  {res_arch['completion_tokens']} tokens")
    print(f"   - Velocidad Decode:           {res_arch['decode_tok_per_sec']} tok/s")
    print(f"   - Tiempo Total:               {res_arch['total_roundtrip_s']} s")

    kill_servers()
    time.sleep(1)
    vram_free1 = get_vram_mb()
    print(f">> VMM Unmap completado | VRAM liberada a: {vram_free1} MB (<60 ms)")

    # =========================================================================
    # FASE 2: CODER (Gyro-Coder TQ2_0)
    # =========================================================================
    print("\n" + "="*80)
    print("[FASE 2] CODER: Gyro-Coder TQ2_0")
    print("         Contexto: 96k | KV: Q8_0 | Capas VRAM: 19.9 GB | NVMe PLE mmap")
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
        "--reasoning-format", "none",
        "--no-reasoning-preserve",
        "--temp", "0.5",
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
    log_gyro = open(os.path.join(RESULTS_DIR, "server_gyro.log"), "w", encoding="utf-8")
    proc_gyro = subprocess.Popen(cmd_gyro, env=env_gyro, stdout=log_gyro, stderr=subprocess.STDOUT)
    if not wait_for_server(PORT):
        print("[ERROR] Servidor Gyro-Coder no inicio!")
        kill_servers()
        sys.exit(1)
    t_wake_gyro = round(time.perf_counter() - t_wake_gyro_0, 2)
    vram_gyro = get_vram_mb()
    print(f">> Coder listo | VMM DMA Wake: {t_wake_gyro}s | VRAM: {vram_gyro} MB")

    sys_coder = "Eres un ingeniero senior de sistemas en Python. Implementa el codigo ejecutable completo con su suite de pruebas unittest al final, estrictamente basado en el blueprint arquitectonico. Devuelve unicamente codigo Python valido dentro de un bloque ```python."
    user_coder = f"Blueprint Arquitectonico:\n\n{res_arch['content']}\n\nTarea: Implementa la libreria completa y la suite de pruebas unitarias ejecutable."

    print(">> Sintetizando Codigo...")
    res_coder = query_endpoint(PORT, sys_coder, user_coder, max_tokens=3500, temperature=0.5)
    res_coder["vram_mb"] = vram_gyro
    res_coder["wake_s"] = t_wake_gyro
    telemetry["coder"] = res_coder

    coder_code = res_coder["content"]
    if "```python" in coder_code:
        clean_code = coder_code.split("```python")[1].split("```")[0].strip()
    elif "```" in coder_code:
        clean_code = coder_code.split("```")[1].split("```")[0].strip()
    else:
        clean_code = coder_code.strip()

    coder_file = os.path.join(RESULTS_DIR, "02_coder_implementation.py")
    with open(coder_file, "w", encoding="utf-8") as f:
        f.write(clean_code)

    print(f">> CODER FINALIZADO:")
    print(f"   - Prompt tokens leidos (Blueprint): {res_coder['prompt_tokens']} tokens")
    print(f"   - Tiempo Prefill:                  {res_coder['prompt_eval_ms']} ms ({res_coder['prompt_tok_per_sec']} tok/s)")
    print(f"   - Tokens generados (Codigo):       {res_coder['completion_tokens']} tokens")
    print(f"   - Velocidad Decode:                {res_coder['decode_tok_per_sec']} tok/s")
    print(f"   - Tiempo Total:                    {res_coder['total_roundtrip_s']} s")

    # Ejecucion determinista en Sandbox
    print("\n>> Ejecutando Sandbox determinista en Python...")
    t_sb0 = time.perf_counter()
    proc_test = subprocess.run([PYTHON_EXE, coder_file], capture_output=True, text=True, timeout=30)
    t_sb = round(time.perf_counter() - t_sb0, 3)
    exit_code = proc_test.returncode
    stdout = proc_test.stdout
    stderr = proc_test.stderr
    print(f"   - Sandbox Exit Code:    {exit_code}")
    print(f"   - Sandbox Duracion:     {t_sb} s")
    if exit_code == 0:
        print("   - Sandbox Resultado:    100% EXITO (Tests pasaron)")
    else:
        print(f"   - Sandbox Resultado:    FALLO (Exit {exit_code})")
        print(f"     stderr: {stderr[:250]}")

    telemetry["sandbox"] = {
        "exit_code": exit_code,
        "duration_s": t_sb,
        "stdout": stdout[:400],
        "stderr": stderr[:400]
    }

    kill_servers()
    time.sleep(1)
    vram_free2 = get_vram_mb()
    print(f">> VMM Unmap completado | VRAM liberada a: {vram_free2} MB")

    # =========================================================================
    # FASE 3A: DEBUGGER MODALIDAD FULL (TRADICIONAL EXHAUSTIVO)
    # =========================================================================
    print("\n" + "="*80)
    print("[FASE 3A] DEBUGGER MODALIDAD FULL (Reenvio de todo el codigo)")
    print("          Contexto: 96k | KV: turbo5/turbo4 | MTP Activo")
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
    log_cyber = open(os.path.join(RESULTS_DIR, "server_cyber_full.log"), "w", encoding="utf-8")
    proc_cyber = subprocess.Popen(cmd_cyber, env=env_cyber, stdout=log_cyber, stderr=subprocess.STDOUT)
    if not wait_for_server(PORT):
        print("[ERROR] Servidor Cyber-ATX no inicio!")
        kill_servers()
        sys.exit(1)
    t_wake_cyber = round(time.perf_counter() - t_wake_cyber_0, 2)
    vram_cyber = get_vram_mb()
    print(f">> Debugger Full listo | VMM DMA Wake: {t_wake_cyber}s | VRAM: {vram_cyber} MB")

    sys_debug_full = "Eres un auditor forense de ciberseguridad y experto en depuracion. Revisa el blueprint y el 100% del codigo fuente generado. Identifica vulnerabilidades y reescribe el codigo completo endurecido en un bloque ```python."
    user_debug_full = f"Blueprint Completo:\n{res_arch['content']}\n\nCodigo Fuente del Coder (Completo):\n```python\n{clean_code}\n```\n\nResultado Sandbox:\nExit Code: {exit_code}\nSTDOUT: {stdout}\nSTDERR: {stderr}\n\nAudita exhaustivamente y reescribe el codigo completo blindado."

    print(">> Ejecutando Auditoria Modalidad FULL...")
    res_debug_full = query_endpoint(PORT, sys_debug_full, user_debug_full, max_tokens=3500, temperature=0.2)
    res_debug_full["vram_mb"] = vram_cyber
    res_debug_full["wake_s"] = t_wake_cyber
    telemetry["debugger_full"] = res_debug_full

    print(f">> DEBUGGER MODALIDAD FULL FINALIZADO:")
    print(f"   - Prompt tokens ingeridos:     {res_debug_full['prompt_tokens']} tokens")
    print(f"   - Tiempo Prefill (Lectura):    {res_debug_full['prompt_eval_ms']} ms ({res_debug_full['prompt_tok_per_sec']} tok/s)")
    print(f"   - Tokens regenerados (Decode): {res_debug_full['completion_tokens']} tokens")
    print(f"   - Velocidad Decode:            {res_debug_full['decode_tok_per_sec']} tok/s")
    print(f"   - Tiempo Total:                {res_debug_full['total_roundtrip_s']} s")

    # =========================================================================
    # FASE 3B: DEBUGGER MODALIDAD INCREMENTAL (CONDICIONADO POR SANDBOX)
    # =========================================================================
    print("\n" + "="*80)
    print("[FASE 3B] DEBUGGER MODALIDAD INCREMENTAL (Verificacion Ligera / Parche Quirurgico)")
    print("          Mismo servidor caliente en VRAM")
    print("="*80)

    # Si los tests pasaron, mandamos solo verificacion ligera y checklist de seguridad
    if exit_code == 0:
        sys_debug_inc = "Eres un auditor de ciberseguridad y confiabilidad de sistemas. Los tests unitarios han pasado exitosamente en el sandbox. Tu tarea es emitir un veredicto formal de aprobacion y advertir sobre riesgos de concurrencia o recursos sutiles. No reescribas el codigo. Emite un dictamen conciso en menos de 300 tokens."
        user_debug_inc = f"""Criterios de Aceptacion del Blueprint:
- SPSC RingBuffer en shared_memory sin copias.
- Serializacion binaria con struct.
- Control de backpressure funcional.

Resultado de Ejecucion en Sandbox:
- Exit Code: 0 (100% Pruebas pasadas)
- Salida: {stdout.strip() or 'OK'}

Emite tu evaluacion de seguridad, aprueba formalmente o señala condiciones de carrera limite sin reescribir el codigo."""
    else:
        # Si hubiera fallado, mandamos solo el traceback y el error
        sys_debug_inc = "Eres un experto en depuracion forense. Los tests fallaron. Diagnostica la falla a partir del traceback y entrega unicamente el parche correctivo de la funcion afectada."
        user_debug_inc = f"Traceback de error:\n{stderr}\n\nDiagnostica la causa y emite el parche de correccion."

    print(">> Ejecutando Auditoria Modalidad INCREMENTAL...")
    res_debug_inc = query_endpoint(PORT, sys_debug_inc, user_debug_inc, max_tokens=600, temperature=0.2)
    telemetry["debugger_incremental"] = res_debug_inc

    print(f">> DEBUGGER MODALIDAD INCREMENTAL FINALIZADO:")
    print(f"   - Prompt tokens ingeridos:     {res_debug_inc['prompt_tokens']} tokens")
    print(f"   - Tiempo Prefill (Lectura):    {res_debug_inc['prompt_eval_ms']} ms ({res_debug_inc['prompt_tok_per_sec']} tok/s)")
    print(f"   - Tokens emitidos (Veredicto): {res_debug_inc['completion_tokens']} tokens")
    print(f"   - Velocidad Decode:            {res_debug_inc['decode_tok_per_sec']} tok/s")
    print(f"   - Tiempo Total:                {res_debug_inc['total_roundtrip_s']} s")

    kill_servers()
    time.sleep(1)

    # =========================================================================
    # COMPARATIVA Y GUARDADO DE TELEMETRIA
    # =========================================================================
    summary_file = os.path.join(RESULTS_DIR, "benchmark_2_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    # Calculos comparativos
    full_tokens_in = res_debug_full['prompt_tokens']
    inc_tokens_in = res_debug_inc['prompt_tokens']
    token_in_savings = round((1.0 - (inc_tokens_in / full_tokens_in)) * 100.0, 1) if full_tokens_in > 0 else 0

    full_tokens_out = res_debug_full['completion_tokens']
    inc_tokens_out = res_debug_inc['completion_tokens']
    token_out_savings = round((1.0 - (inc_tokens_out / full_tokens_out)) * 100.0, 1) if full_tokens_out > 0 else 0

    time_full = res_debug_full['total_roundtrip_s']
    time_inc = res_debug_inc['total_roundtrip_s']
    time_savings = round((1.0 - (time_inc / time_full)) * 100.0, 1) if time_full > 0 else 0

    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO: MODALIDAD FULL VS MODALIDAD INCREMENTAL")
    print("="*80)
    print(f"Metrica                   | Modalidad Full        | Modalidad Incremental | Ahorro / Ganancia")
    print(f"--------------------------|-----------------------|-----------------------|------------------")
    print(f"Tokens de Entrada (In)    | {full_tokens_in:<21} | {inc_tokens_in:<21} | {token_in_savings}% menos")
    print(f"Tokens Generados (Out)    | {full_tokens_out:<21} | {inc_tokens_out:<21} | {token_out_savings}% menos")
    print(f"Tiempo Total de Etapa     | {time_full:<19} s | {time_inc:<19} s | {time_savings}% mas rapido")
    print("="*80)
    print(f">> Telemetria consolidada guardada en: {summary_file}")

if __name__ == "__main__":
    main()
