import os
import sys
import time
import subprocess
import requests
import json
import threading

SERVER_EXE = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe"
MODEL_PATH = r"D:\models\Qwen3.8-27B-Uncensored-Cyber-ATX-i1-Q5_K_M.gguf"
VOCAB_MAP = r"D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"
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

def main():
    print("=================================================================")
    print("LIVE TEST: Hot VMM Swapping (cuMemUnmap + DMA Wake) on RTX 3090")
    print("Target Model: Cyber-ATX 27B Q5_K_M (18.8 GB)")
    print("=================================================================")

    init_vram = get_vram_mb()
    print(f"Baseline VRAM: {init_vram} MB")

    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    except Exception:
        pass
    time.sleep(1)

    env = os.environ.copy()
    env["LLAMAMPERE_ENABLE_SLEEP"] = "1"
    env["GGML_CUDA_GRAPH_OPT"] = "1"
    env["GGML_Q8_TURBO3_MMA_FUSED"] = "1"
    env["GGML_CUDA_FA_I8QK"] = "1"

    cmd = [
        SERVER_EXE,
        "-m", MODEL_PATH,
        "-ngl", "99",
        "-c", "4096",
        "-ctk", "turbo5",
        "-ctv", "turbo4",
        "-b", "2048",
        "-ub", "512",
        "-fa", "on",
        "--sleep-idle-seconds", "4",
        "--host", "127.0.0.1",
        "--port", str(PORT)
    ]

    if os.path.exists(VOCAB_MAP):
        cmd.extend(["--spec-draft-vocab-map", VOCAB_MAP])

    print("Launching llama-server...")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def reader():
        for line in iter(proc.stdout.readline, ''):
            l = line.strip()
            if l:
                print(f"   [server] {l}", flush=True)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    url_health = f"http://127.0.0.1:{PORT}/health"
    url_chat = f"http://127.0.0.1:{PORT}/v1/chat/completions"

    ready = False
    for attempt in range(120):
        try:
            r = requests.get(url_health, timeout=1)
            if r.status_code == 200:
                ready = True
                print(f"\nServer is ready after {attempt+1}s!")
                break
        except Exception:
            pass
        if attempt % 5 == 0:
            print(f"Waiting for server... ({attempt+1}s)", flush=True)
        time.sleep(1)

    if not ready:
        print("ERROR: Server failed to start within 120s")
        proc.kill()
        return

    load_vram = get_vram_mb()
    print(f"\n[PHASE 1] Model Loaded in VRAM: {load_vram} MB (Model footprint ~{load_vram - init_vram} MB)")

    print("\n[PHASE 2] Executing prompt 1...")
    payload1 = {
        "messages": [{"role": "user", "content": "What is 25 * 4? Give a brief answer."}],
        "max_tokens": 40,
        "temperature": 0.6
    }
    t0 = time.time()
    res1 = requests.post(url_chat, json=payload1, timeout=60).json()
    t1 = time.time()
    print(f"Prompt 1 finished in {t1 - t0:.2f}s:")
    answer1 = res1.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f"   Answer: {answer1}")

    print("\n[PHASE 3] Waiting for idle sleep trigger (4s)...")
    sleep_dropped = False
    for i in range(20):
        time.sleep(1)
        cur_vram = get_vram_mb()
        print(f"   t={i+1}s -> VRAM: {cur_vram} MB")
        if cur_vram < load_vram - 10000:
            print(f"\n   SUCCESS! VRAM dropped from {load_vram} MB to {cur_vram} MB!")
            print(f"   VRAM released: ~{load_vram - cur_vram} MB freed into driver!")
            sleep_dropped = True
            break

    time.sleep(1)

    print("\n[PHASE 4] Executing prompt 2 to trigger HOT VMM DMA WAKE from RAM...")
    payload2 = {
        "messages": [{"role": "user", "content": "What is 100 / 4? Give a brief answer."}],
        "max_tokens": 50,
        "temperature": 0.6
    }
    t_wake_start = time.time()
    res2 = requests.post(url_chat, json=payload2, timeout=60).json()
    t_wake_end = time.time()

    total_wake_latency = t_wake_end - t_wake_start
    print(f"\n[PHASE 5] Verification Results:")
    print(f"   Total Response Latency (Wake + DMA Transfer + Prefill + Decode): {total_wake_latency:.2f}s")
    answer2 = res2.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f"   Answer: {answer2}")

    final_vram = get_vram_mb()
    print(f"   VRAM after wake: {final_vram} MB")

    print("\nTerminating server cleanly...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("Test complete!")

if __name__ == "__main__":
    main()
