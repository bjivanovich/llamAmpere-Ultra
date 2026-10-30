import sys
import time
import json
import urllib.request

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

def run_benchmark(prompt, max_tokens=1000):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": max_tokens
    }

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print("\n" + "="*70)
    print(" [llamAmpere Ultra] Benchmark de Streaming en Tiempo Real")
    print(" Endpoint: " + SERVER_URL)
    print("="*70 + "\n")

    t_start = time.time()
    t_first = None
    token_count = 0
    recent_times = []
    peak_toks = 0.0

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                raw = line.decode("utf-8").strip()
                if not raw or raw.startswith(":"):
                    continue
                if raw == "data: [DONE]":
                    break
                if raw.startswith("data: "):
                    try:
                        data = json.loads(raw[6:])
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        chunk = delta.get("content") or delta.get("reasoning_content") or ""
                        if chunk:
                            now = time.time()
                            if t_first is None:
                                t_first = now
                                ttft_ms = (t_first - t_start) * 1000.0
                                print(f"-> TTFT (Time-To-First-Token): {ttft_ms:.1f} ms\n")

                            token_count += 1
                            recent_times.append(now)
                            # Sliding window 1.5 seconds
                            while recent_times and (now - recent_times[0]) > 1.5:
                                recent_times.pop(0)

                            dt_win = now - recent_times[0]
                            instant_toks = (len(recent_times) / dt_win) if (len(recent_times) > 1 and dt_win > 0.05) else 0.0
                            if instant_toks > peak_toks:
                                peak_toks = instant_toks

                            total_dt = now - t_first
                            avg_toks = (token_count / total_dt) if total_dt > 0 else 0.0

                            # Dynamic ASCII meter
                            meter_len = int(min(instant_toks, 60.0) / 2.0)
                            meter = "#" * meter_len + "-" * (30 - meter_len)

                            status_line = (
                                f"\r[{meter}] {instant_toks:5.1f} t/s | "
                                f"Avg: {avg_toks:4.1f} t/s | "
                                f"Peak: {peak_toks:4.1f} t/s | "
                                f"Tokens: {token_count:5d} | "
                                f"Time: {total_dt:5.1f}s"
                            )
                            sys.stdout.write(status_line)
                            sys.stdout.flush()
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\n\nGeneracion cancelada por el usuario.")
    except Exception as e:
        print(f"\nError de conexion: {e}")

    total_time = (time.time() - t_first) if t_first else 0.0
    final_avg = (token_count / total_time) if total_time > 0 else 0.0

    print("\n\n" + "-"*70)
    print(f" Resumen Final:")
    print(f" Total Tokens:     {token_count}")
    print(f" Tiempo Total:     {total_time:.2f} s")
    print(f" Velocidad Media:  {final_avg:.2f} tok/s")
    print(f" Pico Maximo:      {peak_toks:.2f} tok/s")
    print("-"*70 + "\n")

if __name__ == "__main__":
    test_prompt = (
        "Implementa una clase SpatialQuadTree en Python con metodos de insercion, "
        "busqueda por radio de colision, subdivision recursiva de nodos y documentacion completa."
    )
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])

    run_benchmark(test_prompt)
