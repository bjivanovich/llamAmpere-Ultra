import urllib.request
import json
import time

system_prompt = """You are an expert Principal Software Engineer and elite coding agent.
Your core principles:
1. Code Quality: Write production-grade, modular, self-contained, and fully typed code.
2. Architecture: Prioritize O(1) or optimal time/space complexity, explicit error handling, and thread/async safety.
3. Reasoning: Keep internal thoughts concise and focused on edge cases, data structures, and invariants.
4. Output Style: Provide the complete, working implementation without placeholders (no 'TODO' or 'insert logic here'). Follow language idioms (Python 3.12+, TypeScript 5+, Rust 2024)."""

user_prompt = """Write a complete production-ready distributed Rate Limiter middleware for FastAPI in Python using Redis and the Token Bucket algorithm.

Specifications:
1. `RedisTokenBucketLimiter`:
   - Implemented using atomic Lua scripts to guarantee thread and race condition safety across multiple replica instances.
   - Configurable parameters: `capacity` (burst capacity) and `refill_rate` (tokens added per second).
   - Dynamic key derivation based on authenticated `user_id`, client IP address, or endpoint path.
   - Returns remaining tokens, reset timestamp, and allowance status on every check.
2. ASGI Middleware:
   - Intercepts incoming HTTP requests before reaching route handlers.
   - Enforces rate limits with zero latency overhead (connection pooling via `redis.asyncio`).
   - Injects standard HTTP headers on responses: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`.
   - Emits HTTP 429 Too Many Requests with a `Retry-After` header when limits are exceeded.
3. Resilience and Fallback:
   - Circuit-breaker mechanism: If Redis becomes unreachable, fail-open gracefully with structured logging instead of dropping user requests."""

chat_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
Edge cases: clock skew across Redis replicas, burst capacity, Redis connection loss, graceful fail-open.
Atomic Lua implementation with EVALSHA.
</think>
"""

payload = {
    "prompt": chat_prompt,
    "n_predict": 1200,
    "temperature": 0.5,
    "top_p": 0.95,
    "stream": False
}

req = urllib.request.Request(
    "http://127.0.0.1:8080/completion",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Enviando solicitud de generacion directa...")
t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
t1 = time.time()

code = res.get("content", "")
with open("rate_limiter_output.py", "w", encoding="utf-8") as f:
    f.write(code)

timings = res.get("timings", {})
print("=" * 50)
print(f"Generacion completada en {t1 - t0:.2f} s")
print(f"Tokens generados: {res.get('tokens_predicted')}")
print(f"Velocidad de generacion: {timings.get('predicted_per_second', 0):.2f} tokens/s")
print(f"Drafts aceptados: {timings.get('draft_n_accepted', 0)} / {timings.get('draft_n', 0)}")
print("=" * 50)
