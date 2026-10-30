$env:GGML_CUDA_GRAPH_OPT = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"
$env:GGML_CUDA_MOE_CACHE_SERIAL_FILL = "0"
$env:GGML_CUDA_MOE_CACHE_DEDICATED_MMV = "1"
$env:GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS = "-1"
$env:GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB = "64"
$env:GGML_CUDA_MOE_CACHE_INSERTS = "64"
$env:GGML_CUDA_MOE_CACHE_ADMIT_AFTER = "1"
$env:GGML_CUDA_MOE_CACHE_THROTTLE = "1"
$env:GGML_CUDA_MOE_CACHE_QUEUE = "4096"
$env:GGML_CUDA_MOE_CACHE_QUEUE_MB = "8192"
$env:GGML_CUDA_MOE_CACHE_STATS = "50"

$modelPath = "C:\Users\bj_iv\Downloads\LLM\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"

& "D:\llamAmpere\build-sm86\bin\Release\llama-server.exe" `
  -m $modelPath `
  -ngl 99 `
  -cmoe `
  --fit off `
  --load-mode none `
  -c 112640 `
  -b 2048 `
  -ub 512 `
  -t 8 `
  -tb 8 `
  -fa on `
  -ctk q8_0 `
  -ctv q8_0 `
  --kv-unified `
  --prio 3 `
  --prio-batch 3 `
  --moe-cache 11500 `
  --moe-cache-mib 11500 `
  --reasoning on `
  --reasoning-effort low `
  --reasoning-format deepseek `
  --no-reasoning-preserve `
  -n -1 `
  --parallel 1 `
  --cache-prompt `
  --cache-ram 2048 `
  --temp 0.6 `
  --top-p 0.95 `
  --min-p 0.05 `
  --alias "Qwen3.8-Flash-Next-Ultra-110K" `
  --host 127.0.0.1 `
  --port 8080
