$env:GGML_CUDA_GRAPH_OPT = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"

$modelPath = "C:\Users\bj_iv\Downloads\LLM\agentionai\Qwen3.8-Flash-Next-Gyro-Coder-GGUF\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Iniciando Gyro-Coder TQ2_0 (Arquitecto / Depurador)" -ForegroundColor Green
Write-Host " Hardware: RTX 3090 (24GB VRAM) - 100% Offload en GPU" -ForegroundColor Yellow
Write-Host " Contexto: 96K (98,304 tokens) | KV Cache: Q8_0" -ForegroundColor Magenta
Write-Host " Velocidad Sostenida Verificada: 53.54 tok/s | Prefill: 219 tok/s" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& "C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe" `
  -m $modelPath `
  -ngl 999 `
  -c 98304 `
  -ctk q8_0 `
  -ctv q8_0 `
  -b 2048 `
  -ub 512 `
  -t 8 `
  -tb 8 `
  -fa on `
  --ctx-checkpoints 0 `
  --parallel 1 `
  --prio 3 `
  --moe-profile python `
  --temp 0.6 `
  --top-p 0.95 `
  --min-p 0.05 `
  --alias "Qwen3.8-Flash-Next-Gyro-Coder-Baseline" `
  --host 127.0.0.1 `
  --port 8085
