$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_GRAPH_OPT = "1"
$env:GGML_CUDA_FA_I8QK = "1"
$env:LLAMA_SPEC_CHAIN = "1"
$env:LLAMA_SPEC_CHAIN_SAMPLED = "1"
$env:LLAMA_SPEC_PQ = "1"
$env:LLAMA_SPEC_ADAPT_COST = "1"

$modelPath = "C:/Users/bj_iv/Downloads/LLM/bjivanovich/Qwen3.8-27B-Uncensored-Cyber-ATX-MTP-GGUF/Qwen3.8-27B-Uncensored-Cyber-ATX-i1-Q5_K_M.gguf"
$vocabMap = "D:\llamAmpere\docs\mtp-vocab\atx_65536.txt"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Iniciando Swift-1.5 Cyber-ATX 27B Q5_K_M (Coder Especializado)" -ForegroundColor Green
Write-Host " Hardware: RTX 3090 (24GB VRAM) - 100% Offload en GPU" -ForegroundColor Yellow
Write-Host " Contexto: 160K (163,840 tokens) | KV: turbo5 / turbo4 unificado" -ForegroundColor Magenta
Write-Host " Velocidad Sostenida: 46.73 tok/s | Pico Maximo: 74.16 tok/s" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& "C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe" `
  -m $modelPath `
  -ngl 99 `
  -c 163840 `
  -b 4096 `
  -ub 1024 `
  -t 20 `
  -tb 8 `
  -fa on `
  -ctk turbo5 `
  -ctv turbo4 `
  --kv-unified `
  --prio 3 `
  --parallel 1 `
  --jinja `
  --fit off `
  --reasoning_effort low `
  --cache-prompt `
  --cache-ram 16384 `
  --ctx-checkpoints 24 `
  --checkpoint-min-step 10240 `
  --spec-draft-type-k q8_0 `
  --spec-draft-type-v q8_0 `
  --spec-draft-vocab-map $vocabMap `
  --reasoning-format none `
  --temp 0.2 `
  --top-p 0.90 `
  --top-k 40 `
  --min-p 0.05 `
  --repeat-penalty 1.08 `
  --repeat-last-n 256 `
  --alias "Swift-1.5-Qwen3.8-27B-Uncensored-Q5_K_M" `
  --host 127.0.0.1 `
  --port 8080
