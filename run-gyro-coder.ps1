$env:GGML_CUDA_GRAPH_OPT = "1"
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"
$env:GGML_CUDA_FA_I8QK = "1"

$modelPath = "C:\Users\bj_iv\Downloads\LLM\agentionai\Qwen3.8-Flash-Next-Gyro-Coder-GGUF\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"
$draftPath = "C:\Users\bj_iv\Downloads\LLM\agentionai\mtp\mtp-Qwen3.8-Flash-Next-draft.gguf"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Iniciando Qwen3.8-Flash-Next Gyro-Coder (65B Pruned MoE)" -ForegroundColor Green
Write-Host " Motor: llamAmpere-Ultra (Desktop)" -ForegroundColor Yellow
Write-Host " Hardware: RTX 3090 (24GB VRAM) - Offload 100% en VRAM" -ForegroundColor Green
Write-Host " Decodificacion Especulativa: MTP Neuronal (Multi-Token)" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Cyan

& "C:\Users\bj_iv\Desktop\llamAmpere-Ultra\bin\Release\llama-server.exe" `
  -m $modelPath `
  -md $draftPath `
  --spec-type draft-mtp `
  --spec-draft-n-max 4 `
  --no-spec-draft-backend-sampling `
  -ngl 999 `
  -fa on `
  --ngram-on-disk `
  --ngram-cache 512 `
  --ngram-io-threads 32 `
  -c 32768 `
  -b 2048 `
  -ub 512 `
  -t 8 `
  -tb 8 `
  -ctk q8_0 `
  -ctv q8_0 `
  --temp 0.6 `
  --top-p 0.95 `
  --min-p 0.05 `
  --reasoning on `
  --reasoning-effort low `
  --reasoning-format deepseek `
  --alias "Qwen3.8-Flash-Next-Gyro-Coder-MTP" `
  --host 127.0.0.1 `
  --port 8080
