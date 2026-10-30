import os
import sys
import time

# Enable Rust-based parallel multi-connection transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download, get_token

repo_id = "agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF"
filename = "Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"
local_dir = r"C:\Users\bj_iv\Downloads\LLM\agentionai\Qwen3.8-Flash-Next-Gyro-Coder-GGUF"

os.makedirs(local_dir, exist_ok=True)

token = get_token()
print(f"Iniciando descarga acelerada con HF-Transfer (Rust)...")
print(f"Modelo: {repo_id}")
print(f"Archivo: {filename}")
print(f"Destino: {local_dir}")
print(f"Token autenticado: {bool(token)}")
sys.stdout.flush()

t0 = time.time()
try:
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token
    )
    t1 = time.time()
    size_gb = os.path.getsize(file_path) / (1024**3)
    print(f"\nDescarga completada exitosamente en {t1 - t0:.1f} s ({(size_gb * 1024) / (t1 - t0):.2f} MB/s)")
    print(f"Archivo guardado en: {file_path}")
    print(f"Tamano: {size_gb:.2f} GiB")
except Exception as e:
    print(f"\nError durante la descarga: {e}", file=sys.stderr)
    sys.exit(1)
