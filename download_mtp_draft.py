import os
import sys
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, get_token

repo_id = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
filename = "mtp-Qwen3.8-Flash-Next-draft.gguf"
local_dir = r"C:\Users\bj_iv\Downloads\LLM\agentionai\Qwen3.8-Flash-Next-Gyro-GGUF"

os.makedirs(local_dir, exist_ok=True)
token = get_token()

print(f"Iniciando descarga de MTP Draft Head (2.51 GiB)...")
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
    print(f"MTP Draft Head descargado con exito: {size_gb:.2f} GiB en {t1 - t0:.1f} s")
except Exception as e:
    print(f"Error descargando MTP: {e}", file=sys.stderr)
