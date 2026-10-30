import os
import sys
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, get_token

mtp_targets = [
    {
        "repo_id": "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF",
        "filename": "mtp-Qwen3.8-Flash-Next-draft.gguf",
        "desc": "MTP Draft Head Oficial para Gyro (2.51 GiB)"
    },
    {
        "repo_id": "agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF",
        "filename": "Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf",
        "desc": "MTP Draft Head FP4 Ultra-Rapido (2.28 GiB)"
    },
    {
        "repo_id": "agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF",
        "filename": "Qwen3.8-Flash-Next-MTP-Q8_0.gguf",
        "desc": "MTP Draft Head Q8_0 Completo (3.85 GiB)"
    }
]

local_dir = r"C:\Users\bj_iv\Downloads\LLM\agentionai\mtp"
os.makedirs(local_dir, exist_ok=True)
token = get_token()

print("==========================================================")
print(" Descargando cabezales MTP (Multi-Token Prediction)...")
print(f" Destino: {local_dir}")
print("==========================================================")
sys.stdout.flush()

for target in mtp_targets:
    repo = target["repo_id"]
    fn = target["filename"]
    desc = target["desc"]
    print(f"\n-> Descargando: {desc}")
    print(f"   Origen: {repo} | Archivo: {fn}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        dest_file = hf_hub_download(
            repo_id=repo,
            filename=fn,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        t1 = time.time()
        sz = os.path.getsize(dest_file) / (1024**3)
        print(f"   Completado: {sz:.2f} GiB en {t1 - t0:.1f} s")
    except Exception as e:
        print(f"   Error descargando {fn}: {e}", file=sys.stderr)

print("\n==========================================================")
print(" Todos los cabezales MTP descargados exitosamente!")
print("==========================================================")
