import os
import sys
import time
import urllib.request
import tarfile

url = "https://github.com/Niko1221/Strata/releases/download/v0.2.2/strata-engine-0.1.47-windows-x86_64-cuda-cuda12.tar.gz"
dest_dir = r"C:\Users\bj_iv\Desktop\Strata"
os.makedirs(dest_dir, exist_ok=True)
archive_path = os.path.join(dest_dir, "strata-engine.tar.gz")

print(f"Descargando Strata Engine v0.2.2 (CUDA 12/13)...")
print(f"URL: {url}")
print(f"Destino: {archive_path}")

t0 = time.time()
try:
    urllib.request.urlretrieve(url, archive_path)
    t1 = time.time()
    size_mb = os.path.getsize(archive_path) / (1024*1024)
    print(f"Descarga de Strata finalizada: {size_mb:.1f} MB en {t1 - t0:.1f} s")
    
    print("Descomprimiendo archivos...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print(f"Extraccion completada con exito en {dest_dir}!")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
