import os
import sys
import time
import urllib.request
import zipfile
import io

dest_dir = r"C:\Users\bj_iv\Desktop\LlamaForge"
os.makedirs(dest_dir, exist_ok=True)
url = "https://api.github.com/repos/dadwritestech/LlamaForge/zipball/v0.16.4"

print(f"Descargando LlamaForge v0.16.4 desde {url}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
t0 = time.time()
try:
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    t1 = time.time()
    print(f"Descarga completada: {len(content) / (1024*1024):.2f} MB en {t1 - t0:.1f} s")
    
    print("Descomprimiendo en", dest_dir, "...")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # Zipball contains a root folder like dadwritestech-LlamaForge-hash/
        root_prefix = zf.namelist()[0].split('/')[0] + '/'
        for member in zf.namelist():
            if member.startswith(root_prefix):
                rel_path = member[len(root_prefix):]
                if not rel_path:
                    continue
                target_path = os.path.join(dest_dir, rel_path.replace('/', os.sep))
                if member.endswith('/'):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, 'wb') as f:
                        f.write(zf.read(member))
    print("Instalacion de LlamaForge extraida con exito!")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
