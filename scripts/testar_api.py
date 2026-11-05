"""Uso: python -m scripts.testar_api caminho/imagem.jpg [URL] [--debug]"""
import os
import sys

import requests

args = [a for a in sys.argv[1:] if not a.startswith("--")]
debug = "--debug" in sys.argv
img = args[0]
url = args[1] if len(args) > 1 else "http://localhost:7860"
headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}

with open(img, "rb") as f:
    r = requests.post(f"{url}/predict", params={"debug": str(debug).lower()}, headers=headers,
                      files={"file": (os.path.basename(img), f)}, timeout=300)
print(r.status_code)
print(r.json())
