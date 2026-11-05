"""Confere se o pipeline em CPU reproduz as features geradas no Colab (GPU).

Uso: python -m scripts.comparar_features imagem.jpg features_todas.csv "<id_documento da imagem no CSV>"
"""
import sys

import pandas as pd
from PIL import Image

from app import ocr
from app.features import COLUNAS_FEATURES, calcular_features

img_path, csv_path, id_doc = sys.argv[1:4]
esperado = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
linha = esperado[esperado["id_documento"] == id_doc]
if linha.empty:
    sys.exit(f"id_documento '{id_doc}' não encontrado no CSV.")
linha = linha.iloc[0]

passes, meta = ocr.extrair_passes(Image.open(img_path))
obtido = calcular_features(passes)

difs = [(c, int(linha[c]), obtido[c]) for c in COLUNAS_FEATURES if int(linha[c]) != obtido[c]]
print("rotação aplicada:", meta["rotacao_aplicada"])
print("IDÊNTICO ao Colab" if not difs else "DIFERENÇAS (feature, colab, cpu):")
for d in difs:
    print("  ", d)
