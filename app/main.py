# -*- coding: utf-8 -*-
"""
main.py - ORQUESTRAÇÃO: imagem -> OCR -> features -> modelo -> predição

Fluxo:

    imagem (URL / base64 / caminho / bytes)
        |  ler_imagem.ler_imagem()                 PaddleOCR: texto de cada passada
        v
    passes (dict)
        |  extrair_features.extrair_features_dict() 16 features a partir do texto
        v
    matriz X (1 linha x N colunas)
        |  modelo.predict()
        v
    resultado (dict simples, pronto para virar JSON na API)

Uso em Python:
    from main import prever
    resultado = prever("https://site.com/cnh.jpg")      # ou caminho / base64 / bytes

Uso pela linha de comando:
    python main.py caminho_ou_url_ou_base64
"""
import json
import logging
import os
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__:
    from . import extrair_features as ef
    from .ler_imagem import carregar_modelos_ocr, ler_imagem
else:
    import extrair_features as ef
    from ler_imagem import carregar_modelos_ocr, ler_imagem

log = logging.getLogger(__name__)

# ======================================================================
# CONFIGURAÇÕES (EDITE AQUI)
# ======================================================================
CAMINHO_MODELO = os.getenv(
    "CAMINHO_MODELO",
    str(Path(__file__).resolve().with_name("modelo.joblib")),
)

MAPA_CLASSES = {0: "cnh_digital", 1: "cnh_fisica", 2: "invalido"}

# Se o modelo NÃO guarda os nomes das colunas, informe aqui as colunas usadas no treino,
# na ordem (por ex.: list(X_train.columns)). Com None, usa as 16 features na ordem padrão.
COLUNAS_MODELO = None

# Se a imagem vier de um cliente externo, deixe False: a API não deve abrir arquivos do servidor.
PERMITIR_ARQUIVO_LOCAL = os.getenv("PERMITIR_ARQUIVO_LOCAL", "1") == "1"

# ======================================================================
# CARREGAMENTO (acontece uma única vez)
# ======================================================================
_ESTADO = {}
_TRAVA = threading.Lock()


def para_python(valor):
    """np.int64 / np.float32 -> tipos nativos do Python (necessário para virar JSON)."""
    return valor.item() if hasattr(valor, "item") else valor


def carregar_tudo():
    """Carrega o modelo e o PaddleOCR. Idempotente: chame na inicialização da API."""
    with _TRAVA:
        if _ESTADO:
            return _ESTADO

        objeto = joblib.load(CAMINHO_MODELO)
        extras = {}
        if isinstance(objeto, dict):          # o .joblib foi salvo como dicionário
            if "modelo" not in objeto:
                raise KeyError(f"O .joblib é um dicionário sem a chave 'modelo'. Chaves: {list(objeto)}")
            modelo = objeto["modelo"]
            extras = {k: v for k, v in objeto.items() if k != "modelo"}
        else:
            modelo = objeto
        log.info("Modelo carregado: %s | chaves extras no joblib: %s", type(modelo).__name__, list(extras))

        # Quais colunas o modelo usa, e em que ordem
        nomes_modelo = getattr(modelo, "feature_names_in_", None)
        if nomes_modelo is not None:
            colunas = list(nomes_modelo)
        elif COLUNAS_MODELO is not None:
            colunas = list(COLUNAS_MODELO)
        else:
            colunas = list(ef.NOMES_FEATURES)

        desconhecidas = [c for c in colunas if c not in ef.NOMES_FEATURES]
        if desconhecidas:
            raise ValueError(f"O modelo usa colunas que a extração não gera: {desconhecidas}\n"
                             f"A extração gera: {ef.NOMES_FEATURES}")

        n_esperado = getattr(modelo, "n_features_in_", None)
        if n_esperado is not None and n_esperado != len(colunas):
            raise ValueError(f"O modelo espera {n_esperado} features, mas a configuração gera {len(colunas)}. "
                             "Preencha COLUNAS_MODELO com as colunas exatas (e na ordem) do treino.")

        mapa = MAPA_CLASSES if MAPA_CLASSES is not None else extras.get("mapa_classes")

        carregar_modelos_ocr()    # evita que a 1ª requisição pague o custo de carregar o Paddle

        _ESTADO.update(modelo=modelo, colunas=colunas,
                       usa_nomes=nomes_modelo is not None, mapa=mapa)
        return _ESTADO


# ======================================================================
# ETAPAS
# ======================================================================
def montar_matriz(features, colunas, usa_nomes):
    """dict de features -> matriz (1 amostra x N features), só com as colunas do modelo, na ordem certa."""
    vetor = [features[c] for c in colunas]
    if usa_nomes:                                   # modelo treinado com DataFrame
        return pd.DataFrame([vetor], columns=colunas)
    return np.asarray(vetor, dtype=float).reshape(1, -1)


def traduzir_predicao(valor, mapa):
    valor = para_python(valor)
    return mapa.get(valor, valor) if mapa else valor


# ======================================================================
# FUNÇÃO PRINCIPAL
# ======================================================================
def prever(fonte, debug=False):
    """
    fonte: URL, string base64, caminho de arquivo ou bytes.
    Devolve um dicionário simples (serializável em JSON):
        {"predicao": ..., "predicao_bruta": ..., "probabilidades": {...}, "features": {...}, "ocr": {...}}
    Com debug=True inclui também o texto lido pelo OCR.
    """
    e = carregar_tudo()
    modelo, colunas, mapa = e["modelo"], e["colunas"], e["mapa"]

    # 1) imagem -> texto
    passes, meta_ocr = ler_imagem(fonte, permitir_arquivo_local=PERMITIR_ARQUIVO_LOCAL)

    # 2) texto -> features -> matriz
    features = ef.extrair_features_dict(passes)
    X = montar_matriz(features, colunas, e["usa_nomes"])

    # 3) predição
    predicao = modelo.predict(X)[0]
    resultado = {
        "predicao": traduzir_predicao(predicao, mapa),
        "predicao_bruta": para_python(predicao),
    }

    # 4) probabilidades (nem todo modelo tem)
    if hasattr(modelo, "predict_proba"):
        try:
            proba = modelo.predict_proba(X)[0]
            resultado["probabilidades"] = {
                str(traduzir_predicao(c, mapa)): round(float(p), 4)
                for c, p in zip(modelo.classes_, proba)
            }
        except Exception:
            pass

    resultado["features"] = {c: para_python(features[c]) for c in colunas}
    resultado["ocr"] = {"rotacao": meta_ocr["rotacao"],
                        "n_linhas": meta_ocr["n_palavras"],
                        "conf_media": round(meta_ocr["conf_media"], 1)}
    if debug:
        resultado["texto_ocr"] = passes
        resultado["diagnostico"] = ef.diagnostico(passes)
    return resultado


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("uso: python main.py <caminho | url | base64> [--debug]")
    print(json.dumps(prever(sys.argv[1], debug="--debug" in sys.argv),
                     ensure_ascii=False, indent=2))
