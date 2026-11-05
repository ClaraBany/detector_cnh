"""Etapa 1 (ocr_paddle_v2.ipynb): imagem -> texto das 7 passadas, com PaddleOCR.

Mesmos parâmetros do notebook de treino (RECORTAR_CARTAO=False, CORRIGIR_ORIENTACAO=True).
A diferença é só o dispositivo: no Space gratuito roda em CPU (OCR_DEVICE=cpu).
"""
import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("cnh.ocr")

# ---------- configuração (igual ao notebook) ----------
CORRIGIR_ORIENTACAO = True
LARGURA_ALVO = 1600
ESCALA_MAX = 2.0
OCR_LANG = "pt"
OCR_DEVICE = os.getenv("OCR_DEVICE", "CPU")          # "gpu:0" se um dia usar GPU

REC_SCORE_PIPELINE = 0.0
CONF_MIN = 0.50
CONF_MIN_CABECALHO = 0.30
DET_LADO_MAX = LARGURA_ALVO
DET_TIPO = "max"

ORI_MODELO = "PP-LCNet_x1_0_doc_ori"
ORI_SCORE_MIN = 0.60
ORI_LADO = 800
USAR_TEXTLINE_ORIENT = False
WL_MRZ = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
PARAMS_ESPARSO = dict(text_det_thresh=0.15, text_det_box_thresh=0.35)

CHAVES_PASSADAS = ("principal", "binaria", "esparso", "cabecalho", "rot90", "rot270", "mrz")

_OCR = None
_ORI = None
_LOCK = threading.Lock()      # os predictors do Paddle não são thread-safe


# ---------- modelos ----------
def _ocr():
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        extra = {}
        if os.getenv("OCR_ENABLE_MKLDNN") is not None:      # só se precisar desligar (ver README)
            extra["enable_mkldnn"] = os.getenv("OCR_ENABLE_MKLDNN") == "1"
        _OCR = PaddleOCR(lang=OCR_LANG, device=OCR_DEVICE,
                         use_doc_orientation_classify=False,
                         use_doc_unwarping=False,
                         use_textline_orientation=USAR_TEXTLINE_ORIENT,
                         **extra)
        if CORRIGIR_ORIENTACAO:
            _orientador()
    return _OCR


def _orientador():
    global _ORI
    if _ORI is None:
        from paddleocr import DocImgOrientationClassification
        _ORI = DocImgOrientationClassification(model_name=ORI_MODELO, device=OCR_DEVICE)
    return _ORI


def carregar():
    """Carrega (e baixa, se preciso) os modelos. Chamado no startup e no build do Docker."""
    with _LOCK:
        _ocr()


# ---------- OCR ----------
def _bgr(im):
    return np.array(im.convert("RGB"))[:, :, ::-1].copy()


def _paddle(im, **kw):
    params = dict(text_rec_score_thresh=REC_SCORE_PIPELINE,
                  text_det_limit_side_len=DET_LADO_MAX, text_det_limit_type=DET_TIPO)
    params.update(kw)
    try:
        resultados = _ocr().predict(_bgr(im), **params)
    except Exception:
        log.exception("Erro no PaddleOCR")
        return []
    itens = []
    for r in resultados:
        for poly, txt, sc in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
            itens.append((np.asarray(poly, dtype=float), str(txt), float(sc)))
    return itens


def _linhas(itens, conf_min=None):
    conf_min = CONF_MIN if conf_min is None else conf_min
    sel = []
    for poly, txt, sc in itens:
        txt = txt.strip()
        if sc < conf_min or not txt:
            continue
        ys, xs = poly[:, 1], poly[:, 0]
        sel.append((float(ys.mean()), float(xs.min()), float(ys.max() - ys.min()), txt))
    if not sel:
        return ""
    sel.sort()
    alt = float(np.median([i[2] for i in sel])) or 1.0
    linhas, atual, y_ref = [], [], None
    for yc, x, h, t in sel:
        if y_ref is None or abs(yc - y_ref) <= 0.6 * alt:
            atual.append((x, t))
            y_ref = yc if y_ref is None else y_ref
        else:
            linhas.append(atual)
            atual, y_ref = [(x, t)], yc
    linhas.append(atual)
    return "\n".join(" ".join(t for _, t in sorted(l)) for l in linhas)


def _stats(itens):
    confs = [sc for _, t, sc in itens if t.strip()]
    return {"n_linhas": len(confs),
            "n_confiaveis": sum(c >= CONF_MIN for c in confs),
            "conf_media": round(100 * float(np.mean(confs)), 1) if confs else 0.0}


def _linhas_mrz(itens):
    saida = []
    for _, txt, sc in itens:
        if sc < CONF_MIN:
            continue
        t = txt.strip().upper()
        if " " in t or len(t) < 15:
            continue
        limpo = "".join(c for c in t if c in WL_MRZ)
        if len(limpo) >= 0.9 * len(t):
            saida.append(limpo)
    return "\n".join(saida)


# ---------- orientação / pré-processamento ----------
def _angulo(im):
    try:
        r = _orientador().predict(_bgr(im), batch_size=1)[0]
        r = r.get("res", r)
        return int(r["label_names"][0]), float(r["scores"][0])
    except Exception:
        log.exception("Erro na orientação")
        return 0, 0.0


def corrigir_orientacao(img_rgb):
    peq = img_rgb.copy()
    peq.thumbnail((ORI_LADO, ORI_LADO))
    ang, sc = _angulo(peq)
    if ang == 0 or sc < ORI_SCORE_MIN:
        return img_rgb, 0
    melhor = None
    for giro in dict.fromkeys((ang, (360 - ang) % 360)):
        a2, s2 = _angulo(peq.rotate(giro, expand=True))
        if a2 == 0 and s2 >= ORI_SCORE_MIN and (melhor is None or s2 > melhor[1]):
            melhor = (giro, s2)
    if melhor is None:
        return img_rgb, 0
    return img_rgb.rotate(melhor[0], expand=True), melhor[0]


def _escalar(img):
    maior = max(img.size)
    if maior < LARGURA_ALVO:
        k = min(ESCALA_MAX, LARGURA_ALVO / maior)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return img


def _preparar(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    ang = 0
    if CORRIGIR_ORIENTACAO:
        img, ang = corrigir_orientacao(img)
    return _escalar(img), ang


def _variantes(img_rgb):
    cinza = np.array(img_rgb.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)
    binaria = cv2.adaptiveThreshold(cv2.GaussianBlur(clahe, (3, 3), 0), 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(clahe), Image.fromarray(binaria)


# ---------- API pública ----------
def extrair_passes(img: Image.Image):
    """Imagem PIL -> (dict com as 7 passadas, metadados)."""
    with _LOCK:
        img, ang = _preparar(img)
        clahe, binaria = _variantes(img)

        itens = _paddle(clahe)
        passes = {"principal": _linhas(itens)}
        st = _stats(itens)

        passes["binaria"] = _linhas(_paddle(binaria))
        passes["esparso"] = _linhas(_paddle(clahe, **PARAMS_ESPARSO))
        topo = clahe.crop((0, 0, clahe.width, int(clahe.height * 0.22)))
        passes["cabecalho"] = _linhas(_paddle(topo), conf_min=CONF_MIN_CABECALHO)
        passes["rot90"] = _linhas(_paddle(clahe.rotate(90, expand=True)))
        passes["rot270"] = _linhas(_paddle(clahe.rotate(270, expand=True)))
        passes["mrz"] = _linhas_mrz(itens)

    assert set(passes) == set(CHAVES_PASSADAS)
    return passes, {"rotacao_aplicada": int(ang), **st}
