# -*- coding: utf-8 -*-
"""
ler_imagem.py - ETAPA 1 da API: IMAGEM -> TEXTO (OCR com PaddleOCR)

Versão "de produção" do notebook ocr_paddle_v2.ipynb. A lógica e os parâmetros do OCR são
os MESMOS usados para gerar os CSVs de treino (mesmas passadas, mesmos limiares, mesma
correção de orientação). Só saiu o que era específico do treino: Colab/Drive, leitura de
pastas, cache em disco e escrita de CSV.

Uso:
    from ler_imagem import ler_imagem
    passes, meta = ler_imagem("https://site.com/cnh.jpg")   # ou caminho, base64, bytes

    passes -> dict com o texto de cada passada:
              principal, binaria, esparso, cabecalho, rot90, rot270, mrz
    meta   -> dict com rotacao, n_palavras, n_confiaveis, conf_media

ATENÇÃO: não altere os parâmetros da seção CONFIGURAÇÃO sem refazer o treino. O modelo
aprendeu com o texto produzido por ESTA configuração.
"""
import base64
import io
import logging
import os
import re
import threading

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# ======================================================================
# CONFIGURAÇÃO (igual à do notebook de treino)
# ======================================================================
CORRIGIR_ORIENTACAO = True    # corrige 0/90/180/270 com o classificador de orientação do Paddle
LARGURA_ALVO = 1600           # só amplia imagens cujo maior lado é menor que isso...
ESCALA_MAX = 2.0              # ...e nunca mais que isso

OCR_LANG = "pt"
# Dispositivo: "gpu:0" ou "cpu". Se não for definido (variável de ambiente OCR_DEVICE),
# usa a GPU quando o Paddle enxergar uma, senão cai para CPU (necessário em muito servidor).
OCR_DEVICE = os.getenv("OCR_DEVICE")

REC_SCORE_PIPELINE = 0.0
CONF_MIN = 0.50
CONF_MIN_CABECALHO = 0.30

DET_LADO_MAX = LARGURA_ALVO
DET_TIPO = "max"

ORI_MODELO = "PP-LCNet_x1_0_doc_ori"
ORI_SCORE_MIN = 0.60
ORI_LADO = 800
USAR_TEXTLINE_ORIENT = False
PASSADAS = {"binaria": True, "esparso": True, "cabecalho": True, "rot": True, "mrz": True}
WL_MRZ = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
PARAMS_ESPARSO = dict(text_det_thresh=0.15, text_det_box_thresh=0.35)

CHAVES_PASSADAS = ("principal", "binaria", "esparso", "cabecalho", "rot90", "rot270", "mrz")


# ======================================================================
# CARREGAR IMAGEM (URL, base64, caminho ou bytes) -> PIL RGB
# ======================================================================
def _eh_url(texto):
    return texto.lower().startswith(("http://", "https://"))


def carregar_imagem(fonte, permitir_arquivo_local=True):
    """
    Recebe URL, string base64 (com ou sem 'data:image/...;base64,'), caminho de arquivo
    ou bytes. Devolve um PIL.Image em RGB, já com a rotação do EXIF aplicada (igual ao
    que foi feito no treino).

    Em produção, passe permitir_arquivo_local=False para a API não ler arquivos do
    servidor a pedido de quem chama.
    """
    if isinstance(fonte, Image.Image):
        return ImageOps.exif_transpose(fonte).convert("RGB")

    if isinstance(fonte, (bytes, bytearray)):
        dados = bytes(fonte)

    elif isinstance(fonte, str):
        fonte = fonte.strip()
        if _eh_url(fonte):
            resp = requests.get(fonte, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            dados = resp.content
        elif (permitir_arquivo_local and not fonte.startswith("data:")
              and len(fonte) < 1024 and os.path.isfile(fonte)):
            with open(fonte, "rb") as f:
                dados = f.read()
        else:
            if fonte.startswith("data:"):
                fonte = fonte.split(",", 1)[1]
            fonte = re.sub(r"\s+", "", fonte)
            try:
                dados = base64.b64decode(fonte, validate=True)
            except Exception:
                raise ValueError("Não consegui interpretar a entrada. Envie uma URL, "
                                 "uma imagem em base64 ou um caminho de arquivo válido.")
    else:
        raise TypeError("A fonte da imagem deve ser texto (URL, base64, caminho) ou bytes.")

    try:
        img = Image.open(io.BytesIO(dados))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise ValueError(f"O conteúdo recebido não é uma imagem válida: {e}")


# ======================================================================
# PADDLEOCR (modelos criados uma única vez)
# ======================================================================
_OCR = None
_ORI = None
_DEVICE = None
_LOCK = threading.RLock()     # também serializa a inferência (o Paddle não é thread-safe)


def _dispositivo():
    global _DEVICE
    if _DEVICE is None:
        if OCR_DEVICE:
            _DEVICE = OCR_DEVICE
        else:
            try:
                import paddle
                tem_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            except Exception:
                tem_gpu = False
            _DEVICE = "gpu:0" if tem_gpu else "cpu"
        log.info("PaddleOCR usando o dispositivo: %s", _DEVICE)
    return _DEVICE


def _ocr():
    global _OCR
    with _LOCK:
        if _OCR is None:
            from paddleocr import PaddleOCR
            _OCR = PaddleOCR(lang=OCR_LANG, device=_dispositivo(),
                             use_doc_orientation_classify=False,   # a orientação é corrigida antes
                             use_doc_unwarping=False,
                             use_textline_orientation=USAR_TEXTLINE_ORIENT)
            if CORRIGIR_ORIENTACAO:
                _orientador()
        return _OCR


def _orientador():
    global _ORI
    with _LOCK:
        if _ORI is None:
            from paddleocr import DocImgOrientationClassification
            _ORI = DocImgOrientationClassification(model_name=ORI_MODELO, device=_dispositivo())
        return _ORI


def carregar_modelos_ocr():
    """Chame uma vez na inicialização da API (evita que a 1ª requisição fique lenta)."""
    _ocr()


def _bgr(im):
    """PIL (L ou RGB) -> ndarray BGR 3 canais, que é o que o Paddle espera."""
    return np.array(im.convert("RGB"))[:, :, ::-1].copy()


def _paddle(im, **kw):
    """Roda o OCR. Devolve lista de (poligono Nx2, texto, confiança 0-1)."""
    params = dict(text_rec_score_thresh=REC_SCORE_PIPELINE,
                  text_det_limit_side_len=DET_LADO_MAX, text_det_limit_type=DET_TIPO)
    params.update(kw)
    try:
        resultados = _ocr().predict(_bgr(im), **params)
    except Exception as e:
        log.error("[ERRO PaddleOCR] %s", e)
        return []
    itens = []
    for r in resultados:
        for poly, txt, sc in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
            itens.append((np.asarray(poly, dtype=float), str(txt), float(sc)))
    return itens


def _linhas(itens, conf_min=None):
    """Filtra por confiança e junta os trechos da mesma linha visual (esquerda -> direita)."""
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
    return {"n_palavras": len(confs),
            "n_confiaveis": sum(c >= CONF_MIN for c in confs),
            "conf_media": 100 * float(np.mean(confs)) if confs else 0.0}


def _linhas_mrz(itens):
    """Guarda só as linhas longas, sem espaços, quase só A-Z, 0-9 e '<' (zona legível por máquina)."""
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


# ======================================================================
# ORIENTAÇÃO E PRÉ-PROCESSAMENTO
# ======================================================================
def _angulo(im):
    """Pergunta ao classificador do Paddle quanto o documento está girado. Devolve (graus, confiança)."""
    try:
        r = _orientador().predict(_bgr(im), batch_size=1)[0]
        r = r.get("res", r)
        return int(r["label_names"][0]), float(r["scores"][0])
    except Exception as e:
        log.error("[ERRO orientação] %s", e)
        return 0, 0.0


def corrigir_orientacao(img_rgb):
    """Devolve (imagem, ângulo aplicado). Na dúvida, não gira."""
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


def _variantes(img_rgb):
    """CLAHE (contraste local) e binarização adaptativa."""
    cinza = np.array(img_rgb.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)
    binaria = cv2.adaptiveThreshold(cv2.GaussianBlur(clahe, (3, 3), 0), 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(clahe), Image.fromarray(binaria)


# ======================================================================
# OCR MULTIPASSE (função principal)
# ======================================================================
def ocr_multipasse(img):
    """Recebe um PIL.Image RGB e devolve (passes, meta)."""
    ang = 0
    if CORRIGIR_ORIENTACAO:
        img, ang = corrigir_orientacao(img)
    img = _escalar(img)
    clahe, binaria = _variantes(img)
    passes = {k: "" for k in CHAVES_PASSADAS}

    itens = _paddle(clahe)                       # passada principal (também alimenta o MRZ)
    passes["principal"] = _linhas(itens)
    st = _stats(itens)

    if PASSADAS.get("binaria"):
        passes["binaria"] = _linhas(_paddle(binaria))
    if PASSADAS.get("esparso"):
        passes["esparso"] = _linhas(_paddle(clahe, **PARAMS_ESPARSO))
    if PASSADAS.get("cabecalho"):
        topo = clahe.crop((0, 0, clahe.width, int(clahe.height * 0.22)))
        passes["cabecalho"] = _linhas(_paddle(topo), conf_min=CONF_MIN_CABECALHO)
    if PASSADAS.get("rot"):
        passes["rot90"] = _linhas(_paddle(clahe.rotate(90, expand=True)))
        passes["rot270"] = _linhas(_paddle(clahe.rotate(270, expand=True)))
    if PASSADAS.get("mrz"):
        passes["mrz"] = _linhas_mrz(itens)

    meta = {"rotacao": int(ang), **st}
    return passes, meta


def ler_imagem(fonte, permitir_arquivo_local=True):
    """
    fonte: URL, base64, caminho de arquivo, bytes ou PIL.Image.
    Devolve (passes, meta): o texto de cada passada do OCR e as estatísticas do OCR.
    """
    img = carregar_imagem(fonte, permitir_arquivo_local=permitir_arquivo_local)
    with _LOCK:      # uma imagem por vez no Paddle
        return ocr_multipasse(img)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 2:
        sys.exit("uso: python ler_imagem.py <caminho | url | base64>")
    p, m = ler_imagem(sys.argv[1])
    print(json.dumps({"meta": m, "passes": p}, ensure_ascii=False, indent=2))
