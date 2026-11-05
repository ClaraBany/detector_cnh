import io
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import ocr
from .features import calcular_features
from .modelo import ModeloCNH

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cnh.api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/modelo_cnh.joblib")
API_KEY = os.getenv("API_KEY")                       # se definida, exige o header X-API-Key
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000                  # proteção contra "decompression bomb"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = ModeloCNH(MODEL_PATH)
    log.info("Modelo carregado. Classes: %s", app.state.modelo.classes)
    ocr.carregar()
    log.info("PaddleOCR pronto (device=%s)", ocr.OCR_DEVICE)
    yield


app = FastAPI(title="Classificador de CNH", version="1.0.0", lifespan=lifespan)


def checar_chave(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="API key inválida ou ausente (header X-API-Key).")


class Predicao(BaseModel):
    label: str
    confianca: Optional[float] = None
    probabilidades: Optional[dict] = None
    tempo_ms: int
    debug: Optional[dict] = None


@app.get("/")
def raiz():
    return {"servico": "classificador-cnh", "docs": "/docs", "predict": "POST /predict (multipart, campo 'file')"}


@app.get("/health")
def health():
    return {"status": "ok", "classes": app.state.modelo.classes}


@app.post("/predict", response_model=Predicao, dependencies=[Depends(checar_chave)])
def predict(file: UploadFile = File(...),
            debug: bool = Query(False, description="Inclui features, rotação e texto do OCR na resposta")):
    t0 = time.perf_counter()

    dados = file.file.read(MAX_BYTES + 1)
    if not dados:
        raise HTTPException(400, "Arquivo vazio.")
    if len(dados) > MAX_BYTES:
        raise HTTPException(413, f"Arquivo maior que {MAX_BYTES // 1024 // 1024} MB.")

    try:
        img = Image.open(io.BytesIO(dados))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "Arquivo não é uma imagem válida (use jpg, png, webp, bmp ou tiff).")

    passes, meta = ocr.extrair_passes(img)
    feats = calcular_features(passes)
    resultado = app.state.modelo.prever(feats)

    out = {**resultado, "tempo_ms": int((time.perf_counter() - t0) * 1000)}
    if debug:
        out["debug"] = {"features": feats, "ocr": meta, "texto_principal": passes["principal"]}
    log.info("predict label=%s conf=%s tempo=%dms", out["label"], out["confianca"], out["tempo_ms"])
    return out
