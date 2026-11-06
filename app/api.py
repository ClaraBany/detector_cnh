"""API HTTP do classificador de CNH para execução no Cloud Run."""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

if __package__:
    from . import main as pipeline
else:
    import main as pipeline

log = logging.getLogger("cnh.api")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024


def _carregar_modelo(app: FastAPI) -> None:
    try:
        pipeline.carregar_tudo()
        app.state.status = "ready"
        log.info("API pronta para receber imagens.")
    except Exception:
        app.state.status = "error"
        log.exception("Falha ao carregar o modelo ou o PaddleOCR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.status = "loading"
    threading.Thread(
        target=_carregar_modelo,
        args=(app,),
        name="model-loader",
        daemon=True,
    ).start()
    yield


app = FastAPI(
    title="Classificador de CNH",
    description="Classifica imagens de CNH usando OCR e um modelo de machine learning.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def raiz():
    return {
        "servico": "classificador-cnh",
        "health": "/health",
        "docs": "/docs",
        "predict": "POST /predict (multipart, campo 'file')",
    }


@app.get("/health")
def health(request: Request):
    status = request.app.state.status
    if status != "ready":
        detail = (
            "Falha ao inicializar o modelo/OCR; consulte os logs."
            if status == "error"
            else "Modelo e OCR ainda estão inicializando."
        )
        raise HTTPException(status_code=503, detail=detail)
    return {"status": "ok"}


@app.post("/predict")
def predict(
    request: Request,
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Inclui texto OCR e diagnóstico na resposta."),
):
    status = request.app.state.status
    if status != "ready":
        detail = (
            "Falha ao inicializar o modelo/OCR; consulte os logs."
            if status == "error"
            else "Modelo e OCR ainda estão inicializando."
        )
        raise HTTPException(status_code=503, detail=detail)

    dados = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not dados:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(dados) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem maior que {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    try:
        return pipeline.prever(dados, debug=debug)
    except (ValueError, TypeError) as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro
