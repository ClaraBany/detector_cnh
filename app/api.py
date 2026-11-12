"""API HTTP do classificador de CNH para execução no Cloud Run."""

import base64
import binascii
import logging
import os
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

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


def _verificar_pronto(request: Request) -> None:
    status = request.app.state.status
    if status != "ready":
        detail = (
            "Falha ao inicializar o modelo/OCR; consulte os logs."
            if status == "error"
            else "Modelo e OCR ainda estão inicializando."
        )
        raise HTTPException(status_code=503, detail=detail)


def _decodificar_base64(valor: str) -> bytes:
    valor = valor.strip()
    if valor.startswith("data:"):
        cabecalho, separador, valor = valor.partition(",")
        if not separador or ";base64" not in cabecalho.lower():
            raise HTTPException(
                status_code=400,
                detail="Data URI inválida; esperado data:image/...;base64,...",
            )

    valor = "".join(valor.split())
    limite_codificado = 4 * ((MAX_UPLOAD_BYTES + 2) // 3)
    if len(valor) > limite_codificado:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem maior que {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    try:
        dados = base64.b64decode(valor, validate=True)
    except (binascii.Error, ValueError) as erro:
        raise HTTPException(
            status_code=400,
            detail="O campo imagem_base64 não contém Base64 válido.",
        ) from erro

    if not dados:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(dados) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem maior que {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )
    return dados


async def _fonte_da_requisicao(request: Request) -> bytes | str:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        campos = await request.form()
    elif content_type == "application/json":
        try:
            campos = await request.json()
        except (ValueError, UnicodeDecodeError) as erro:
            raise HTTPException(status_code=400, detail="JSON inválido.") from erro
    else:
        raise HTTPException(
            status_code=415,
            detail="Envie multipart/form-data ou JSON com file, url ou imagem_base64.",
        )

    if not hasattr(campos, "get"):
        raise HTTPException(status_code=400, detail="O corpo da requisição deve ser um objeto.")

    arquivo = campos.get("file")
    url = campos.get("url")
    imagem_base64 = campos.get("imagem_base64")
    entradas = [valor for valor in (arquivo, url, imagem_base64) if valor is not None]
    if len(entradas) != 1:
        raise HTTPException(
            status_code=400,
            detail="Envie exatamente uma das opções: file, url ou imagem_base64.",
        )

    if arquivo is not None:
        if not isinstance(arquivo, UploadFile):
            raise HTTPException(status_code=400, detail="O campo file deve ser um arquivo enviado.")
        dados = arquivo.file.read(MAX_UPLOAD_BYTES + 1)
        if not dados:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")
        if len(dados) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Imagem maior que {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
            )
        return dados

    if url is not None:
        if not isinstance(url, str) or not url.strip():
            raise HTTPException(status_code=400, detail="O campo url deve conter uma URL.")
        url = url.strip()
        try:
            partes = urlsplit(url)
        except ValueError as erro:
            raise HTTPException(status_code=400, detail="URL inválida.") from erro
        if partes.scheme not in ("http", "https") or not partes.netloc:
            raise HTTPException(status_code=400, detail="A URL deve começar com http:// ou https://.")
        return url

    if not isinstance(imagem_base64, str):
        raise HTTPException(status_code=400, detail="O campo imagem_base64 deve ser uma string.")
    return _decodificar_base64(imagem_base64)


def _prever(fonte: bytes | str, debug: bool):
    try:
        resultado = pipeline.prever(fonte, debug=debug)
    except (ValueError, TypeError) as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro

    predicao = resultado["predicao"]
    probabilidades = resultado.get("probabilidades", {})
    resposta = {
        "doc": predicao,
        "confidence": probabilidades.get(str(predicao)),
    }
    if debug:
        resposta["debug"] = {
            chave: resultado[chave]
            for chave in ("texto_ocr", "diagnostico")
            if chave in resultado
        }
    return resposta


@app.get("/")
def raiz():
    return {
        "servico": "classificador-cnh",
        "health": "/health",
        "docs": "/docs",
        "predict": "POST /predict (file, url ou imagem_base64; envie apenas uma opção)",
    }


@app.get("/health")
def health(request: Request):
    _verificar_pronto(request)
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    request: Request,
    debug: bool = Query(False, description="Inclui texto OCR e diagnóstico na resposta."),
):
    _verificar_pronto(request)
    fonte = await _fonte_da_requisicao(request)
    return await run_in_threadpool(_prever, fonte, debug)
