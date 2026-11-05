FROM python:3.11-slim

# libs de sistema exigidas pelo OpenCV e pelo Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces roda como usuário 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/home/user/.paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OCR_DEVICE=cpu

WORKDIR /home/user

# Copia os arquivos mantendo a estrutura de pastas na raiz do usuário
COPY --chown=user . /home/user/

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Baixa os modelos do PaddleOCR durante o build
RUN python -m scripts.warmup

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]