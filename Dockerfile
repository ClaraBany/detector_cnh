FROM python:3.12

# libs de sistema exigidas pelo OpenCV e pelo Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Executa como usuário sem privilégios.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/home/user/.paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OCR_DEVICE=gpu:0 \
    CAMINHO_MODELO=/home/user/app/modelo.joblib

WORKDIR /home/user

# Copia os arquivos mantendo a estrutura de pastas na raiz do usuário
COPY --chown=user . /home/user/

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]