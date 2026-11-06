# 1. Altera a imagem base para uma que inclua o CUDA 12.6 Runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Evita perguntas interativas durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instala o Python 3.12 e as dependências do sistema necessárias para OpenCV e Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3-pip \
        python3.12-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Define o Python 3.12 recém-instalado como o padrão para os comandos 'python' e 'pip'
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Executa como usuário sem privilégios (mantendo as boas práticas de segurança)
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

# Copia os arquivos mantendo a estrutura de pastas
COPY --chown=user . /home/user/

# Garante que o pip esteja atualizado e instala as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
