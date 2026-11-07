# 1. Usamos Ubuntu 24.04 que já possui suporte nativo ao Python 3.12 e inclui cuDNN
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# Evita perguntas interativas durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instala o Python 3.12 (nativo do Ubuntu 24) e as libs de OpenCV/Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Mapeia 'python' e 'pip' globais
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

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
# Nota: O Ubuntu 24 exige a flag '--break-system-packages' para instalações pip globais em containers.
RUN pip install --no-cache-dir --upgrade pip --break-system-packages && \
    pip install --no-cache-dir -r requirements.txt --break-system-packages

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
