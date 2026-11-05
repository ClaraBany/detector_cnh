---
title: Classificador de CNH
emoji: 🪪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Classificador de CNH (física / digital / inválido)

API que recebe a imagem de um documento e devolve a classe prevista.

Pipeline: **PaddleOCR (7 passadas)** → **16 features** → **modelo (joblib)**.

## Estrutura

```
app/
  main.py        FastAPI (endpoints, validação do upload)
  ocr.py         Etapa 1: OCR multipasse (igual ao ocr_paddle_v2.ipynb)
  features.py    Etapa 2: features (igual ao extrair_features.ipynb)
  modelo.py      Carrega o .joblib e prevê
models/
  modelo_cnh.joblib   <- coloque seu modelo aqui
scripts/         warmup (build), testar_api, comparar_features
Dockerfile
requirements.txt
```

## Antes de subir

1. Copie o modelo para `models/modelo_cnh.joblib` (ou mude `MODEL_PATH`).
2. Em `requirements.txt`, deixe `scikit-learn` (e o que mais o modelo usar) **na mesma versão do treino**.
   No notebook de treino: `!pip list | grep -i -E "scikit-learn|joblib|pandas|numpy|xgboost|lightgbm"`.
3. Fixe também a versão do `paddleocr` usada no Colab (`!pip show paddleocr`).
4. Se o `.joblib` tiver mais de 10 MB, use Git LFS (o Hugging Face exige):
   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   ```

## Rodar localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

Ou com Docker:

```bash
docker build -t cnh-api . && docker run -p 7860:7860 cnh-api
```

## Usar a API

```bash
curl -X POST "http://localhost:7860/predict" -F "file=@minha_cnh.jpg"
# com debug (features, rotação, texto do OCR):
curl -X POST "http://localhost:7860/predict?debug=true" -F "file=@minha_cnh.jpg"
```

Resposta:

```json
{
  "label": "cnh_digital",
  "confianca": 0.97,
  "probabilidades": {"cnh_digital": 0.97, "cnh_fisica": 0.02, "invalido": 0.01},
  "tempo_ms": 18432
}
```

Documentação interativa em `/docs`. Health check em `/health`.

## Deploy no Hugging Face Spaces

1. Crie um Space em huggingface.co/new-space com **SDK = Docker** (Blank). Hardware: CPU basic (gratuito).
2. Conecte ao GitHub, uma das opções:
   - **GitHub Action** (já incluída em `.github/workflows/sync_to_hf.yml`): crie o secret `HF_TOKEN`
     (token com permissão *write*) e a variável `HF_SPACE` (`usuario/nome-do-space`). Todo push na `main` atualiza o Space.
   - **Manual**: `git remote add space https://huggingface.co/spaces/USUARIO/NOME && git push space main`
3. (Recomendado) Em *Settings → Variables and secrets* do Space, crie o secret `API_KEY`.
   Com ele definido, toda chamada a `/predict` precisa do header `X-API-Key`.
4. O primeiro build demora (instala o Paddle e baixa os modelos de OCR). Depois a URL fica em
   `https://USUARIO-NOME.hf.space`.

## Variáveis de ambiente

| Variável | Padrão | Para que serve |
|---|---|---|
| `MODEL_PATH` | `models/modelo_cnh.joblib` | caminho do modelo |
| `API_KEY` | (vazio) | se definida, exige `X-API-Key` |
| `MAX_UPLOAD_MB` | `10` | tamanho máximo do upload |
| `OCR_DEVICE` | `cpu` | `gpu:0` se usar Space com GPU |
| `OCR_ENABLE_MKLDNN` | (não definida) | `0` desliga o MKLDNN se o Paddle der erro de inferência em CPU |

## Pontos de atenção

- **CPU vs GPU:** o treino rodou em GPU; a API roda em CPU. Em geral o texto sai igual, mas pode haver
  pequenas diferenças numéricas. Confira com `python -m scripts.comparar_features imagem.jpg features_todas.csv "<id_documento>"`.
- **Latência:** cada imagem faz 7 passadas de OCR + classificador de orientação. No CPU gratuito (2 vCPU)
  conte com dezenas de segundos por requisição. As requisições são processadas uma por vez.
- **Space gratuito dorme** após ~48 h sem uso; a primeira chamada depois disso demora para acordar.
- Não altere `app/ocr.py` nem `app/features.py` sem retreinar o modelo: as features precisam sair
  exatamente como no treino.

# detector_cnh