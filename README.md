---
title: Classificador de CNH
emoji: 🪪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# API Classificadora de CNH

API HTTP para classificar imagens de documentos como CNH digital, CNH física ou inválido.

## URL base

- Local: `http://localhost:8080`
- Produção: substitua pelo endereço do serviço, por exemplo `https://SUA_URL`

A API não exige autenticação própria. Em produção, restrinja o acesso usando os controles
de acesso da plataforma onde ela está hospedada. Use HTTPS para enviar documentos.

## Endpoints

### `GET /health`

Verifica se o modelo e o OCR terminaram de inicializar.

**Sucesso — `200 OK`**

```json
{"status": "ok"}
```

Durante a inicialização ou se o carregamento falhar, retorna `503 Service Unavailable`.
Consulte os logs do serviço se o erro persistir.

### `POST /predict`

Envia uma imagem para classificação. Envie **exatamente uma** das opções abaixo:

| Campo | Formato aceito | Descrição |
|---|---|---|
| `file` | `multipart/form-data` | Arquivo da imagem |
| `url` | `multipart/form-data` ou JSON | URL HTTP/HTTPS acessível pelo servidor |
| `imagem_base64` | `multipart/form-data` ou JSON | Base64 puro ou Data URI (`data:image/...;base64,...`) |

O limite padrão para arquivo enviado e Base64 decodificado é **10 MB**. Pode ser alterado
pela variável `MAX_UPLOAD_MB`. O limite não é aplicado ao conteúdo baixado de uma URL.

#### Enviar arquivo

```bash
curl -X POST "https://SUA_URL/predict" \
  -F "file=@minha_cnh.jpg"
```

#### Enviar URL

Como formulário:

```bash
curl -X POST "https://SUA_URL/predict" \
  -F "url=https://servidor.com/imagens/cnh.jpg"
```

Como JSON:

```bash
curl -X POST "https://SUA_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://servidor.com/imagens/cnh.jpg\"}"
```

A URL precisa usar `http://` ou `https://` e estar acessível pelo servidor da API.

#### Enviar Base64

Como JSON:

```bash
curl -X POST "https://SUA_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"imagem_base64\":\"<BASE64_DA_IMAGEM>\"}"
```

O campo também aceita uma Data URI completa, por exemplo:
`data:image/png;base64,iVBORw0KGgo...`.

#### Modo de depuração

Adicione `?debug=true` à URL para incluir OCR e diagnóstico:

```bash
curl -X POST "https://SUA_URL/predict?debug=true" \
  -F "file=@minha_cnh.jpg"
```

## Resposta

**Sucesso — `200 OK`**

```json
{
  "doc": "cnh_digital",
  "confidence": 0.97
}
```

- `doc`: classe prevista pelo modelo. Valores atuais: `cnh_digital`, `cnh_fisica` ou `invalido`.
- `confidence`: probabilidade retornada pelo modelo para a classe prevista, entre `0` e `1`.
  O modelo atual arredonda para quatro casas decimais. Se ele não disponibilizar probabilidade,
  o campo será `null`.
- Com `debug=true`, a resposta também inclui `debug.texto_ocr` e `debug.diagnostico`.

## Erros

| HTTP | Significado |
|---|---|
| `400 Bad Request` | Entrada inválida, Base64 inválido, arquivo vazio ou imagem não reconhecida |
| `413 Payload Too Large` | Arquivo ou Base64 acima do limite configurado |
| `415 Unsupported Media Type` | Formato diferente de JSON ou formulário |
| `503 Service Unavailable` | Modelo/OCR inicializando ou falha ao carregar |

## Documentação interativa

Com o serviço ativo, acesse `https://SUA_URL/docs` para testar os endpoints pela interface
do FastAPI.