"""Roda no build do Docker: baixa os modelos do Paddle para dentro da imagem (cold start rápido)."""
from app import ocr

if __name__ == "__main__":
    ocr.carregar()
    print("Modelos do PaddleOCR baixados e carregados.")
