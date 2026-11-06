# -*- coding: utf-8 -*-
"""
extrair_features.py - ETAPA 2 da API: TEXTO DO OCR -> FEATURES

Versão "de produção" do notebook extrair_features.ipynb. As 16 features e a ORDEM delas são
as mesmas usadas no treino (COLUNAS_FEATURES). Não abre imagem, não faz OCR e não usa
pandas: recebe o dicionário `passes` devolvido por ler_imagem.ler_imagem().

Contrato (usado pelo main.py):
    NOMES_FEATURES                 lista com o nome de cada feature, na ordem do vetor
    extrair_features_dict(passes)  -> dict {nome_da_feature: valor}
    extrair_features(passes)       -> np.ndarray 1D com as 16 features, na ordem de NOMES_FEATURES

Para mudar as features, edite só a seção "DEFINIÇÃO DAS FEATURES" (e refaça o treino!).
"""
import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz

# ======================================================================
# CONFIGURAÇÃO
# ======================================================================
LIMIAR_FUZZY = 0.80   # similaridade mínima (0-1) para considerar a frase presente (flag = 1)

# ======================================================================
# DEFINIÇÃO DAS FEATURES (igual ao treino)
# ======================================================================
UF = (r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)")

# Em quais passadas do OCR cada feature procura
GERAL = ("principal", "binaria", "esparso", "cabecalho")
LATERAL = ("rot90", "rot270")      # texto vertical (girado)

PASSADAS_NECESSARIAS = ["principal", "binaria", "esparso", "cabecalho", "rot90", "rot270", "mrz"]

FLAGS_REGEX = [
    # marcadores da CNH digital
    ("flag_senatran",             r"SENATRAN", GERAL),
    ("flag_serpro",               r"SERPRO", GERAL),
    ("flag_assinatura_digital",   r"CERTIFICADO DIGITAL|ASSINADO DIGITALMENTE|\b2200\s*[-–]\s*2\b", GERAL),
    ("flag_qrcode",               r"QR\s*-?\s*CODE", GERAL),
    ("flag_legenda_multilingue",  r"NAME AND SURNAME|NOMBRE Y APELLIDOS|FIRST DRIVER LICENSE", GERAL),
    # marcadores da CNH física
    ("flag_proibido_plastificar", r"PROIBID|PLASTIF", GERAL + LATERAL),
    ("flag_assinatura_emissor",   r"ASSINATURA\s+DO\s+EMISSOR", GERAL),
    # formato dos dados
    ("flag_nascimento_formato",   r"\d{2}/\d{2}/\d{4}\s*,\s*[A-Z ]{2,40}?\s*,\s*" + UF + r"\b", GERAL),
    ("flag_registro_11_digitos",  r"(?<!\d)\d{11}(?!\d)", GERAL),
]

FRASES_FUZZY = [
    ("flag_titulo_cnh",           "CARTEIRA NACIONAL DE HABILITACAO"),
    ("flag_republica_federativa", "REPUBLICA FEDERATIVA DO BRASIL"),
]

ROTULOS_CAMPOS = [
    r"NOME\s*E\s*SOBRENOME",
    r"DATA\s*,?\s*LOCAL\s*E\s*UF",
    r"DATA\s*EMISSAO",
    r"\bVALIDADE\b",
    r"DOC\.?\s*IDENTIDADE",
    r"\bCPF\b",
    r"\bREGISTRO\b",
    r"CAT\.?\s*HAB",
    r"OBSERVACOES|OBSERVACAO",
    r"FILIACAO",
    r"NACIONALIDADE",
    r"\bLOCAL\b(?!\s*E\s*UF)",
]

# Lista oficial das features do modelo, NA ORDEM do treino
COLUNAS_FEATURES = (
    [n for n, _, _ in FLAGS_REGEX]
    + [n for n, _ in FRASES_FUZZY]
    + ["qtd_linhas_mrz", "qtd_rotulos_campos",
       "flag_cpf_formato", "flag_cpf_checksum", "flag_codigo_lateral_11"]
)
NOMES_FEATURES = list(COLUNAS_FEATURES)


# ======================================================================
# FUNÇÕES AUXILIARES
# ======================================================================
def normalizar(t):
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).upper()
    return re.sub(r"\s+", " ", t).strip()


def cpf_valido(cpf):
    d = [int(x) for x in re.sub(r"\D", "", cpf)]
    if len(d) != 11 or len(set(d)) == 1:
        return False
    for n in (9, 10):
        dv = (sum(d[i] * (n + 1 - i) for i in range(n)) * 10 % 11) % 10
        if dv != d[n]:
            return False
    return True


def _completar(passes):
    """Garante que as 7 passadas existam (ausente vira texto vazio)."""
    return {p: (passes.get(p) or "") for p in PASSADAS_NECESSARIAS}


# ======================================================================
# CÁLCULO DAS FEATURES
# ======================================================================
def extrair_features_dict(passes):
    """passes: dict com o texto de cada passada do OCR. Devolve {feature: valor}."""
    row = _completar(passes)
    f = {}
    cache = {}

    def T(passadas):
        if passadas not in cache:
            cache[passadas] = normalizar("\n".join(row[p] for p in passadas))
        return cache[passadas]

    # 1) presença de texto/formato (regex)
    for nome, padrao, passadas in FLAGS_REGEX:
        f[nome] = int(re.search(padrao, T(passadas)) is not None)

    # 2) similaridade fuzzy, binarizada pelo limiar
    for nome, frase in FRASES_FUZZY:
        f[nome] = int(fuzz.partial_ratio(frase, T(GERAL)) / 100 >= LIMIAR_FUZZY)

    # 3) MRZ (linhas longas, sem espaços, quase só A-Z, 0-9 e '<')
    linhas_mrz = [l.replace(" ", "") for l in row["mrz"].splitlines()]
    f["qtd_linhas_mrz"] = min(sum(1 for l in linhas_mrz if l.count("<") >= 3 and 20 <= len(l) <= 40), 3)

    # 4) quantos rótulos de campo aparecem
    f["qtd_rotulos_campos"] = sum(1 for p in ROTULOS_CAMPOS if re.search(p, T(GERAL)))

    # 5) CPF: formato e dígitos verificadores
    cpfs = re.findall(r"\d{3}[.,\s]\d{3}[.,\s]\d{3}\s?[-–]\s?\d{2}", T(GERAL))
    f["flag_cpf_formato"] = int(len(cpfs) > 0)
    f["flag_cpf_checksum"] = int(any(cpf_valido(c) for c in cpfs))

    # 6) código lateral (vertical) de 11 dígitos
    lat = [re.sub(r"\s", "", l) for l in (row["rot90"] + "\n" + row["rot270"]).splitlines()]
    f["flag_codigo_lateral_11"] = int(any(re.search(r"(?<!\d)\d{11}(?!\d)", l) for l in lat))

    assert set(f) == set(COLUNAS_FEATURES), "COLUNAS_FEATURES não bate com o que é calculado"
    return f


def extrair_features(passes):
    """Vetor 1D com as 16 features, na ordem de NOMES_FEATURES."""
    f = extrair_features_dict(passes)
    return np.array([f[c] for c in NOMES_FEATURES], dtype=float)


def diagnostico(passes):
    """Só para conferir o OCR (NÃO vai para o modelo)."""
    principal = (passes.get("principal") or "")
    n_tokens = len(re.findall(r"\w+", principal))
    return {
        "n_tokens": n_tokens,
        "flag_texto_vazio": int(n_tokens < 5),
        "texto_ocr": re.sub(r"\s*\n\s*", " | ", principal.strip()),
    }
