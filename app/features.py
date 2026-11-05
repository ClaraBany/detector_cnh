import re
import unicodedata

from rapidfuzz import fuzz

LIMIAR_FUZZY = 0.80

UF = (r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)")

GERAL = ("principal", "binaria", "esparso", "cabecalho")
LATERAL = ("rot90", "rot270")

FLAGS_REGEX = [
    ("flag_senatran",             r"SENATRAN", GERAL),
    ("flag_serpro",               r"SERPRO", GERAL),
    ("flag_assinatura_digital",   r"CERTIFICADO DIGITAL|ASSINADO DIGITALMENTE|\b2200\s*[-–]\s*2\b", GERAL),
    ("flag_qrcode",               r"QR\s*-?\s*CODE", GERAL),
    ("flag_legenda_multilingue",  r"NAME AND SURNAME|NOMBRE Y APELLIDOS|FIRST DRIVER LICENSE", GERAL),
    ("flag_proibido_plastificar", r"PROIBID|PLASTIF", GERAL + LATERAL),
    ("flag_assinatura_emissor",   r"ASSINATURA\s+DO\s+EMISSOR", GERAL),
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

COLUNAS_FEATURES = (
    [n for n, _, _ in FLAGS_REGEX]
    + [n for n, _ in FRASES_FUZZY]
    + ["qtd_linhas_mrz", "qtd_rotulos_campos",
       "flag_cpf_formato", "flag_cpf_checksum", "flag_codigo_lateral_11"]
)


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


def calcular_features(passes: dict) -> dict:
    """passes: dict com as chaves principal, binaria, esparso, cabecalho, rot90, rot270, mrz."""
    f = {}
    cache = {}

    def T(passadas):
        if passadas not in cache:
            cache[passadas] = normalizar("\n".join(passes[p] for p in passadas))
        return cache[passadas]

    for nome, padrao, passadas in FLAGS_REGEX:
        f[nome] = int(re.search(padrao, T(passadas)) is not None)

    for nome, frase in FRASES_FUZZY:
        f[nome] = int(fuzz.partial_ratio(frase, T(GERAL)) / 100 >= LIMIAR_FUZZY)

    linhas_mrz = [l.replace(" ", "") for l in passes["mrz"].splitlines()]
    f["qtd_linhas_mrz"] = min(sum(1 for l in linhas_mrz if l.count("<") >= 3 and 20 <= len(l) <= 40), 3)

    f["qtd_rotulos_campos"] = sum(1 for p in ROTULOS_CAMPOS if re.search(p, T(GERAL)))

    cpfs = re.findall(r"\d{3}[.,\s]\d{3}[.,\s]\d{3}\s?[-–]\s?\d{2}", T(GERAL))
    f["flag_cpf_formato"] = int(len(cpfs) > 0)
    f["flag_cpf_checksum"] = int(any(cpf_valido(c) for c in cpfs))

    lat = [re.sub(r"\s", "", l) for l in (passes["rot90"] + "\n" + passes["rot270"]).splitlines()]
    f["flag_codigo_lateral_11"] = int(any(re.search(r"(?<!\d)\d{11}(?!\d)", l) for l in lat))

    assert set(COLUNAS_FEATURES) == set(f), "features calculadas != COLUNAS_FEATURES"
    return f
