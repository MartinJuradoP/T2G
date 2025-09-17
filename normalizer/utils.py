# -*- coding: utf-8 -*-
"""
normalizer/utils.py
===================

Utilidades usadas por la etapa de Normalización:
- Heurísticas de tipo por superficie (dirección, ORG por sufijos/acrónimos, título, ciudad/país, disciplinas).
- Normalizadores básicos: fechas (ISO-8601 con precisión), dinero (valor + moneda),
  email, URL, organización (core + sufijos + clave), persona (given/family + clave).
- Helpers: stable_id (hash estable por doc/tipo/clave) y clean_attrs (quita None).

Estos normalizadores no pretenden cubrir todos los casos; son prácticos, sin dependencias externas
y suficientemente robustos para la mayoría de documentos corporativos.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, List
import re
import hashlib
import unicodedata
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------------------------
# Heurísticas de superficie
# ---------------------------------------------------------------------------

ADDRESS_HINTS = {
    "calle", "av", "av.", "avenida", "blvd", "boulevard", "col", "col.",
    "colonia", "no", "no.", "#", "piso", "cp", "c.p.", "mz", "manzana", "lt", "lote",
}

ORG_TOKENS_STRONG = {
    "corp", "inc", "ltd", "gmbh", "srl", "srl.", "sa", "s.a.", "s.a. de c.v.",
    "sa de cv", "s.a de c.v", "s.a de cv", "co.", "compañía", "compania", "company",
    "univ", "universidad", "instituto", "institute", "bank", "banco",
    "ltda", "cooperative", "cooperativa", "asociación", "association",
}

TITLES_ES = {
    "director", "directora", "directora de datos", "jefe", "jefa",
    "gerente", "ingeniero", "ingeniera", "analista",
    "científico de datos", "cientifica de datos",
    "chief data officer", "cdo", "head of data",
}

FIELDS_ES_EN = {
    "matemáticas", "matematicas", "analytics", "statistics", "estadística",
    "computer science", "informática", "data science",
}

COUNTRY_CITY_LITE = {
    "españa", "méxico", "mexico", "mexico city", "cdmx",
    "madrid", "barcelona", "monterrey", "guadalajara",
    "buenos aires", "bogotá", "bogota", "lima", "santiago", "quito",
}

_acronym_re = re.compile(r"^[A-ZÁÉÍÓÚÜÑ]{2,6}$")   # acrónimo 2–6 mayúsculas (ej. UNAM, MIT)
_num_token_re = re.compile(r"\d+")


def _contains_any(text: str, vocab: Iterable[str]) -> bool:
    t = text.lower()
    return any(tok in t for tok in vocab)


def is_all_caps_acronym(text: str) -> bool:
    """True si el texto completo es un acrónimo corto en mayúsculas (no dígitos)."""
    s = text.strip()
    return bool(_acronym_re.match(s)) and not s.isdigit()


def is_address_like(text: str) -> bool:
    """
    Heurística de dirección: presencia de dígitos + palabras típicas de dirección.
    Ej.: "Calle Falsa 123", "Av. Reforma #25", "Col. Roma Norte 55".
    """
    tl = text.lower()
    has_num = bool(_num_token_re.search(tl))
    return has_num and _contains_any(tl, ADDRESS_HINTS)


def is_org_like(text: str) -> bool:
    """
    Señales fuertes de ORG: sufijos societarios, tokens como 'universidad', 'banco', etc.;
    o acrónimo corto en mayúsculas (ej. 'UNAM', 'MIT').
    """
    tl = text.lower()
    return _contains_any(tl, ORG_TOKENS_STRONG) or is_all_caps_acronym(text)


def is_title_like(text: str) -> bool:
    """Título común en ES/EN mapeable a taxonomía simple."""
    return text.strip().lower() in TITLES_ES


def is_field_of_study(text: str) -> bool:
    """Disciplinas/áreas (no deben mapearse a LOC si vienen solas)."""
    return text.strip().lower() in FIELDS_ES_EN


def is_country_or_city_lite(text: str) -> bool:
    """Lista ligera de países/ciudades para corrección rápida (sin gazetteer completo)."""
    return text.strip().lower() in COUNTRIES_OR_CITIES


# Alias (por si ya usaste el nombre en el normalizer)
COUNTRIES_OR_CITIES = COUNTRY_CITY_LITE


# ---------------------------------------------------------------------------
# Normalizadores simples de valores
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _slugify(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# --- Date -------------------------------------------------------------------

# Patrones simples para fechas comunes ES/EN
_date_ymd = re.compile(r"(?P<y>\d{4})[-/\.](?P<m>\d{1,2})[-/\.](?P<d>\d{1,2})")
_date_dmy = re.compile(r"(?P<d>\d{1,2})[-/\.](?P<m>\d{1,2})[-/\.](?P<y>\d{4})")
_date_ym  = re.compile(r"(?P<y>\d{4})[-/\.](?P<m>\d{1,2})(?![-/\.]\d)")
_date_y   = re.compile(r"\b(?P<y>\d{4})\b")

# meses abreviados y largos en ES/EN (min set)
_MONTHS = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "agosto": 8, "ago": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "septiembre": 9, "sep.": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "diciembre": 12, "dic": 12,
}

# "12 de Ene 2022", "Jan 12, 2022", etc.
_date_textual = re.compile(
    r"(?P<d>\d{1,2})\s*(de\s+)?(?P<m>[A-Za-zÁÉÍÓÚÜÑ\.]+)\s*,?\s*(?P<y>\d{4})",
    flags=re.IGNORECASE
)


def normalize_date(text: str, locale: str = "es") -> Tuple[Optional[str], Optional[str]]:
    """
    Normaliza fechas a ISO-8601 parcial:
        - YYYY-MM-DD
        - YYYY-MM
        - YYYY
    Devuelve (value_iso, precision: 'day'|'month'|'year'|None).
    """
    s = text.strip()

    # 1) YYYY-MM-DD / YYYY/MM/DD
    m = _date_ymd.search(s)
    if m:
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
        return f"{y:04d}-{mo:02d}-{d:02d}", "day"

    # 2) DD/MM/YYYY (común ES)
    m = _date_dmy.search(s)
    if m:
        d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
        return f"{y:04d}-{mo:02d}-{d:02d}", "day"

    # 3) YYYY-MM (parcial)
    m = _date_ym.search(s)
    if m:
        y, mo = int(m.group("y")), int(m.group("m"))
        return f"{y:04d}-{mo:02d}", "month"

    # 4) "12 Ene 2022" / "Jan 12, 2022"
    m = _date_textual.search(s)
    if m:
        d = int(m.group("d"))
        mm = (_strip_accents(m.group("m")).lower()).strip(".")
        y = int(m.group("y"))
        mo = _MONTHS.get(mm)
        if mo:
            return f"{y:04d}-{mo:02d}-{d:02d}", "day"

    # 5) Año suelto
    m = _date_y.search(s)
    if m:
        y = int(m.group("y"))
        return f"{y:04d}", "year"

    return None, None


# --- Money -------------------------------------------------------------------

_currency_symbols = {
    "$": None,     # puede ser MXN/USD, decidimos con default
    "US$": "USD", "USD": "USD",
    "MXN": "MXN", "MX$": "MXN", "MEX$": "MXN",
    "€": "EUR", "EUR": "EUR",
}

_thousand_sep = re.compile(r"[,\u00A0\u202F]")
_decimal_comma = re.compile(r"\d,\d{1,2}$")  # detecta decimal con coma al final (2 dígitos)

def _parse_number_like(s: str) -> Optional[float]:
    # elimina separadores de miles comunes
    s2 = _thousand_sep.sub("", s)
    # si parece formato europeo con coma decimal, reemplaza coma por punto
    if _decimal_comma.search(s2):
        s2 = s2.replace(".", "").replace(",", ".")
    try:
        return float(s2)
    except Exception:
        return None

def normalize_money(text: str, default_currency: str = "MXN") -> Tuple[Optional[float], Optional[str], str]:
    """
    Devuelve (value: float|None, currency: str|None, currency_source: 'symbol'|'default'|'unknown')
    """
    s = text.strip()
    cur_found: Optional[str] = None
    src = "unknown"

    # Detecta prefijo/símbolo simple
    for sym, code in _currency_symbols.items():
        if s.startswith(sym) or sym in s:
            cur_found = code
            src = "symbol"
            break

    # Extrae números
    nums = re.findall(r"[\d\.,\u00A0\u202F]+", s)
    val: Optional[float] = None
    if nums:
        # intenta parsear el más largo (probablemente el monto)
        nums.sort(key=len, reverse=True)
        val = _parse_number_like(nums[0])

    if cur_found is None:
        # si había '$' pero no resolvimos, usa default
        if "$" in s and src == "symbol":
            cur_found = default_currency
        elif val is not None:
            cur_found = default_currency
            src = "default"

    return val, cur_found, src


# --- Email -------------------------------------------------------------------

_email_re = re.compile(r"\b([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")

def normalize_email(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Devuelve (email_norm, user, domain) o None si no matchea.
    """
    m = _email_re.search(text)
    if not m:
        return None
    user = m.group(1).lower()
    domain = m.group(2).lower()
    return f"{user}@{domain}", user, domain


# --- URL ---------------------------------------------------------------------

def normalize_url(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Normaliza una URL:
    - fuerza esquema si falta (http)
    - lowercase host
    - elimina slash final redundante
    - no toca el querystring
    Devuelve (url_norm, parts) o None si no parece URL.
    """
    s = text.strip()
    if not re.search(r"https?://", s):
        s = "http://" + s
    try:
        u = urlparse(s)
    except Exception:
        return None
    if not u.netloc:
        return None

    host = u.netloc.lower()
    path = u.path or ""
    if path.endswith("/") and path != "/":
        path = path[:-1]

    rebuilt = urlunparse((u.scheme, host, path, u.params, u.query, u.fragment))
    parts = {
        "scheme": u.scheme,
        "host": host,
        "path": path,
        "query": u.query,
        "fragment": u.fragment,
    }
    return rebuilt, parts


# --- Organización ------------------------------------------------------------

def normalize_org(text: str, suffixes: List[str]) -> Tuple[str, List[str], str]:
    """
    - Extrae core + lista de sufijos societarios detectados (conservando el texto original).
    - Genera org_key slugificada para merge/dedupe.
    """
    raw = text.strip()
    tl = _strip_accents(raw.lower())
    found: List[str] = []

    # busca sufijos conocidos (case-insensitive; simple contains)
    for suf in suffixes:
        suf_norm = _strip_accents(suf.lower())
        if suf_norm and suf_norm in tl:
            found.append(suf)

    # quita sufijos al final si coinciden claramente (heurística simple)
    core = raw
    # ordena sufijos por longitud para quitar primero los largos
    for suf in sorted(found, key=len, reverse=True):
        pat = re.escape(suf) + r"\s*$"
        core = re.sub(pat, "", core, flags=re.IGNORECASE).strip(",. \t")

    org_key = _slugify(core)
    return core.lower(), found, org_key


# --- Persona -----------------------------------------------------------------

def _split_person(text: str) -> Tuple[str, str]:
    """
    Heurística simple: 2–3 tokens → nombre + apellido(s).
    Si solo hay 1 token, given=token, family="".
    """
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    if not toks:
        return "", ""
    if len(toks) == 1:
        return toks[0], ""
    if len(toks) == 2:
        return toks[0], toks[1]
    # 3 o más: junta el resto como apellido compuesto
    return toks[0], " ".join(toks[1:])

def normalize_person(text: str) -> Tuple[str, str, str]:
    """
    Devuelve (given_name, family_name, person_key)
    person_key = slug de given+family sin acentos.
    """
    given, family = _split_person(text)
    key = _slugify(f"{given} {family}".strip())
    return given, family, key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_id(doc_id: str, etype: str, key: str) -> str:
    """
    Genera un ID estable (hash corto) a partir de (doc_id, etype, key).
    """
    h = hashlib.sha1(f"{doc_id}::{etype}::{key}".encode("utf-8")).hexdigest()[:10]
    return f"ENT-{h}"


def clean_attrs(d: Dict[str, object]) -> Dict[str, object]:
    """Elimina las claves con valor None para JSONs más limpios."""
    return {k: v for k, v in d.items() if v is not None}
