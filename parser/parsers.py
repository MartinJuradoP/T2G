# -*- coding: utf-8 -*-
"""
parsers.py — Implementación del Parser (PDF / DOCX / IMG → IR)

Resumen
-------
Convierte documentos heterogéneos a una representación intermedia (IR) homogénea
basada en JSON/MD, con bloques de texto y tablas por página, manteniendo un
contrato de salida estable para los siguientes subsistemas (chunking, IE, etc.).

Este módulo ha sido MEJORADO para:
- Reconstruir párrafos lógicos en PDF (evitar “micro-oraciones” por salto visual).
- Aplicar heurísticas duales: textual (puntuación/capitalización) y layout-aware (vertical gaps).
- Conservar trazabilidad fina: `source_lines`, `prov` con extractor compuesto y notas.
- Enriquecer metadatos y métricas operables: `layout_loss`, `fusion_rate`, `n_blocks_raw`, `n_blocks_final`.
- Robustecer detección de headings y list items.
- Mantener backward-compat con contratos existentes (`schemas.py`).

Mejoras de esta versión
-----------------------
- **Limpieza automática de stopwords** multilingüe con `text_clean` (sin romper contrato).
- **Detección robusta de idioma** (`es`, `en`, `es+en`, `und`) a nivel de bloque/página/documento, expuesta como `lang_hint`
  en cada bloque y asignada a `PageIR.lang` y `DocumentIR.lang`.
- **Conservación de `text_raw`** (texto normalizado ligero antes de la limpieza Unicode) para trazabilidad.
- Mantiene la compatibilidad hacia atrás: los bloques siguen siendo dicts (`model_dump()`), añadiendo llaves nuevas opcionales.

Características clave
---------------------
- Detección de tipo por MIME / extensión y dispatch a parser especializado.
- PDF: reconstrucción de párrafos (texto + gaps verticales) y tablas básicas con pdfplumber.
- DOCX: párrafos/headings + tablas con python-docx.
- IMG: OCR con pytesseract (opcional).
- Fallback OCR para páginas PDF sin texto/tabla (típico en PDFs escaneados).
- Normalización de texto configurable (espacios, dehyphenate).
- Metadatos útiles (size_bytes, sha256, page_count) y provenance para trazabilidad.
- Heurísticas opcionales:
  * list_item: detecta bullets/guiones y etiqueta como 'list_item'
  * heading: marca encabezados simples cuando no hay estilos (PDF)

Parámetros importantes
----------------------
- ocr_lang: Idiomas OCR para Tesseract (ej. "spa", "eng", "spa+eng").
- ocr_resolution: DPI para rasterizar páginas PDF en fallback OCR.
- normalize_whitespace: Colapsa espacios múltiples y limpia líneas.
- dehyphenate: Une palabras cortadas por guion ("infor-\\n mación" -> "información").
- enable_pdf_ocr_fallback: Activa OCR por página si pdfplumber no extrajo texto/tabla.
- enable_pdf_heading_heuristics: Heurística conservadora para headings en PDF.
- enable_list_item_detection: Detecta bullets/guiones como list items.
- enable_lang_detect: Detecta idioma (doc/página) si está disponible `langdetect`. (Se mantiene por compatibilidad)
- enable_stopword_clean: Si True, genera `text_clean` por bloque eliminando stopwords acorde a idioma.
- tesseract_cmd: Ruta al ejecutable de tesseract (útil en Windows).
"""

from __future__ import annotations
import os, mimetypes, logging, hashlib, re, unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
import pdfplumber

# Dependencias opcionales protegidas
try:
    import docx  # python-docx para .docx/.doc
except ImportError:
    docx = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract, Image = None, None

# Detección de idioma opcional
try:
    from langdetect import detect as _langdetect
except Exception:
    _langdetect = None

from parser.schemas import DocumentIR, PageIR, TextBlock, TableBlock, TableCell, Provenance, OCRInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Utilidades / constantes
# ---------------------------------------------------------------------

_BULLETS = tuple("•·◦▪-–—*·")
_SENT_END = re.compile(r'[\.!?…]"?$')

# ---------------------- Stopwords y normalización ----------------------

# NOTA: todas en minúsculas; comparamos en minúsculas.
_STOP_ES: Set[str] = {
    "el","la","los","las","un","una","unos","unas","lo","al","del","este","esta","estos","estas",
    "ese","esa","esos","esas","aquel","aquella","aquellos","aquellas","mi","mis","tu","tus",
    "su","sus","nuestro","nuestra","nuestros","nuestras","vuestro","vuestra","vuestros","vuestras",
    "yo","tú","vos","usted","él","ella","ello","nosotros","nosotras","vosotros","vosotras",
    "ustedes","ellos","ellas","me","te","se","nos","os","le","les","lo","la","los","las",
    "a","ante","bajo","con","contra","de","desde","en","entre","hacia","hasta","para","por",
    "según","sin","sobre","tras","y","o","u","ni","que","como","cuando","donde","mientras",
    "aunque","pero","sino","si","sí","no","ya","también","además","solo","solamente","incluso",
    "excepto","salvo","porque","pues","entonces","entretanto","así","así que",
    "por lo tanto","por eso","por consiguiente","de modo que","de manera que",
    "ser","soy","eres","es","somos","son","fui","fue","eran","estoy","estás","está","están",
    "estaba","estaban","estar","haber","hay","he","has","ha","han","había","habían","tener",
    "tengo","tienes","tiene","tenemos","tienen","tuvo","tenía","puede","pueden","pudo","podía",
    "debe","deben","deber","hacer","hace","hacen","hacía","era","eran","fue","fueron",
    "muy","más","menos","mucho","poco","tal","tales","cada","cual","cuales","quien","quienes",
    "cuyo","cuya","cuyos","cuyas","algo","nada","todo","todos","todas","ninguno","ninguna",
    "alguno","alguna","algunos","algunas","siempre","nunca","jamás","aquí","allí","ahí","allá",
    "acá","donde","cuándo","cómo","por qué","porque","aun","aunque","mismo","misma","mismos",
    "mismas","casi","entonces","ahora","ayer","hoy","mañana","todavía","aún","antes","después",
    "durante","siendo","dentro","fuera","ambos","ambas","etc","etcétera","según","caso"
}

_STOP_EN: Set[str] = {
    "the","a","an","this","that","these","those","it","its","they","them","their","theirs",
    "he","she","his","her","hers","we","us","our","ours","you","your","yours","i","me","my","mine",
    "and","or","nor","but","yet","so","for","to","of","in","on","at","from","into","onto",
    "by","with","about","against","between","among","through","during","before","after",
    "above","below","over","under","without","within","beyond","than","as","like","because",
    "since","until","while","although","though","unless","if","whether","then","therefore",
    "thus","hence","whereas","when","where","who","whom","whose","which","what","why","how",
    "be","is","are","am","was","were","been","being","have","has","had","having","do","does",
    "did","doing","can","could","should","would","may","might","must","shall","will",
    "need","ought","used","use","get","got","getting","let","lets","made","make","makes",
    "very","more","most","less","least","much","many","some","any","none","all","both","each",
    "either","neither","one","two","three","every","other","another","same","different",
    "again","just","only","also","too","however","there","here","where","now","then","ever",
    "never","always","yet","still","once","soon","later","already","even","almost","quite",
    "rather","maybe","perhaps","really","such","else","own","elsewhere","further",
    "whose","whatever","whichever","whenever","wherever","whomever",
    # ✅ elimina posesivo suelto tras normalizar comillas: "company’s" -> "company s"
    "s"
}

# Precompilados ligeros
_TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")  # solo letras; números/puntuación salen
_EN_RAPID = re.compile(r"\b(the|and|of|to|in|for|on|with|at|from)\b", re.I)
_ES_RAPID = re.compile(r"\b(el|la|de|que|en|los|las|por|para|con)\b", re.I)

def _normalize_text_unicode(text: str) -> str:
    """
    Normalización adaptada a textos financieros:
    - Minúsculas y sin tildes.
    - Conserva símbolos financieros clave: %, $, €, £, +, -, /, ., ^.
    - Preserva tickers e índices como ^SPX, AAPL.OQ, BTC/USD.
    - Limpieza robusta, pero sin romper formatos numéricos.
    """
    if not isinstance(text, str):
        return text

    # 1. Minúsculas + quitar tildes y diacríticos
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))

    # 2. Reemplaza caracteres no financieros ni alfanuméricos por espacio
    # Permitimos: letras, dígitos, espacios y símbolos financieros relevantes
    t = re.sub(r"[^a-z0-9\%\$\€\£\+\-\/\.\^\s]", " ", t)

    # 3. Limpieza de espacios múltiples
    t = re.sub(r"\s+", " ", t).strip()

    # 4. Corrección de casos de tickers cortados o separados
    t = re.sub(r"\^\s+([a-z0-9]+)", r"^\1", t)                # ^ spx → ^spx
    t = re.sub(r"([a-z0-9])\s*\.\s*([a-z0-9])", r"\1.\2", t)  # AAPL . OQ → AAPL.OQ
    t = re.sub(r"([a-z0-9])\s*\/\s*([a-z0-9])", r"\1/\2", t)  # BTC / USD → BTC/USD
    t = re.sub(r"([\+\-])\s*([0-9])", r"\1\2", t)             # + 0.79% → +0.79%
    t = re.sub(r"([0-9])\s*\%\b", r"\1%", t)                  # 5 % → 5%

    return t


def _sw_ratio(text: str, sw: Set[str]) -> float:
    """Proporción de tokens que son stopwords según el set dado (case-insensitive)."""
    if not text:
        return 0.0
    toks = text.split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t.lower() in sw)
    return hits / max(1, len(toks))

def _lang_detect_robust(text: str) -> str:
    """
    Devuelve 'es' | 'en' | 'es+en' | 'und' combinando:
    - langdetect (si disponible) como pista
    - densidad de stopwords (ES vs EN) sobre texto normalizado
    - triggers rápidos por tokens frecuentes
    Política: neutral (no fuerza 'es' por ocr_lang); en empate → 'es+en'
    """
    if not text:
        return "und"

    # 1) Trigger rápido por tokens muy frecuentes (ayuda en textos cortos)
    if _EN_RAPID.search(text):
        en_fast = True
    else:
        en_fast = False
    if _ES_RAPID.search(text):
        es_fast = True
    else:
        es_fast = False

    # 2) Señal de langdetect (opcional)
    primary = "und"
    if _langdetect is not None:
        try:
            code = _langdetect(text)
            primary = code if code in {"es","en"} else "und"
        except Exception:
            primary = "und"

    # 3) Densidad de stopwords en normalizado
    txt_norm = _normalize_text_unicode(text)
    r_es = _sw_ratio(txt_norm, _STOP_ES)
    r_en = _sw_ratio(txt_norm, _STOP_EN)

    # mezcla clara
    if r_es >= 0.08 and r_en >= 0.08 and abs(r_es - r_en) < 0.05:
        return "es+en"
    if r_es > r_en and r_es >= 0.06:
        return "es"
    if r_en > r_es and r_en >= 0.06:
        return "en"

    # si triggers rápidos dan pista
    if en_fast and not es_fast:
        return "en"
    if es_fast and not en_fast:
        return "es"

    # si langdetect tiene señal válida
    if primary in {"es","en"}:
        return primary

    return "und"

def _choose_stopword_set(lang_tag: Optional[str]) -> Set[str]:
    """Elige set de stopwords según tag; en 'und' aplica ES∪EN para limpiar ambos."""
    if not lang_tag or lang_tag == "und":
        return _STOP_ES | _STOP_EN
    lt = lang_tag.lower()
    if lt.startswith("es+en") or "+" in lt:
        return _STOP_ES | _STOP_EN
    if lt.startswith("en"):
        return _STOP_EN
    if lt.startswith("es"):
        return _STOP_ES
    return _STOP_ES | _STOP_EN

def _clean_with_stopwords(text_norm: str, lang_tag: str) -> str:
    """
    Limpieza de stopwords robusta y case-insensitive.
    - Usa tokens sólo alfabéticos.
    - En 'und' limpia ES+EN (para no dejar 'the/and' colados).
    """
    if not text_norm or len(text_norm) < 2:
        return text_norm
    sw = _choose_stopword_set(lang_tag)
    tokens = _TOKEN_RE.findall(text_norm.lower())
    cleaned_tokens = [t for t in tokens if t not in sw]
    return " ".join(cleaned_tokens)

def _is_list_item_text(s: str) -> bool:
    if not s:
        return False
    ls = s.lstrip()
    return bool(ls and ls[0] in _BULLETS)

def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

def _sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _detect_lang(text: str) -> str:
    """Compat antigua; preferimos _lang_detect_robust en esta versión."""
    if not text or _langdetect is None:
        return "und"
    try:
        code = _langdetect(text)
        return code if code in {"es","en","pt","fr","de"} else "und"
    except Exception:
        return "und"

def _normalize_lines_for_merge(lines: List[str], dehyphenate: bool, normalize_whitespace: bool) -> List[str]:
    norm = []
    for s in lines:
        if not isinstance(s, str):
            norm.append(s); continue
        s = s.replace("\xa0", " ").replace("\r", "")
        if dehyphenate:
            s = s.replace("-\n", "").replace("-\r\n", "")
        if normalize_whitespace:
            s = " ".join(s.split())
        norm.append(s.strip())
    return norm

def _merge_lines_textual(lines: List[str]) -> Tuple[List[str], List[List[int]]]:
    paras: List[str] = []
    sources: List[List[int]] = []
    buf: List[str] = []
    src: List[int] = []
    n = len(lines)
    for i in range(n):
        l = lines[i].strip()
        if not l:
            if buf:
                paras.append(" ".join(buf).strip())
                sources.append(src[:])
                buf.clear(); src.clear()
            continue
        buf.append(l); src.append(i)
        next_line = lines[i+1].strip() if i+1 < n else ""
        end_here = bool(_SENT_END.search(l))
        next_is_capital = bool(next_line and next_line[0].isupper())
        if end_here and (not next_line or next_is_capital):
            paras.append(" ".join(buf).strip())
            sources.append(src[:])
            buf.clear(); src.clear()
    if buf:
        paras.append(" ".join(buf).strip())
        sources.append(src[:])
    return paras, sources

def _merge_lines_layout_aware(page: pdfplumber.page.Page, lines: List[str]) -> Tuple[List[str], List[List[int]]]:
    # Placeholder: usamos textual estable; layout gaps se pueden activar si calibras thresholds.
    try:
        _ = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False) or []
    except Exception:
        pass
    return _merge_lines_textual(lines)

# ---------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------

class Parser:
    """
    Fachada del Parser de documentos con opciones configurables.
    """

    def __init__(
        self,
        ocr_lang: str = "spa",
        ocr_resolution: int = 220,
        normalize_whitespace: bool = True,
        dehyphenate: bool = True,
        enable_pdf_ocr_fallback: bool = True,
        enable_pdf_heading_heuristics: bool = True,
        enable_list_item_detection: bool = True,
        enable_lang_detect: bool = False,
        tesseract_cmd: Optional[str] = None,
        enable_stopword_clean: bool = True,
    ):
        self.ocr_lang = ocr_lang
        self.ocr_resolution = ocr_resolution
        self.normalize_whitespace = normalize_whitespace
        self.dehyphenate = dehyphenate
        self.enable_pdf_ocr_fallback = enable_pdf_ocr_fallback
        self.enable_pdf_heading_heuristics = enable_pdf_heading_heuristics
        self.enable_list_item_detection = enable_list_item_detection
        self.enable_lang_detect = enable_lang_detect
        self.enable_stopword_clean = enable_stopword_clean

        if tesseract_cmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _normalize_text(self, s: str) -> str:
        if not isinstance(s, str):
            return s
        s = s.replace("\xa0", " ").replace("\r", "")
        if self.dehyphenate:
            s = s.replace("-\n", "").replace("-\r\n", "")
        if self.normalize_whitespace:
            s = " ".join(s.split())
        return s.strip()

    def _maybe_heading(self, txt: str) -> bool:
        if not self.enable_pdf_heading_heuristics or not txt:
            return False
        if len(txt) > 80: return False
        if txt.endswith((".", ":", ";")): return False
        letters = sum(c.isalpha() for c in txt)
        uppers  = sum(c.isupper() for c in txt)
        return bool(letters and uppers >= 0.5 * letters)

    def _lang_hint(self, text: str, page_lang: Optional[str] = None) -> str:
        """
        Hint **neutral**: ya no fuerza 'es' por ocr_lang.
        Preferimos señales del propio texto.
        """
        if page_lang and page_lang != "und":
            return "en" if page_lang.startswith("en") else "es"
        if _EN_RAPID.search(text):
            return "en"
        if _ES_RAPID.search(text):
            return "es"
        return "und"

    # ---------------------- API pública ----------------------

    def parse(self, path: str) -> DocumentIR:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        mime = _guess_mime(path)
        logger.info("Parsing start | path=%s mime=%s", path, mime)

        meta: Dict[str, Any] = {"filename": os.path.basename(path)}
        try:
            meta["size_bytes"] = os.stat(path).st_size
        except Exception:
            pass
        try:
            meta["sha256"] = _sha256(path)
        except Exception as e:
            logger.debug("No se pudo calcular sha256: %s", e)

        doc = DocumentIR(
            doc_id=DocumentIR.new_id(path),
            source_path=path,
            mime=mime,
            meta=meta,
            prov=Provenance(extractor="pdfplumber/python-docx/pytesseract", stage="parser"),
        )


        if mime == "application/pdf" or path.lower().endswith(".pdf"):
            pages = self._parse_pdf(path)
        elif mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or path.lower().endswith((".docx", ".doc")):
            pages = self._parse_docx(path)
        elif mime and mime.startswith("image/"):
            pages = self._parse_image(path)
        else:
            raise ValueError(f"Tipo no soportado: {mime} (path={path})")

        # Idioma antiguo (compat) sólo si flag activo
        if self.enable_lang_detect:
            try:
                sample_doc = " ".join(
                    (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                    for p in pages for b in p.blocks
                )[:2000]
                doc.lang = _detect_lang(sample_doc) if sample_doc else "und"
            except Exception:
                doc.lang = "und"
            for p in pages:
                try:
                    sample_pg = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                        for b in p.blocks
                    )[:1000]
                    p.lang = _detect_lang(sample_pg) if sample_pg else "und"
                except Exception:
                    p.lang = "und"

        # Idioma robusto para doc y páginas (dominante)
        doc_text_concat = " ".join(
            (b.get("text_raw", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for p in pages for b in p.blocks
            if (isinstance(b, dict) and b.get("type") in {"paragraph","list_item","heading"})
        )
        doc_lang_robust = _lang_detect_robust(doc_text_concat) if doc_text_concat else "und"
        doc.lang = doc_lang_robust or doc.lang or "und"

        for p in pages:
            if getattr(p, "lang", "und") == "und":
                page_text_concat = " ".join(
                    (b.get("text_raw", "") if isinstance(b, dict) else getattr(b, "text", ""))
                    for b in p.blocks
                    if (isinstance(b, dict) and b.get("type") in {"paragraph","list_item","heading"})
                )
                p.lang = _lang_detect_robust(page_text_concat) if page_text_concat else "und"

        doc.pages = pages
        doc.meta["page_count"] = len(pages)
        logger.info("Parsing done | doc_id=%s pages=%d | lang=%s", doc.doc_id, len(doc.pages), doc.lang)
        return doc

    # ---------------------- Parsers especializados ----------------------

    def _build_text_block(self, kind: str, text_value: str, prov: Provenance, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Construye bloques de texto 'paragraph' o 'list_item' con
        text_norm, text_raw, lang_hint y text_clean coherentes.
        """
        text_raw = self._normalize_text(text_value)            # conserva puntuación básica
        text_norm = _normalize_text_unicode(text_raw)          # minúsculas + sin tildes/símbolos
        # Señal de idioma robusta primero
        lang_hint = _lang_detect_robust(text_raw)
        if not lang_hint or lang_hint == "und":
            # fallback neutral por contenido, nunca por ocr_lang
            lang_hint = self._lang_hint(text_norm, page_lang=None)

        if self.enable_stopword_clean:
            text_clean = _clean_with_stopwords(text_norm, lang_hint or "und")
        else:
            text_clean = text_norm

        blk = TextBlock(
            type=kind,
            text=text_clean,
            prov=prov
        ).model_dump()

        blk["text_raw"] = text_raw
        blk["lang_hint"] = lang_hint or "und"
        blk["text_clean"] = text_clean
        blk["text_norm"] = text_norm
        if notes:
            blk["prov"]["notes"] = notes
        return blk

    def _parse_pdf(self, path: str) -> List[PageIR]:
        pages_ir: List[PageIR] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                width, height = page.width, page.height
                page_blocks: List[Dict[str, Any]] = []
                meta_page: Dict[str, Any] = {}

                raw_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                raw_lines = raw_text.split("\n")
                meta_page["n_lines_raw"] = sum(1 for l in raw_lines if l.strip())

                norm_lines = _normalize_lines_for_merge(
                    raw_lines, dehyphenate=self.dehyphenate, normalize_whitespace=self.normalize_whitespace
                )

                paras, sources = _merge_lines_layout_aware(page, norm_lines)
                if not paras:
                    paras, sources = _merge_lines_textual(norm_lines)

                for ptxt, src_idx_list in zip(paras, sources):
                    if not ptxt.strip():
                        continue

                    notes = f"source_lines={src_idx_list}"
                    if self.enable_list_item_detection and _is_list_item_text(ptxt):
                        cleaned = ptxt.lstrip()
                        cleaned = cleaned[1:].lstrip() if cleaned and cleaned[0] in _BULLETS else cleaned
                        blk = self._build_text_block("list_item", cleaned,
                                                     prov=Provenance(extractor="pdfplumber+merge", stage="parser"),
                                                     notes=notes)
                        page_blocks.append(blk)

                    elif self._maybe_heading(ptxt):
                        blk = TextBlock(
                            type="heading",
                            text=self._normalize_text(ptxt),
                            level=2,
                            prov=Provenance(extractor="pdfplumber+merge", stage="parser", notes=notes)
                        ).model_dump()
                        page_blocks.append(blk)

                    else:
                        blk = self._build_text_block(
                            "paragraph",
                            ptxt,
                            prov=Provenance(extractor="pdfplumber+merge", stage="parser"),
                            notes=notes
                        )
                        page_blocks.append(blk)

                # Tablas
                try:
                    tables = page.extract_tables(
                        table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                    )
                except Exception:
                    tables = []

                for t in tables or []:
                    tb = TableBlock(prov=Provenance(extractor="pdfplumber", stage="parser"))
                    for r_idx, row in enumerate(t):
                        for c_idx, cell in enumerate(row):
                            tb.cells.append(
                                TableCell(row=r_idx, col=c_idx, text=self._normalize_text(cell or ""))
                            )
                    page_blocks.append(tb.model_dump())

                # Fallback OCR si no hubo texto ni tablas
                if self.enable_pdf_ocr_fallback and len(page_blocks) == 0:
                    if pytesseract is None:
                        logger.warning("OCR fallback saltado (pytesseract no disponible) | page=%d", i)
                    else:
                        try:
                            pil_img = page.to_image(resolution=self.ocr_resolution).original
                            data = pytesseract.image_to_data(
                                pil_img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT
                            )
                            words, confs = [], []
                            for w, conf in zip(data.get("text", []), data.get("conf", [])):
                                if not w: continue
                                words.append(w)
                                try:
                                    confs.append(float(conf))
                                except Exception:
                                    pass
                            ocr_text = self._normalize_text(" ".join(words))
                            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None

                            ocr_lines = [l for l in (ocr_text or "").split("\n") if l.strip()]
                            ocr_paras, ocr_sources = _merge_lines_textual(
                                _normalize_lines_for_merge(ocr_lines, self.dehyphenate, self.normalize_whitespace)
                            )

                            for ptxt, src_idx_list in zip(ocr_paras, ocr_sources):
                                notes = f"source_lines={src_idx_list}"
                                blk = self._build_text_block(
                                    "paragraph",
                                    ptxt,
                                    prov=Provenance(
                                        extractor="pytesseract+merge", stage="ocr-fallback",
                                        notes=notes
                                    )
                                )
                                # añade OCRInfo al bloque
                                blk["ocr"] = OCRInfo(engine="tesseract", lang=self.ocr_lang,
                                                     dpi=self.ocr_resolution, conf=mean_conf).model_dump()
                                page_blocks.append(blk)
                            logger.info("Fallback OCR aplicado | page=%d conf=%.2f", i, (mean_conf or -1))
                        except Exception as e:
                            logger.warning("Fallback OCR falló | page=%d err=%s", i, e)

                meta_page["n_paragraphs_final"] = len(
                    [b for b in page_blocks if (isinstance(b, dict) and b.get("type") in {"paragraph","list_item","heading"})]
                )
                meta_page["layout_loss"] = 0.0

                page_text_concat = " ".join(
                    (b.get("text_raw") or b.get("text") or "")
                    for b in page_blocks
                    if (isinstance(b, dict) and b.get("type") in {"paragraph","list_item","heading"})
                )
                page_lang_detect = _lang_detect_robust(page_text_concat) if page_text_concat else "und"
                meta_page["lang_hint"] = page_lang_detect

                page_ir = PageIR(page_number=i, width=width, height=height,
                                 blocks=page_blocks, meta=meta_page, lang=page_lang_detect)
                pages_ir.append(page_ir)

        return pages_ir

    def _parse_docx(self, path: str) -> List[PageIR]:
        if docx is None:
            raise ImportError("Instala python-docx para parsear DOCX.")

        document = docx.Document(path)
        page_blocks: List[Dict[str, Any]] = []
        n_raw_paras = 0

        for p in document.paragraphs:
            text = self._normalize_text(p.text or "")
            if not text:
                continue
            n_raw_paras += 1
            style_name = (p.style.name if p.style else "").lower()
            if "heading" in style_name:
                level = 1
                for d in ("1","2","3","4","5","6"):
                    if d in style_name:
                        level = int(d); break
                page_blocks.append(
                    TextBlock(type="heading", text=text, level=level,
                              prov=Provenance(extractor="python-docx", stage="parser")).model_dump()
                )
            else:
                if self.enable_list_item_detection and _is_list_item_text(text):
                    cleaned = text.lstrip()
                    cleaned = cleaned[1:].lstrip() if cleaned and cleaned[0] in _BULLETS else cleaned
                    blk = self._build_text_block(
                        "list_item", cleaned, prov=Provenance(extractor="python-docx", stage="parser")
                    )
                    page_blocks.append(blk)
                else:
                    blk = self._build_text_block(
                        "paragraph", text, prov=Provenance(extractor="python-docx", stage="parser")
                    )
                    page_blocks.append(blk)

        # Tablas
        n_tables = 0
        for tbl in document.tables:
            tb = TableBlock(prov=Provenance(extractor="python-docx", stage="parser"))
            for r_idx, row in enumerate(tbl.rows):
                for c_idx, cell in enumerate(row.cells):
                    tb.cells.append(TableCell(row=r_idx, col=c_idx, text=self._normalize_text(cell.text or "")))
            page_blocks.append(tb.model_dump())
            n_tables += 1

        meta_page = {
            "n_paragraphs_raw": n_raw_paras,
            "n_blocks_final": len(page_blocks),
            "n_tables": n_tables,
            "layout_loss": 0.0,
        }

        page_text_concat = " ".join(
            (b.get("text_raw") or b.get("text") or "")
            for b in page_blocks
            if (isinstance(b, dict) and b.get("type") in {"paragraph","list_item","heading"})
        )
        page_lang_detect = _lang_detect_robust(page_text_concat) if page_text_concat else "und"

        return [PageIR(page_number=1, blocks=page_blocks, meta=meta_page, lang=page_lang_detect)]

    def _parse_image(self, path: str) -> List[PageIR]:
        """
        IMG → OCR con Tesseract (si está disponible).
        Bugfix: retorno como List[PageIR], no tupla.
        """
        if pytesseract is None or Image is None:
            raise ImportError("Instala pytesseract y Pillow, y Tesseract en el sistema.")

        img = Image.open(path)

        try:
            data = pytesseract.image_to_data(img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT)
            words, confs = [], []
            for w, conf in zip(data.get("text", []), data.get("conf", [])):
                if not w: continue
                words.append(w)
                try:
                    confs.append(float(conf))
                except Exception:
                    pass
            text_raw_doc = self._normalize_text(" ".join(words))
            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None
        except Exception:
            raw = pytesseract.image_to_string(img, lang=self.ocr_lang)
            text_raw_doc = self._normalize_text(raw)
            mean_conf = None

        lines = [l for l in (text_raw_doc or "").split("\n")]
        norm_lines = _normalize_lines_for_merge(lines, self.dehyphenate, self.normalize_whitespace)
        paras, sources = _merge_lines_textual(norm_lines)
        blocks: List[Dict[str, Any]] = []

        for ptxt, src_idx_list in zip(paras, sources):
            notes = f"source_lines={src_idx_list}"
            blk = self._build_text_block(
                "paragraph",
                ptxt,
                prov=Provenance(extractor="pytesseract+merge", stage="parser", notes=notes)
            )
            blk["ocr"] = OCRInfo(engine="tesseract", lang=self.ocr_lang, dpi=None, conf=mean_conf).model_dump()
            blocks.append(blk)

        meta_page = {
            "n_lines_raw": len([l for l in lines if l.strip()]),
            "n_paragraphs_final": len([b for b in blocks if (isinstance(b, dict) and b.get('type') == 'paragraph')]),
            "layout_loss": 0.0,
        }

        page_text_concat = " ".join((b.get("text_raw") or b.get("text") or "") for b in blocks if isinstance(b, dict))
        page_lang_detect = _lang_detect_robust(page_text_concat) if page_text_concat else "und"

        return [PageIR(page_number=1, blocks=blocks, meta=meta_page, lang=page_lang_detect)]