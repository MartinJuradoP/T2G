# parser/parsers.py
# -*- coding: utf-8 -*-
"""
parsers.py — Implementación del Parser (PDF / DOCX / IMG → IR)

Resumen
-------
Convierte documentos heterogéneos a una representación intermedia (IR) homogénea
basada en JSON/MD, con bloques de texto y tablas por página, manteniendo un
contrato de salida estable para los siguientes subsistemas (chunking, IE, etc.).

Características clave
---------------------
- Detección de tipo por MIME / extensión y dispatch a parser especializado.
- PDF: texto por líneas + tablas básicas con pdfplumber.
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
- enable_lang_detect: Detecta idioma (doc/página) si está disponible `langdetect`.
- tesseract_cmd: Ruta al ejecutable de tesseract (útil en Windows).

Mejoras futuras (ideas)
-----------------------
- Headings PDF robustos con font-size (page.extract_words()).
- Tablas PDF robustas con camelot o tabula-py (cuando el PDF es vectorial).
- Preproceso de imagen (deskew/denoise/binarización) antes de OCR (OpenCV).
"""

from __future__ import annotations
import os
import io
import mimetypes
import logging
import hashlib
from typing import List, Dict, Any, Optional

import pdfplumber

# Import opcionales protegidos: no romper si no están presentes
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

from parser.schemas import (
    DocumentIR, PageIR, TextBlock, TableBlock, TableCell,
    Provenance, OCRInfo
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------

# Bullets/viñetas comunes para detectar list items
_BULLETS = tuple("•·◦▪-–—*·")

def _is_list_item_text(s: str) -> bool:
    """Heurística simple: detecta líneas con bullets/guiones al inicio."""
    if not s:
        return False
    ls = s.lstrip()
    return bool(ls and ls[0] in _BULLETS)

def _guess_mime(path: str) -> str:
    """
    Infiera el MIME a partir de la extensión.
    Si no se conoce, usa 'application/octet-stream' como fallback.
    """
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

def _sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Calcula sha256 en streaming para trazabilidad."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _detect_lang(text: str) -> str:
    """
    Detecta idioma 'es'/'en' si langdetect está disponible, si no 'und'.
    Nota: puedes extender mapeos a pt/fr/de si lo necesitas.
    """
    if not text or _langdetect is None:
        return "und"
    try:
        code = _langdetect(text)
        return code if code in {"es", "en", "pt", "fr", "de"} else "und"
    except Exception:
        return "und"


class Parser:
    """
    Fachada del Parser de documentos con opciones configurables.

    Args
    ----
    ocr_lang : str
        Idioma(s) para Tesseract. Ej: "spa", "eng" o "spa+eng".
    ocr_resolution : int
        DPI para rasterizar páginas PDF cuando se usa fallback OCR (200–300 recomendado).
    normalize_whitespace : bool
        Si True, colapsa espacios múltiples, quita extra whitespaces y normaliza líneas.
    dehyphenate : bool
        Si True, une palabras cortadas por guion al final de línea ("infor-\\nmación" -> "información").
    enable_pdf_ocr_fallback : bool
        Si True, intenta OCR por página si pdfplumber no extrajo texto ni tablas (PDFs escaneados).
    enable_pdf_heading_heuristics : bool
        Activa una heurística conservadora para marcar encabezados en PDF.
    enable_list_item_detection : bool
        Si True, intenta clasificar líneas con bullets/guiones como 'list_item'.
    enable_lang_detect : bool
        Si True, intenta detectar idioma del documento y por página (si hay lib).
    tesseract_cmd : Optional[str]
        Ruta al ejecutable de Tesseract (útil en Windows). Ej:
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    Uso
    ---
        parser = Parser(ocr_lang="spa+eng", ocr_resolution=220)
        doc_ir = parser.parse("archivo.pdf")
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
    ):
        self.ocr_lang = ocr_lang
        self.ocr_resolution = ocr_resolution
        self.normalize_whitespace = normalize_whitespace
        self.dehyphenate = dehyphenate
        self.enable_pdf_ocr_fallback = enable_pdf_ocr_fallback
        self.enable_pdf_heading_heuristics = enable_pdf_heading_heuristics
        self.enable_list_item_detection = enable_list_item_detection
        self.enable_lang_detect = enable_lang_detect

        # Configurar Tesseract manualmente si pasamos ruta (Windows / entornos custom)
        if tesseract_cmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # ---------------------- Helpers internos ----------------------

    def _normalize_text(self, s: str) -> str:
        """
        Normaliza texto de salida para IR:
        - Reemplaza NBSP (\xa0) por espacio y remueve \r
        - Une palabras cortadas por guion (opcional)
        - Colapsa espacios múltiples y limpia bordes (opcional)
        """
        if not isinstance(s, str):
            return s
        s = s.replace("\xa0", " ").replace("\r", "")
        if self.dehyphenate:
            # Heurística simple: elimina "-\n" para unir palabras cortadas al salto de línea
            s = s.replace("-\n", "").replace("-\r\n", "")
        if self.normalize_whitespace:
            # Normaliza cada línea y vuelve a unir con \n
            s = "\n".join(" ".join(line.split()) for line in s.split("\n"))
        return s.strip()

    def _maybe_heading(self, txt: str) -> bool:
        """
        Heurística MUY conservadora de heading:
        - Línea relativamente corta
        - No termina en puntuación fuerte (.,:;)
        - Proporción de mayúsculas alta respecto a alfabéticos
        """
        if not self.enable_pdf_heading_heuristics:
            return False
        if not txt:
            return False
        if len(txt) > 80:
            return False
        if txt.endswith((".", ":", ";")):
            return False
        letters = sum(c.isalpha() for c in txt)
        uppers = sum(c.isupper() for c in txt)
        return bool(letters and uppers >= 0.5 * letters)

    # ---------------------- API pública ----------------------

    def parse(self, path: str) -> DocumentIR:
        """
        Detecta el tipo de documento y lo parsea a IR.

        Devuelve
        --------
        DocumentIR
            Estructura con doc_id, meta (size_bytes, page_count, sha256), pages[PageIR], etc.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        mime = _guess_mime(path)
        logger.info("Parsing start | path=%s mime=%s", path, mime)

        # Metadatos útiles para trazabilidad
        meta: Dict[str, Any] = {"filename": os.path.basename(path)}
        try:
            stat = os.stat(path)
            meta["size_bytes"] = stat.st_size
        except Exception:
            pass
        try:
            meta["sha256"] = _sha256(path)
        except Exception as e:
            logger.debug("No se pudo calcular sha256: %s", e)

        doc = DocumentIR(
            doc_id=DocumentIR.new_id(),
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

        # Idioma (opcional)
        if self.enable_lang_detect:
            # Doc-level
            try:
                sample_doc = " ".join(
                    (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                    for p in pages for b in p.blocks
                )[:2000]
                doc.lang = _detect_lang(sample_doc)
            except Exception:
                doc.lang = "und"
            # Page-level
            for p in pages:
                try:
                    sample_pg = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                        for b in p.blocks
                    )[:1000]
                    p.lang = _detect_lang(sample_pg)
                except Exception:
                    p.lang = "und"

        doc.pages = pages
        doc.meta["page_count"] = len(pages)
        logger.info("Parsing done | doc_id=%s pages=%d", doc.doc_id, len(doc.pages))
        return doc

    # ---------------------- Parsers especializados ----------------------

    def _parse_pdf(self, path: str) -> List[PageIR]:
        """
        PDF → PageIR[]:
        - Texto por líneas con pdfplumber (simple pero robusto).
        - Tablas básicas con pdfplumber (mejorable según caso).
        - Fallback OCR por página si no hubo texto ni tablas (PDFs escaneados),
          aplicado con Tesseract si está disponible y habilitado.
        """
        pages_ir: List[PageIR] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                width, height = page.width, page.height
                page_blocks: List[Dict[str, Any]] = []

                # 1) Texto por líneas
                lines = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                lines = self._normalize_text(lines)

                if lines:
                    for paragraph in [p for p in lines.split("\n") if p.strip()]:
                        txt = paragraph.strip()
                        if self.enable_list_item_detection and _is_list_item_text(txt):
                            # Quita el bullet inicial (si existe) y espacios
                            cleaned = txt.lstrip()
                            cleaned = cleaned[1:].lstrip() if cleaned and cleaned[0] in _BULLETS else txt
                            blk = TextBlock(
                                type="list_item", text=cleaned,
                                prov=Provenance(extractor="pdfplumber", stage="parser")
                            )
                        else:
                            if self._maybe_heading(txt):
                                blk = TextBlock(
                                    type="heading", text=txt, level=2,
                                    prov=Provenance(extractor="pdfplumber", stage="parser")
                                )
                            else:
                                blk = TextBlock(
                                    type="paragraph", text=txt,
                                    prov=Provenance(extractor="pdfplumber", stage="parser")
                                )
                        page_blocks.append(blk.model_dump())

                # 2) Tablas básicas (nota: PDFs escaneados normalmente NO tendrán tablas extraíbles)
                tables = []
                try:
                    tables = page.extract_tables(
                        table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                    )
                except Exception as e:
                    logger.debug("Table extraction failed on page %d: %s", i, e)

                for t in tables or []:
                    tb = TableBlock(prov=Provenance(extractor="pdfplumber", stage="parser"))
                    for r_idx, row in enumerate(t):
                        for c_idx, cell in enumerate(row):
                            tb.cells.append(
                                TableCell(row=r_idx, col=c_idx, text=self._normalize_text(cell or ""))
                            )
                    page_blocks.append(tb.model_dump())

                # 3) Fallback OCR si no hubo texto ni tablas (caso típico: PDF escaneado)
                if self.enable_pdf_ocr_fallback and len(page_blocks) == 0:
                    if pytesseract is None:
                        logger.warning("OCR fallback saltado (pytesseract no disponible) | page=%d", i)
                    else:
                        try:
                            # Renderizar la página a imagen con DPI configurables.
                            pil_img = page.to_image(resolution=self.ocr_resolution).original
                            # Usamos image_to_data para extraer conf promedio por palabra
                            data = pytesseract.image_to_data(
                                pil_img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT
                            )
                            words, confs = [], []
                            for w, conf in zip(data.get("text", []), data.get("conf", [])):
                                if not w:
                                    continue
                                words.append(w)
                                # conf puede venir como str; si numérico, normaliza a 0..1
                                try:
                                    confs.append(float(conf))
                                except Exception:
                                    pass
                            ocr_text = self._normalize_text(" ".join(words))
                            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None

                            for line in [l for l in (ocr_text or "").split("\n") if l.strip()]:
                                page_blocks.append(
                                    TextBlock(
                                        type="paragraph",
                                        text=line,
                                        ocr=OCRInfo(engine="tesseract", lang=self.ocr_lang,
                                                    dpi=self.ocr_resolution, conf=mean_conf),
                                        prov=Provenance(extractor="pytesseract", stage="ocr-fallback"),
                                    ).model_dump()
                                )
                            logger.info("Fallback OCR aplicado | page=%d conf=%.2f", i, (mean_conf or -1))
                        except Exception as e:
                            logger.warning("Fallback OCR falló | page=%d err=%s", i, e)

                pages_ir.append(PageIR(page_number=i, width=width, height=height, blocks=page_blocks))
        return pages_ir

    def _parse_docx(self, path: str) -> List[PageIR]:
        """
        DOCX → PageIR único:
        - Párrafos / Headings detectados por estilo.
        - Tablas con contenido por celda.
        Nota: DOCX no expone páginas físicas; devolvemos una 'página lógica' (page_number=1).
        """
        if docx is None:
            raise ImportError("Instala python-docx para parsear DOCX.")

        document = docx.Document(path)
        page_blocks: List[Dict[str, Any]] = []

        # Párrafos / Headings
        for p in document.paragraphs:
            text = self._normalize_text(p.text or "")
            if not text:
                continue
            style_name = (p.style.name if p.style else "").lower()
            if "heading" in style_name:
                # Detecta nivel (Heading 1, Heading 2, …) de forma simple
                level = 1
                for d in ("1", "2", "3", "4", "5", "6"):
                    if d in style_name:
                        level = int(d)
                        break
                page_blocks.append(
                    TextBlock(type="heading", text=text, level=level,
                              prov=Provenance(extractor="python-docx", stage="parser")).model_dump()
                )
            else:
                if self.enable_list_item_detection and _is_list_item_text(text):
                    cleaned = text.lstrip()
                    cleaned = cleaned[1:].lstrip() if cleaned and cleaned[0] in _BULLETS else text
                    page_blocks.append(
                        TextBlock(type="list_item", text=cleaned,
                                  prov=Provenance(extractor="python-docx", stage="parser")).model_dump()
                    )
                else:
                    page_blocks.append(
                        TextBlock(type="paragraph", text=text,
                                  prov=Provenance(extractor="python-docx", stage="parser")).model_dump()
                    )

        # Tablas
        for tbl in document.tables:
            tb = TableBlock(prov=Provenance(extractor="python-docx", stage="parser"))
            for r_idx, row in enumerate(tbl.rows):
                for c_idx, cell in enumerate(row.cells):
                    tb.cells.append(TableCell(row=r_idx, col=c_idx, text=self._normalize_text(cell.text or "")))
            page_blocks.append(tb.model_dump())

        return [PageIR(page_number=1, blocks=page_blocks)]

    def _parse_image(self, path: str) -> List[PageIR]:
        """
        IMG → OCR con Tesseract (si está disponible).
        Recomendado: preprocesar con OpenCV (deskew/denoise/binarización) si las imágenes son ruidosas.
        """
        if pytesseract is None or Image is None:
            raise ImportError("Instala pytesseract y Pillow, y Tesseract en el sistema.")

        img = Image.open(path)

        # Igual que en PDF fallback: obtenemos conf promedio
        try:
            data = pytesseract.image_to_data(img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT)
            words, confs = [], []
            for w, conf in zip(data.get("text", []), data.get("conf", [])):
                if not w:
                    continue
                words.append(w)
                try:
                    confs.append(float(conf))
                except Exception:
                    pass
            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None
            text = self._normalize_text(" ".join(words))
        except Exception:
            # Fallback a image_to_string si falla image_to_data
            raw = pytesseract.image_to_string(img, lang=self.ocr_lang)
            text = self._normalize_text(raw)
            mean_conf = None

        blocks: List[Dict[str, Any]] = []
        for line in [l for l in (text or "").split("\n") if l.strip()]:
            blocks.append(
                TextBlock(
                    type="paragraph",
                    text=line,
                    ocr=OCRInfo(engine="tesseract", lang=self.ocr_lang, dpi=None, conf=mean_conf),
                    prov=Provenance(extractor="pytesseract", stage="parser"),
                ).model_dump()
            )
        return [PageIR(page_number=1, blocks=blocks)]