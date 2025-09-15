# chunker/hybrid_chunker.py
# -*- coding: utf-8 -*-
"""
HybridChunker — Segmentación híbrida y estable (≤2048 chars)

Resumen
-------
Genera chunks semánticamente "cohesionados" a partir de una IR por páginas/bloques,
respetando:
- Límites naturales (encabezados, párrafos, tablas)
- Fronteras de oración cuando sea posible (spaCy o regex)
- Ventanas por tamaño con solapamiento (overlap) controlado

Estrategia
----------
1) Aplana la IR a una secuencia de unidades atómicas: {kind, text, page_idx, block_idx}
   - "paragraph" y "heading" → texto
   - "table" → texto tabular serializado (CSV simple por filas)
   - "figure" → usa caption si existe
2) Junta unidades hasta alcanzar ~target_chars (respetando máximos/mínimos).
3) Si se excede, recorta en límites de oración (spaCy/regex) con preferencia a target_chars.
4) Aplica solapamiento entre chunks (overlap_chars) para robustecer recuperación.
5) Etiqueta chunk.type según el % de contenido de tabla vs texto.

Compatibilidad
--------------
- Soporta PageIR.blocks como dicts (model_dump) o como modelos Pydantic
  (TextBlock, TableBlock, FigureBlock, ...).
- API estable: ChunkerConfig y HybridChunker(chunk_document) se mantienen.

Notas
-----
- Si el splitter es "spacy" o "auto" y el modelo cargado no trae 'parser' ni 'senter',
  se añade 'sentencizer' → evita el error [E030] “Sentence boundaries unset”.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Contratos de la IR
from parser.schemas import (
    DocumentIR, PageIR, ChunkIR, DocumentChunks,
    TextBlock, TableBlock, FigureBlock, TableCell
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Regex de split por oración (fallback). Incluye '…' y mayúsculas con diéresis.
_SENT_SPLIT_REGEX = re.compile(r"(?<=[\.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡0-9])")


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
@dataclass
class ChunkerConfig:
    target_chars: int = 1400          # tamaño objetivo del chunk
    max_chars: int = 2048             # límite duro que no se debe superar
    min_chars: int = 400              # preferencia mínima (relajada si el doc es corto)
    overlap_chars: int = 120          # solapamiento entre chunks consecutivos
    keep_headings_with_next: bool = True
    table_policy: str = "isolate"     # "isolate" | "merge"
    sentence_splitter: str = "auto"   # "spacy" | "regex" | "auto"


# -----------------------------------------------------------------------------
# Chunker
# -----------------------------------------------------------------------------
class HybridChunker:
    """
    Implementación determinística y rápida de un chunker híbrido.
    - Predomina la semántica, pero con límites estables por tamaño.
    - Usa spaCy para cortar por oraciones si es posible; sino, regex robusto.
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.cfg = config or ChunkerConfig()
        self._nlp = None  # pipeline spaCy opcional

        # Solo intentamos spaCy si el usuario lo pide o si está en auto
        if self.cfg.sentence_splitter in ("spacy", "auto"):
            try:
                import spacy
                # Cargamos un modelo liviano; si no está, seguimos al siguiente idioma.
                for model in ("es_core_news_sm", "en_core_web_sm"):
                    try:
                        # Deshabilitamos componentes costosos por performance;
                        # 'parser' queda fuera, pero luego garantizamos doc.sents con 'senter' o 'sentencizer'.
                        self._nlp = spacy.load(
                            model,
                            disable=[
                                "tagger", "ner", "lemmatizer", "attribute_ruler",
                                "tok2vec", "morphologizer", "parser"
                            ],
                        )
                        break
                    except Exception:
                        continue

                # Garantizar límites de oración:
                # Si no hay 'parser' (deshabilitado) ni 'senter', añadimos 'sentencizer'.
                if self._nlp is not None:
                    pipe_names = set(self._nlp.pipe_names)
                    if ("parser" not in pipe_names) and ("senter" not in pipe_names):
                        try:
                            self._nlp.add_pipe("sentencizer")
                            logger.info("spaCy: 'sentencizer' agregado para habilitar doc.sents en HybridChunker.")
                        except Exception as e:
                            logger.warning("No se pudo agregar 'sentencizer' (%s). Se usará regex en el chunker.", e)
                            self._nlp = None
            except ImportError:
                self._nlp = None

        # Si el usuario pidió explícitamente "spacy" y no se cargó, avisamos.
        if self.cfg.sentence_splitter == "spacy" and self._nlp is None:
            logger.warning("sentence_splitter='spacy' solicitado, pero no se pudo cargar spaCy. Usando regex.")

    # -------------------------- API pública --------------------------

    def chunk_document(self, doc: DocumentIR) -> DocumentChunks:
        """
        Crea chunks para un DocumentIR y devuelve DocumentChunks con trazabilidad.
        """
        units = self._flatten_ir(doc)
        chunks = self._greedy_pack(doc, units)
        return DocumentChunks(
            doc_id=doc.doc_id,
            strategy="hybrid-v1",
            version="1.0",
            chunks=chunks,
            meta={
                "target_chars": self.cfg.target_chars,
                "max_chars": self.cfg.max_chars,
                "min_chars": self.cfg.min_chars,
                "overlap_chars": self.cfg.overlap_chars,
                "table_policy": self.cfg.table_policy,
                "sentence_splitter": ("spacy" if self._nlp else "regex")
                    if self.cfg.sentence_splitter == "auto"
                    else self.cfg.sentence_splitter,
            },
        )

    # -------------------------- Etapas internas --------------------------

    @staticmethod
    def _block_kind(block: Union[dict, TextBlock, TableBlock, FigureBlock]) -> str:
        """
        Devuelve el tipo de bloque ('paragraph', 'heading', 'table', 'figure', etc.)
        para dicts (model_dump) o modelos Pydantic.
        """
        if isinstance(block, dict):
            return block.get("type", "paragraph")
        return getattr(block, "type", "paragraph")

    @staticmethod
    def _block_text(block: Union[dict, TextBlock]) -> str:
        """Extrae el texto del bloque (solo para tipos textuales)."""
        if isinstance(block, dict):
            return (block.get("text") or "").strip()
        return (getattr(block, "text", "") or "").strip()

    @staticmethod
    def _block_level(block: Union[dict, TextBlock]) -> int:
        """Devuelve nivel de heading (si aplica)."""
        if isinstance(block, dict):
            return int(block.get("level", 1) or 1)
        return int(getattr(block, "level", 1) or 1)

    @staticmethod
    def _table_to_text(block: Union[dict, TableBlock]) -> str:
        """
        Serializa una tabla a texto plano (CSV-like por filas).
        Soporta celdas como dicts o como modelos TableCell.
        """
        rows_map: Dict[int, Dict[int, str]] = {}

        if isinstance(block, dict):
            cells = (block.get("cells", []) or [])
            for cell in cells:
                if isinstance(cell, dict):
                    r = int(cell.get("row", 0))
                    c = int(cell.get("col", 0))
                    t = (cell.get("text") or "").replace("\n", " ").strip()
                else:
                    # Si accidentalmente viniera un modelo
                    r = int(getattr(cell, "row", 0))
                    c = int(getattr(cell, "col", 0))
                    t = (getattr(cell, "text", "") or "").replace("\n", " ").strip()
                rows_map.setdefault(r, {})[c] = t
        else:
            # TableBlock (modelo)
            for cell in (block.cells or []):
                if isinstance(cell, TableCell):
                    r = int(cell.row); c = int(cell.col)
                    t = (cell.text or "").replace("\n", " ").strip()
                else:
                    r = int(getattr(cell, "row", 0))
                    c = int(getattr(cell, "col", 0))
                    t = (getattr(cell, "text", "") or "").replace("\n", " ").strip()
                rows_map.setdefault(r, {})[c] = t

        lines: List[str] = []
        for r in sorted(rows_map.keys()):
            cols = rows_map[r]
            line = ", ".join(cols.get(c, "") for c in sorted(cols.keys()))
            lines.append(line)
        return "\n".join(lines).strip()

    def _flatten_ir(self, doc: DocumentIR) -> List[Dict[str, Any]]:
        """
        Convierte PageIR.blocks en una lista lineal de unidades normalizadas:
        {kind, text, page_idx, block_idx}
        - paragraph → texto
        - heading(level=n) → "## ..." para conservar jerarquía
        - table → CSV-like por filas
        - figure → usa caption (si existe), si no se ignora
        """
        units: List[Dict[str, Any]] = []

        for p_idx, page in enumerate(doc.pages):
            for b_idx, block in enumerate(page.blocks):
                kind = self._block_kind(block)

                if kind in ("paragraph", "heading", "list_item", "footer", "header", "unknown"):
                    text = self._block_text(block)
                    if not text:
                        continue
                    if kind == "heading":
                        level = max(1, min(self._block_level(block), 6))
                        text = f"{'#' * level} {text}"
                    units.append({
                        "kind": "heading" if kind == "heading" else "paragraph",
                        "text": text,
                        "page_idx": p_idx,
                        "block_idx": b_idx
                    })

                elif kind == "table":
                    text = self._table_to_text(block)
                    if text:
                        units.append({
                            "kind": "table",
                            "text": text,
                            "page_idx": p_idx,
                            "block_idx": b_idx
                        })

                elif kind == "figure":
                    # Extrae caption si existe; si no, omite
                    caption = ""
                    if isinstance(block, dict):
                        caption = (block.get("caption") or "").strip()
                    else:
                        caption = (getattr(block, "caption", "") or "").strip()
                    if caption:
                        units.append({
                            "kind": "paragraph",
                            "text": caption,
                            "page_idx": p_idx,
                            "block_idx": b_idx
                        })

                else:
                    # Tipos no contemplados explícitamente → intenta texto si existe
                    text = self._block_text(block)
                    if text:
                        units.append({
                            "kind": "paragraph",
                            "text": text,
                            "page_idx": p_idx,
                            "block_idx": b_idx
                        })

        # Opcional: pega headings con el siguiente bloque del MISMO page_idx
        if self.cfg.keep_headings_with_next:
            merged: List[Dict[str, Any]] = []
            i = 0
            while i < len(units):
                u = units[i]
                if (
                    u["kind"] == "heading"
                    and i + 1 < len(units)
                    and units[i + 1]["page_idx"] == u["page_idx"]
                ):
                    nxt = units[i + 1]
                    merged_text = u["text"] + "\n" + nxt["text"]
                    merged.append({
                        "kind": nxt["kind"],
                        "text": merged_text,
                        "page_idx": u["page_idx"],
                        "block_idx": u["block_idx"],  # conserva idx del heading para trazabilidad
                    })
                    i += 2
                else:
                    merged.append(u)
                    i += 1
            units = merged

        return units

    def _greedy_pack(self, doc: DocumentIR, units: List[Dict[str, Any]]) -> List[ChunkIR]:
        """
        Empaquetado codicioso:
        - Si table_policy == "isolate": cada unidad "table" sale sola como chunk.
        - Para texto: acumulamos hasta target_chars; si nos pasamos, cortamos en oraciones.
        - Añadimos solapamiento de caracteres entre chunks consecutivos.
        """
        cfg = self.cfg
        chunks: List[ChunkIR] = []

        buf = ""
        buf_origin: List[Tuple[int, int, str]] = []  # (page_idx, block_idx, kind)
        i = 0

        def flush_chunk(reason: str):
            """
            Emite el buffer actual como ChunkIR con meta informativa y
            aplica el solapamiento configurado.
            """
            nonlocal buf, buf_origin
            if not buf.strip():
                buf = ""
                buf_origin = []
                return

            # Para IDs reproducibles sin llevar offset global, usamos start=0/end=len(buf)
            start, end = 0, len(buf)

            # Clasifica tipo según origen (proporción de 'table' en el buffer)
            kinds = [k for (_, _, k) in buf_origin]
            table_ratio = sum(1 for k in kinds if k == "table") / max(1, len(kinds))
            ctype = "table" if table_ratio >= 0.8 else ("mixed" if 0 < table_ratio < 0.8 else "text")

            chunk = ChunkIR(
                id=ChunkIR.new_id(doc.doc_id, start, end, "hybrid-v1"),
                type=ctype,
                text=buf,
                meta={
                    "page_span": (
                        min(p for (p, _, _) in buf_origin),
                        max(p for (p, _, _) in buf_origin),
                    ),
                    "block_span": (
                        min(b for (_, b, _) in buf_origin),
                        max(b for (_, b, _) in buf_origin),
                    ),
                    "char_span": (start, end),
                    "reason": reason,
                    "overlap_applied": cfg.overlap_chars,
                },
            )
            chunks.append(chunk)

            # Preparar buffer con overlap
            if cfg.overlap_chars > 0 and len(buf) > cfg.overlap_chars:
                overlap_tail = buf[-cfg.overlap_chars:]
                buf = overlap_tail
                # Origen mínimo para el tail (conserva continuidad)
                if buf_origin:
                    last_p, last_b, _ = buf_origin[-1]
                    buf_origin = [(last_p, last_b, "paragraph")]
                else:
                    buf_origin = []
            else:
                buf = ""
                buf_origin = []

        while i < len(units):
            u = units[i]

            # Manejo de tablas aisladas
            if u["kind"] == "table" and cfg.table_policy == "isolate":
                # Si hay texto pendiente, emitirlo antes para no mezclar
                if buf.strip():
                    flush_chunk("pre-table")
                tchunk = ChunkIR(
                    id=ChunkIR.new_id(doc.doc_id, i, i, "hybrid-v1"),
                    type="table",
                    text=u["text"],
                    meta={
                        "page_span": (u["page_idx"], u["page_idx"]),
                        "block_span": (u["block_idx"], u["block_idx"]),
                        "char_span": (0, len(u["text"])),
                        "reason": "isolated-table",
                        "overlap_applied": 0,
                    },
                )
                chunks.append(tchunk)
                i += 1
                continue

            # Texto normal (o tablas en modo merge)
            candidate = (buf + ("\n\n" if buf else "") + u["text"]).strip()

            # Hasta target → aceptamos
            if len(candidate) <= cfg.target_chars:
                buf = candidate
                buf_origin.append((u["page_idx"], u["block_idx"], u["kind"]))
                i += 1
                continue

            # Entre target y max → empujamos uno más
            if len(candidate) <= cfg.max_chars:
                buf = candidate
                buf_origin.append((u["page_idx"], u["block_idx"], u["kind"]))
                i += 1
                continue

            # Excedimos max_chars → cortar “bonito” por oraciones
            cut_text = self._smart_cut(buf, cfg.target_chars, cfg.max_chars)
            if cut_text.strip():
                buf = cut_text
                flush_chunk("max-reached-cut")
            else:
                # Si no pudimos cortar de forma limpia, vaciamos crudo
                flush_chunk("max-reached-raw")

        # Flush final
        if buf.strip():
            flush_chunk("eof")

        return chunks

    # -------------------------- utilidades --------------------------

    def _smart_cut(self, text: str, target: int, hard_max: int) -> str:
        """
        Intenta cortar respetando oraciones, cercano a 'target'.
        - Si hay spaCy: usa doc.sents (garantizado con 'senter'/'sentencizer').
        - Si no: fallback regex con signos de puntuación.
        Retorna el segmento a emitir (el buffer restante se gestiona fuera).
        """
        if not text or len(text) <= hard_max:
            return text

        if self._nlp is not None and self.cfg.sentence_splitter in ("spacy", "auto"):
            doc = self._nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            sentences = self._regex_sentences(text)

        acc: List[str] = []
        total = 0
        for s in sentences:
            if total + len(s) > hard_max:
                break
            acc.append(s)
            total += len(s) + 1  # + espacio
            if total >= target:
                break

        if not acc:
            # No se pudo cortar "bonito"; devolvemos los primeros hard_max chars
            return text[:hard_max]

        return " ".join(acc).strip()

    def _regex_sentences(self, text: str) -> List[str]:
        """
        Segmentación simple por regex: divide por '.', '!', '?', '…'
        procurando conservar mayúsculas iniciales y no perder separadores.
        """
        parts = _SENT_SPLIT_REGEX.split(text)
        if len(parts) <= 1:
            # Fallback aún más simple si no hay separadores reconocibles
            return [p.strip() for p in re.split(r"[\.!?…]\s+", text) if p.strip()]
        return [p.strip() for p in parts if p and p.strip()]
