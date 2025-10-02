# -*- coding: utf-8 -*-
"""
schemas.py — Modelos y contratos de la IR (Intermediate Representation)

Propósito
---------
Definir una IR estandarizada (JSON/MD-like) para cualquier documento de entrada
(PDF, DOCX, imágenes), preservando texto, layout básico y tablas.

Esta versión MEJORADA incorpora:
- Documentación más exhaustiva por clase/campo.
- Campos de trazabilidad enriquecidos (source_lines, provenance notes).
- Soporte opcional para métricas de layout, fusión de párrafos y OCR.
- Extensibilidad: bloques genéricos (Block = Union[…]) y espacios reservados.
- Compatibilidad hacia atrás: contratos JSON dict siguen funcionando.
- Buenas prácticas: validaciones suaves con Pydantic, defaults sensatos.

Principios
----------
- Backward-compatible: el Parser puede seguir enviando dicts (model_dump()) en PageIR.blocks.
- Extensible: permite nuevos tipos de bloque sin romper a consumidores.
- Trazable: incluye provenance, confidencias y metadatos de extracción/normalización.
- Robusto: validaciones suaves (Pydantic) y defaults sensatos.

Notas
-----
- Los bloques pueden serializarse como dicts (para velocidad) o instancias de modelos Pydantic.
- `DocumentIR.new_id()` genera un ID estable legible.
"""

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import uuid
import hashlib
import time

# -------------------------------------------------------------------
# Tipos base y alias
# -------------------------------------------------------------------

BlockType = Literal[
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "footer",
    "header",
    "unknown",
]

LanguageTag = Literal[
    "und",  # undefined
    "es",
    "en",
    "es+en",
    "pt",
    "fr",
    "de",
]

# -------------------------------------------------------------------
# Utilidades de layout / OCR / provenance
# -------------------------------------------------------------------

class BBox(BaseModel):
    """Caja delimitadora (bounding box) en coordenadas absolutas de página."""
    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1", "y1")
    @classmethod
    def _positive_extent(cls, v):
        # Permitimos 0, pero típicamente deben ser > x0/y0
        return v

class LayoutInfo(BaseModel):
    """Metadatos de layout opcionales a nivel de bloque."""
    bbox: Optional[BBox] = None
    page_rotation: Optional[int] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    # futuro: color, weight, column_index

class OCRInfo(BaseModel):
    """Metadatos de OCR opcionales a nivel de bloque."""
    engine: Optional[str] = None          # p.ej. 'tesseract'
    lang: Optional[str] = None            # p.ej. 'spa', 'spa+eng'
    conf: Optional[float] = None          # confianza 0..1 si disponible
    dpi: Optional[int] = None

class Provenance(BaseModel):
    """
    Procedencia de un bloque o documento:
    - extractor: componente usado (pdfplumber, python-docx, tesseract, etc.)
    - stage: etapa del pipeline que produjo el bloque (parser, ocr-fallback, normalizer, etc.)
    - notes: comentarios adicionales, ej. source_lines=[2,3,4] tras fusión
    """
    extractor: Optional[str] = None
    stage: Optional[str] = None
    notes: Optional[str] = None

# -------------------------------------------------------------------
# Bloques de Tabla y Texto
# -------------------------------------------------------------------

class TableCell(BaseModel):
    """Celda de tabla con posición (fila/columna) y texto normalizado."""
    row: int
    col: int
    text: str = ""
    bbox: Optional[BBox] = None
    conf: Optional[float] = None  # confianza OCR por celda si aplica

class TableBlock(BaseModel):
    """Bloque de tabla en la IR; contiene celdas (row/col)."""
    type: Literal["table"] = "table"
    cells: List[TableCell] = Field(default_factory=list)
    layout: Optional[LayoutInfo] = None
    ocr: Optional[OCRInfo] = None
    prov: Optional[Provenance] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Forma aproximada (n_rows, n_cols) calculada de las celdas."""
        if not self.cells:
            return (0, 0)
        max_r = max(c.row for c in self.cells)
        max_c = max(c.col for c in self.cells)
        return (max_r + 1, max_c + 1)

class TextBlock(BaseModel):
    """
    Bloque de texto (párrafo, heading, list item, etc.).
    """
    type: Literal["heading", "paragraph", "list_item", "footer", "header", "unknown"]
    text: str
    level: Optional[int] = None            # solo aplica si type == 'heading'
    layout: Optional[LayoutInfo] = None
    ocr: Optional[OCRInfo] = None
    prov: Optional[Provenance] = None
    source_lines: Optional[List[int]] = None  # índices de líneas crudas fusionadas

class FigureBlock(BaseModel):
    """Bloque para figuras/imágenes con caption opcional."""
    type: Literal["figure"] = "figure"
    caption: Optional[str] = None
    layout: Optional[LayoutInfo] = None
    ocr: Optional[OCRInfo] = None
    prov: Optional[Provenance] = None

# Alias de unión para escribir PageIR.blocks con tipos fuertes
Block = Union[TextBlock, TableBlock, FigureBlock, Dict[str, Any]]

# -------------------------------------------------------------------
# Páginas y Documento
# -------------------------------------------------------------------

class PageIR(BaseModel):
    """IR a nivel de página: dimensiones, bloques estructurados y metadatos."""
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    blocks: List[Block] = Field(default_factory=list)
    lang: LanguageTag = "und"
    meta: Dict[str, Any] = Field(default_factory=dict)  # métricas (fusion_rate, layout_loss, etc.)

class DocumentIR(BaseModel):
    """
    IR a nivel documento:
    - doc_id: ID estable (útil para correlación e indexación).
    - source_path: ruta original del documento.
    - mime: MIME detectado.
    - pages: lista de PageIR.
    - meta: metadatos (sha256, doc_type heurístico, size_bytes, page_count, etc.).
    """
    doc_id: str
    source_path: str
    mime: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    lang: LanguageTag = "und"
    meta: Dict[str, Any] = Field(default_factory=dict)
    pages: List[PageIR] = Field(default_factory=list)
    prov: Optional[Provenance] = None

    @staticmethod
    def new_id() -> str:
        """Genera un identificador corto y humano-legible (DOC-XXXX...)."""
        return f"DOC-{uuid.uuid4().hex[:12].upper()}"

# -------------------------------------------------------------------
# Extensiones opcionales (para subsistemas posteriores)
# -------------------------------------------------------------------

class CharSpan(BaseModel):
    """Span de caracteres en un texto (start, end)."""
    start: int
    end: int

class BlockRef(BaseModel):
    """Referencia a un bloque dentro del documento (page/block_index)."""
    page_index: int
    block_index: int
    kind: BlockType

class TextCarrier(BaseModel):
    """
    Contenedor ligero que vincula un texto a su procedencia física (página/bloque)
    y a un span relativo, útil en normalización y evaluaciones posteriores.
    """
    text: str
    block_ref: Optional[BlockRef] = None
    char_span: Optional[CharSpan] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

# -------------------------------------------------------------------
# Chunks (Subsistema 2: HybridChunker)
# -------------------------------------------------------------------

ChunkType = Literal["text", "table", "mixed"]

class ChunkIR(BaseModel):
    """
    Representación de un chunk semántico estable.

    Campos clave
    ------------
    - chunk_id: identificador estable (sha1 corto) sobre (doc_id, start, end, strategy).
    - type: tipo dominante del contenido ("text" | "table" | "mixed").
    - text: contenido textual del chunk (tablas serializadas a texto plano).
    - meta: metadata operativa para trazabilidad y evaluación:
        * page_span: (first_page_idx, last_page_idx).
        * block_span: (first_block_idx, last_block_idx).
        * char_span: (start, end) relativo al buffer del chunker.
        * tokens_est: estimación de tokens.
        * reason: causa de corte (ej. "max-reached-cut", "isolated-table", "eof").
        * overlap_applied: solapamiento de caracteres aplicado.
        * strategy/version: identificadores del algoritmo de chunking.
    """
    chunk_id: str
    type: ChunkType = "text"
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def new_id(doc_id: str, start: int, end: int, strategy: str = "hybrid-v1") -> str:
        """Genera un identificador único reproducible por documento + offsets."""
        raw = f"{doc_id}:{start}:{end}:{strategy}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Aproximación simple de tokens (1 token ≈ 4 chars)."""
        return max(1, int(len(text) / 4))

class DocumentChunks(BaseModel):
    """Colección de chunks para un documento con metadatos de ejecución."""
    doc_id: str
    created_at: float = Field(default_factory=lambda: time.time())
    strategy: str = "hybrid-v1"
    version: str = "1.0"
    chunks: List[ChunkIR] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
