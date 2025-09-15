# sentence_filter/schemas.py
# -*- coding: utf-8 -*-
"""
Modelos para la etapa Sentence/Filter (Chunks → Oraciones)
- DocumentSentences: colección de oraciones con trazabilidad a chunks
- SentenceIR: oración individual con offsets y meta de origen
"""

from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import hashlib
import time


class SentenceIR(BaseModel):
    """
    Oración atómica utilizable por subsistemas de IE (NER/RE) y normalización.

    Campos clave
    ------------
    - id: identificador estable (sha1 corto) sobre (doc_id, chunk_idx, sent_idx)
    - text: contenido textual de la oración
    - meta:
        * chunk_id: id del chunk de origen
        * chunk_idx: índice del chunk dentro de DocumentChunks
        * page_span: (first_page_idx, last_page_idx) heredado de chunk.meta
        * char_span_in_chunk: (start, end) offset en el texto del chunk
        * filters: dict con flags útiles (p.ej. "normalized": true)
    """
    id: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def new_id(doc_id: str, chunk_idx: int, sent_idx: int, strategy: str = "sentences-v1") -> str:
        raw = f"{doc_id}:{chunk_idx}:{sent_idx}:{strategy}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class DocumentSentences(BaseModel):
    """
    Colección de oraciones para un documento, con resumen de filtros aplicados.
    """
    doc_id: str
    created_at: float = Field(default_factory=lambda: time.time())
    strategy: str = "sentences-v1"
    version: str = "1.0"
    sentences: List[SentenceIR] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
