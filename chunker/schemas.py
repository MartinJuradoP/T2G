# -*- coding: utf-8 -*-
"""
schemas.py — Esquemas (Pydantic v2) del HybridChunker

Entrada:  DocumentIR+Topics (JSON)  → ver contextizer salida
Salida:   DocumentChunks (JSON)

Principios:
- Contratos claros y estables.
- Metadatos ricos para trazabilidad.
- Transporte de contexto: se conserva topics_doc a nivel documento y
  se generan topic_hints a nivel chunk para la siguiente etapa.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


class ChunkSourceSpan(BaseModel):
    """Rango de procedencia para trazabilidad (página y bloque)."""
    page_number: int = Field(..., ge=1)
    block_indices: List[int] = Field(default_factory=list)


class TopicHints(BaseModel):
    """Pistas de tópico para el chunk (heredadas/derivadas del contexto)."""
    inherited_topic_ids: List[int] = Field(default_factory=list)
    inherited_keywords: List[str] = Field(default_factory=list)
    # Scoring soft de afinidad con tópicos globales (id → score)
    topic_affinity: Dict[str, float] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Unidad semántica estable (≤ max_tokens aprox.)."""
    chunk_id: str
    doc_id: str
    order: int = Field(..., ge=0, description="Posición global del chunk en el documento")
    text: str
    char_len: int
    est_tokens: int
    source_spans: List[ChunkSourceSpan] = Field(default_factory=list)
    topic_hints: TopicHints = Field(default_factory=TopicHints)
    # Campo opcional para scores internos (cohesión, redundancia, etc.)
    scores: Dict[str, float] = Field(default_factory=dict)


class ChunkingMeta(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    # Copiamos el contexto global del documento para transporte a etapas siguientes
    topics_doc: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunks(BaseModel):
    """Salida principal del chunker."""
    doc_id: str
    source_path: Optional[str] = None
    mime: Optional[str] = None
    lang: Optional[str] = None
    chunks: List[Chunk]
    meta: ChunkingMeta


class ChunkingConfig(BaseModel):
    """
    Configuración del HybridChunker.
    """
    # Longitud objetivo
    max_tokens: int = 2048
    min_chars: int = 280
    max_chars: int = 9000  # hard bound de seguridad
    # Segmentación
    prefer_headings: bool = True
    heading_patterns: List[str] = Field(
        default_factory=lambda: [
            r"^\s*(introducción|resumen|abstract|objetivos?)\b.*$",
            r"^\s*(síntomas?|symptoms?)\b.*$",
            r"^\s*(tratamiento|treatment|terapia)\b.*$",
            r"^\s*(discusión|discussion)\b.*$",
            r"^\s*(conclusiones?|conclusión|conclusions?)\b.*$",
            r"^\s*(complicaciones?|limitations?)\b.*$",
            r"^\s*(marco te[oó]rico|background)\b.*$",
            r"^\s*(m[eé]todos?|metodolog[ií]a|methods?)\b.*$",
        ]
    )
    # Embeddings (para cohesión/redundancia y topic_affinity)
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    # Idioma spaCy para sentence split (fallback robusto)
    spacy_model: str = "es_core_news_sm"
    # Afinidad con tópicos globales
    topic_affinity_topk: int = 3
    # Determinismo
    seed: int = 42
