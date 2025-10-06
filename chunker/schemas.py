# -*- coding: utf-8 -*-
"""
schemas.py — Esquemas (Pydantic v2) del HybridChunker

Contratos de datos para la etapa de chunking:
- Entrada efectiva: DocumentIR+Topics (dict/JSON) producido por Contextizer-DOC.
- Salida: DocumentChunks (JSON) con trazabilidad, métricas y contexto heredado.

Diseño:
- Contratos estables y auditables.
- Metadatos suficientes para etapas posteriores (selector, mentions, grafo).
- Fallbacks seguros: campos opcionales con defaults razonables.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChunkSourceSpan(BaseModel):
    """Trazabilidad de procedencia del texto (página y lista de índices de bloque)."""
    page_number: int = Field(..., ge=1)
    block_indices: List[int] = Field(default_factory=list)


class TopicHints(BaseModel):
    """Pistas temáticas heredadas/derivadas desde topics_doc (nivel documento)."""
    inherited_topic_ids: List[int] = Field(default_factory=list)
    inherited_keywords: List[str] = Field(default_factory=list)
    # id de tópico global → score (p. ej., similitud coseno o score híbrido)
    topic_affinity: Dict[str, float] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Unidad semántica estable (≤ max_tokens aprox.) con trazabilidad y métricas."""
    chunk_id: str
    doc_id: str
    order: int = Field(..., ge=0, description="Posición global en el documento")
    text: str
    char_len: int
    est_tokens: int
    source_spans: List[ChunkSourceSpan] = Field(default_factory=list)
    topic_hints: TopicHints = Field(default_factory=TopicHints)
    # Métricas por chunk (cohesión, redundancia, salud, etc.)
    scores: Dict[str, float] = Field(default_factory=dict)
    # Metadatos locales opcionales (embedding y/o idioma detectado/heredado)
    meta_local: Dict[str, Any] = Field(default_factory=dict)


class ChunkingMeta(BaseModel):
    """Metadatos globales del proceso de chunking."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)         # Config efectiva usada
    topics_doc: Optional[Dict[str, Any]] = None                  # Copia literal de topics_doc
    stats: Dict[str, Any] = Field(default_factory=dict)          # Estadísticas y agregados


class DocumentChunks(BaseModel):
    """Raíz de salida del chunker con lista ordenada de chunks y meta global."""
    doc_id: str
    source_path: Optional[str] = None
    mime: Optional[str] = None
    lang: Optional[str] = None
    chunks: List[Chunk]
    meta: ChunkingMeta


class ChunkingConfig(BaseModel):
    """
    Configuración del HybridChunker (resiliente y reproducible).

    Longitud:
    - max_tokens/min_chars/max_chars controlan el empaquetado.

    Segmentación:
    - prefer_headings + heading_patterns guían la pre-segmentación por estructura.

    Embeddings:
    - use_embeddings guarda/usa vectores para métricas y afinidad de tópicos.
    - save_embeddings decide si persistir embeddings en meta_local.

    spaCy / idioma:
    - spacy_model/spacy_model_alt y detect_lang controlan el split de oraciones.

    Afinidad de tópicos:
    - topic_affinity_topk limita cuántos tópicos globales se heredan.
    - topic_affinity_blend mezcla coseno (embeddings) y Jaccard (léxico).

    Scoring interno:
    - alpha_cos_redundancy y beta_jaccard_redundancy mezclan redundancias.
    - w_cohesion y w_novelty ponderan el cómputo de chunk_health.
    """
    # Longitud objetivo y seguridad
    max_tokens: int = 2048
    min_chars: int = 280
    max_chars: int = 9000  # límite duro de seguridad

    # Segmentación por estructura
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

    # Embeddings (para métricas y afinidad de tópicos)
    use_embeddings: bool = True
    save_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # spaCy / idioma (fallback robusto)
    spacy_model: str = "es_core_news_sm"
    spacy_model_alt: str = "en_core_web_sm"
    detect_lang: bool = True
    lang_hint: str = "es"

    # Afinidad con tópicos document-level
    topic_affinity_topk: int = 3
    topic_affinity_blend: float = Field(0.7, ge=0.0, le=1.0)  # 0: solo Jaccard, 1: solo coseno

    # Scoring interno y normalizaciones
    alpha_cos_redundancy: float = Field(0.7, ge=0.0, le=1.0)      # peso coseno en redundancia mixta
    beta_jaccard_redundancy: float = Field(0.3, ge=0.0, le=1.0)   # peso Jaccard en redundancia mixta
    w_cohesion: float = Field(0.6, ge=0.0, le=1.0)                # peso cohesión en chunk_health
    w_novelty: float = Field(0.4, ge=0.0, le=1.0)                 # peso novedad (1 - redundancia) en chunk_health

    # Determinismo
    seed: int = 42
