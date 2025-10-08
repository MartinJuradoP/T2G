# -*- coding: utf-8 -*-
"""
schemas.py — Contratos Pydantic y Configuración (Contextizer)
=============================================================

Rol
---
Define los modelos de configuración y contratos de salida para la etapa de
Contextualización (doc-level y chunk-level), centralizando la configuración
y evitando duplicaciones entre módulos.

Principios
----------
- Pydantic v2 con validación estricta.
- Tolerante a atributos extras en JSON (extra="ignore") para compatibilidad.
- Diseñado para inyección en `meta` del IR y extensión de `DocumentChunks`.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------
# Configuración del modelo de tópicos
# ---------------------------------------------------------------------
class TopicModelConfig(BaseModel):
    """Configuración para embeddings + BERTopic + limpieza de keywords."""
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    # BERTopic / UMAP / HDBSCAN
    nr_topics: Optional[int] = None  # None/-1 → auto
    min_topic_size: int = 2
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    hdbscan_metric: str = "euclidean"

    # Reproducibilidad
    seed: int = 42

    # Keywords / fallback
    stopwords: Optional[List[str]] = None
    max_keywords_per_topic: int = 10
    fallback_max_keywords: int = 10
    min_token_len: int = 3

    # Caché opcional para embeddings
    cache_dir: Optional[str] = None

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Hybrid mode
    use_hybrid: bool = True
    use_keybert: bool = True
    fusion_weights: Optional[List[float]] = [0.5, 0.3, 0.2]  # tfidf, keybert, emb
    hybrid_eps: float = 0.25
    hybrid_min_samples: int = 2
    cache_embeddings: bool = True
    enable_mmr: bool = True
    max_batch_size: int = 64


# ---------------------------------------------------------------------
# Contratos de salida — doc-level
# ---------------------------------------------------------------------
class TopicItem(BaseModel):
    """Unidad canónica de un tópico detectado."""
    topic_id: int
    count: int = 0
    exemplar: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class TopicsDocMeta(BaseModel):
    """Resumen de tópicos a nivel documento (inyectado en IR.meta.topics_doc)."""
    reason: str
    created_at: str
    n_samples: int
    n_topics: int
    keywords_global: List[str] = Field(default_factory=list)
    topics: List[TopicItem] = Field(default_factory=list)
    outlier_ratio: Optional[float] = None
    metrics_ext: Optional[Dict[str, float]] = None

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------
# Contratos de salida — chunk-level
# ---------------------------------------------------------------------
class ChunkTopic(BaseModel):
    """Anotación de tópico asignado a un chunk individual."""
    topic_id: int
    keywords: List[str] = Field(default_factory=list)
    prob: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class TopicsChunksMeta(BaseModel):
    """Resumen de tópicos a nivel chunks (inyectado en meta.topics_chunks)."""
    reason: str
    created_at: str
    n_samples: int
    n_topics: int
    keywords_global: List[str] = Field(default_factory=list)
    topics: List[TopicItem] = Field(default_factory=list)
    metrics_ext: Optional[Dict[str, float]] = None

    model_config = ConfigDict(extra="ignore")
