# -*- coding: utf-8 -*-
"""
contextizer — Subsistema de contextualización semántica
=======================================================

Rol
---
Este paquete implementa el **Contextizer** del proyecto T2G.
Su propósito es enriquecer la representación intermedia (`DocumentIR`)
con **tópicos semánticos**, usando embeddings (SentenceTransformers)
y modelado de tópicos (BERTopic).

Módulos incluidos:
- contextizer.py → lógica procedural (doc-level / chunk-level)
- schemas.py     → contratos Pydantic (TopicModelConfig, TopicsDocMeta, etc.)
- utils.py       → utilidades (normalización, seeds, caché de embeddings)
- metrics.py     → métricas de evaluación
- models.py      → wrapper de modelos BERTopic
"""

from __future__ import annotations
import logging

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# API pública del paquete
# ---------------------------------------------------------------------------
from .contextizer import route_contextizer_doc, route_contextizer_chunks
from .schemas import TopicModelConfig, TopicsDocMeta, TopicItem, TopicsChunksMeta, ChunkTopic
from .metrics import coverage, outlier_rate, topic_size_stats, keywords_diversity
from .utils import prepare_text_for_topic, set_global_seeds

__all__ = [
    # Funciones principales
    "run_contextizer_on_doc",

    # Schemas / contratos
    "TopicModelConfig",
    "TopicsDocMeta",
    "TopicItem",
    "TopicsChunksMeta",
    "ChunkTopic",

    # Métricas
    "coverage",
    "outlier_rate",
    "topic_size_stats",
    "keywords_diversity",

    # Utilidades
    "prepare_text_for_topic",
    "set_global_seeds",
]
