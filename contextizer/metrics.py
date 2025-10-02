# -*- coding: utf-8 -*-
"""
metrics.py — Métricas de evaluación del Contextizer
===================================================

Rol
---
Define métricas para evaluar la calidad de la contextualización semántica.
Estas métricas se calculan a partir de las estructuras de salida definidas en
`schemas.py` (TopicsDocMeta, TopicItem, etc.).

Principios
----------
- Operar sobre los contratos Pydantic, no sobre dicts sueltos.
- Métricas simples pero útiles para monitoreo de calidad.
"""

from __future__ import annotations
from statistics import median
from typing import Dict, List

# Importamos las clases reales del schemas (NO alias)
from .schemas import TopicsDocMeta, TopicItem


def coverage(dt: TopicsDocMeta) -> float:
    """
    Proporción de chunks asignados a algún tópico (no outlier).
    """
    total = len(dt.topics)
    if total == 0:
        return 0.0
    assigned = sum(1 for t in dt.topics if t.topic_id != -1)
    return round(assigned / total, 6)


def outlier_rate(dt: TopicsDocMeta) -> float:
    """
    Proporción de tópicos clasificados como outliers.
    """
    total = len(dt.topics)
    if total == 0:
        return 0.0
    return round(sum(1 for t in dt.topics if t.topic_id == -1) / total, 6)


def topic_size_stats(dt: TopicsDocMeta) -> Dict[str, float]:
    """
    Estadísticas de tamaño de tópicos (min, median, p95).
    """
    sizes = [t.count for t in dt.topics if t.topic_id != -1]
    if not sizes:
        return {"min": 0.0, "median": 0.0, "p95": 0.0}
    sizes_sorted = sorted(sizes)
    p95_idx = max(0, int(0.95 * (len(sizes_sorted) - 1)))
    return {
        "min": float(min(sizes_sorted)),
        "median": float(median(sizes_sorted)),
        "p95": float(sizes_sorted[p95_idx]),
    }


def keywords_diversity(dt: TopicsDocMeta) -> float:
    """
    Diversidad de keywords: proporción de keywords únicas sobre el total.
    """
    all_kws: List[str] = []
    for t in dt.topics:
        all_kws.extend(t.keywords)
    if not all_kws:
        return 0.0
    unique = len(set(all_kws))
    return round(unique / len(all_kws), 6)
