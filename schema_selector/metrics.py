# -*- coding: utf-8 -*-
"""
metrics.py — Métricas de calidad para Adaptive Schema Selector 2.0 (ampliado)

Incluye métricas operacionales del selector:
- selection_coverage: % de chunks con top_domain asignado.
- ambiguity_rate: % de chunks marcados ambiguos.
- entity_space_shrink: proxy de ahorro por reducción de espacio de entidades.
- domain_confidence_stats: estadísticas de confianza chunk-level.
- explanation_coverage: % de chunks con explicación no vacía.
- calibration_gap: diferencia promedio entre confianza y 'ventaja de score' (proxy de calibración).
"""

from __future__ import annotations
from typing import Dict, Any, List
from .schemas import SchemaSelection
import numpy as np


def selection_coverage(sel: SchemaSelection) -> float:
    chunks = sel.chunks or []
    if not chunks:
        return 0.0
    ok = sum(1 for ch in chunks if ch.top_domain)
    return ok / float(len(chunks))


def ambiguity_rate(sel: SchemaSelection) -> float:
    chunks = sel.chunks or []
    if not chunks:
        return 0.0
    amb = sum(1 for ch in chunks if ch.ambiguous)
    return amb / float(len(chunks))


def entity_space_shrink(sel: SchemaSelection, baseline_types: int = 30) -> float:
    """
    Aproximación: promedio de #entity_types en el top-1 dominio por chunk vs baseline.
    """
    chunks = sel.chunks or []
    if not chunks:
        return 0.0
    tot = 0
    cnt = 0
    for ch in chunks:
        if not ch.domain_scores:
            continue
        top = ch.domain_scores[0]
        tot += len(top.entity_type_scores or [])
        cnt += 1
    if cnt == 0:
        return 0.0
    avg = tot / float(cnt)
    return max(0.0, min(1.0, 1.0 - (avg / float(baseline_types))))


def domain_confidence_stats(sel: SchemaSelection) -> Dict[str, float]:
    confs = [ch.schema_confidence for ch in (sel.chunks or [])]
    if not confs:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(np.mean(confs)),
        "std": float(np.std(confs)),
        "min": float(np.min(confs)),
        "max": float(np.max(confs)),
    }


def explanation_coverage(sel: SchemaSelection) -> float:
    chunks = sel.chunks or []
    if not chunks:
        return 0.0
    k = sum(1 for ch in chunks if (ch.explanation or "").strip())
    return k / float(len(chunks))


def calibration_gap(sel: SchemaSelection) -> float:
    """
    Proxy de calibración: para cada chunk, medimos:
      margin = (score_top - score_2nd) / max(1e-6, |score_top|)
      gap = |confidence - sigmoid(margin_z)|
    y promediamos. (No es estadísticamente estricto pero sirve de guardrail.)
    """
    z = []
    for ch in sel.chunks or []:
        if not ch.domain_scores or len(ch.domain_scores) < 2:
            continue
        top = ch.domain_scores[0].score
        snd = ch.domain_scores[1].score
        margin = (top - snd) / (abs(top) + 1e-6)
        # Sigmoid simple
        p = 1.0 / (1.0 + np.exp(-5.0 * margin))
        z.append(abs(ch.schema_confidence - p))
    return float(np.mean(z)) if z else 0.0
