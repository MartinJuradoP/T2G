# -*- coding: utf-8 -*-
"""
metrics.py — Métricas operables del HybridChunker (versión mejorada)

Objetivo:
---------
- Medir salud global del proceso de chunking.
- Evaluar cobertura, cohesión y redundancia.
- Servir como feedback para tuning y auditoría del pipeline.

Nuevas métricas:
----------------
- chunk_health_mean: promedio de salud semántica de los chunks.
- semantic_coverage: % de chunks con cohesión alta (≥0.7).
- redundancy_flag_rate: % de chunks con redundancia normalizada alta (≥0.6).
- novelty_mean, lexical_density_mean, redundancy_norm_mean, type_token_ratio_mean.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from .schemas import DocumentChunks


# ---------------------- Estadísticas básicas ----------------------
def chunk_length_stats(dc: DocumentChunks) -> Dict[str, Any]:
    """Distribución de longitudes (chars / tokens)."""
    if not dc.chunks:
        return {"count": 0}
    chars = np.array([c.char_len for c in dc.chunks], dtype=float)
    toks = np.array([c.est_tokens for c in dc.chunks], dtype=float)
    return {
        "count": len(dc.chunks),
        "chars": {
            "min": float(np.min(chars)),
            "p25": float(np.percentile(chars, 25)),
            "p50": float(np.median(chars)),
            "p75": float(np.percentile(chars, 75)),
            "p95": float(np.percentile(chars, 95)),
            "max": float(np.max(chars)),
            "mean": float(np.mean(chars)),
        },
        "tokens": {
            "min": float(np.min(toks)),
            "p50": float(np.median(toks)),
            "p95": float(np.percentile(toks, 95)),
            "max": float(np.max(toks)),
            "mean": float(np.mean(toks)),
        },
    }


def coverage_rate(dc: DocumentChunks, approx_doc_chars: int | None) -> float:
    """Porcentaje del texto original cubierto por chunks."""
    if not dc.chunks or not approx_doc_chars or approx_doc_chars <= 0:
        return 0.0
    covered = sum(c.char_len for c in dc.chunks)
    return float(min(1.0, covered / approx_doc_chars))


def boundary_alignment(dc: DocumentChunks) -> float:
    """Proxy simple: % de chunks alineados con headings o bloques iniciales."""
    if not dc.chunks:
        return 0.0
    aligns, total = 0, 0
    for c in dc.chunks:
        if not c.source_spans:
            continue
        total += 1
        first = c.source_spans[0]
        aligns += 1 if (0 in first.block_indices or len(first.block_indices) <= 2) else 0
    return float(aligns / total) if total else 0.0


# ---------------------- Métricas semánticas ----------------------
def semantic_stats(dc: DocumentChunks) -> Dict[str, float]:
    """Promedios globales de métricas semánticas."""
    if not dc.chunks:
        return {}
    keys = [
        "cohesion_vs_doc",
        "chunk_health",
        "redundancy_norm",
        "novelty",
        "lexical_density",
        "type_token_ratio",
    ]
    return {
        k + "_mean": float(np.mean([c.scores.get(k, 0.0) for c in dc.chunks]))
        for k in keys
    }


def quality_flags(dc: DocumentChunks) -> Dict[str, float]:
    """Detecta posibles anomalías (redundancia, baja cohesión)."""
    if not dc.chunks:
        return {"semantic_coverage": 0.0, "redundancy_flag_rate": 0.0}
    coh = np.array([c.scores.get("cohesion_vs_doc", 0.0) for c in dc.chunks])
    red = np.array([c.scores.get("redundancy_norm", 0.0) for c in dc.chunks])
    return {
        "semantic_coverage": float(np.mean(coh >= 0.7)),   # proporción chunks coherentes
        "redundancy_flag_rate": float(np.mean(red >= 0.6)),  # redundancia excesiva
    }


# ---------------------- Cohesión / redundancia ----------------------
def cohesion_vs_doc_mean(dc: DocumentChunks) -> float:
    vals = [c.scores.get("cohesion_vs_doc", 0.0) for c in dc.chunks]
    return float(np.mean(vals)) if vals else 0.0


def redundancy_p95(dc: DocumentChunks) -> float:
    vals = [c.scores.get("max_redundancy", 0.0) for c in dc.chunks]
    return float(np.percentile(vals, 95)) if vals else 0.0


# ---------------------- Resumen global ----------------------
def summarize(dc: DocumentChunks, approx_doc_chars: int | None) -> Dict[str, Any]:
    """
    Retorna un resumen completo del desempeño del chunking:
    - Estadísticas de longitud
    - Cobertura
    - Alineación con headings
    - Métricas semánticas promedio
    - Indicadores de salud (coverage, redundancia, etc.)
    """
    if not dc.chunks:
        return {}

    sem = semantic_stats(dc)
    flags = quality_flags(dc)
    health_score = (sem.get("chunk_health_mean", 0) * 0.6) + (
        sem.get("cohesion_vs_doc_mean", 0) * 0.4
    )

    return {
        "length": chunk_length_stats(dc),
        "coverage_rate": coverage_rate(dc, approx_doc_chars),
        "boundary_alignment": boundary_alignment(dc),
        "cohesion_vs_doc_mean": cohesion_vs_doc_mean(dc),
        "redundancy_p95": redundancy_p95(dc),
        "semantic": sem,
        "quality_flags": flags,
        "health_summary": {
            "global_health_score": round(float(health_score), 4),
            "status": (
                "good"
                if health_score >= 0.75
                else "moderate"
                if health_score >= 0.5
                else "poor"
            ),
        },
    }
