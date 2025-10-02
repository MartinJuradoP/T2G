# -*- coding: utf-8 -*-
"""
metrics.py — Métricas operables del HybridChunker

Objetivo:
- Medir salud del chunking y su utilidad para etapas siguientes.
- Señales para tuning (max_tokens, min_chars, headings, etc.).

Métricas:
- chunk_length_stats: distribución de longitud (chars/tokens).
- coverage_rate: % de texto original cubierto (aprox por chars).
- boundary_alignment: % de límites que caen cerca de headings/listas.
- cohesion_vs_doc (promedio): cuán "contextual" es cada chunk.
- redundancy_p95: similitud inter-chunk (95p) — evitar duplicidad.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

from .schemas import DocumentChunks


def chunk_length_stats(dc: DocumentChunks) -> Dict[str, Any]:
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
    if not dc.chunks or not approx_doc_chars or approx_doc_chars <= 0:
        return 0.0
    covered = sum(c.char_len for c in dc.chunks)
    return float(min(1.0, covered / approx_doc_chars))


def boundary_alignment(dc: DocumentChunks) -> float:
    """
    Proxy simple: % de chunks cuyo primer span inicia en bloque índice 0 o en
    un índice con pocos caracteres (asume headings/listas).
    """
    if not dc.chunks:
        return 0.0
    aligns = 0
    total = 0
    for c in dc.chunks:
        if not c.source_spans:
            continue
        total += 1
        first = c.source_spans[0]
        aligns += 1 if (0 in first.block_indices or len(first.block_indices) <= 2) else 0
    return float(aligns / total) if total else 0.0


def cohesion_vs_doc_mean(dc: DocumentChunks) -> float:
    vals = [c.scores.get("cohesion_vs_doc", 0.0) for c in dc.chunks]
    return float(np.mean(vals)) if vals else 0.0


def redundancy_p95(dc: DocumentChunks) -> float:
    vals = [c.scores.get("max_redundancy", 0.0) for c in dc.chunks]
    return float(np.percentile(vals, 95)) if vals else 0.0


def summarize(dc: DocumentChunks, approx_doc_chars: int | None) -> Dict[str, Any]:
    return {
        "length": chunk_length_stats(dc),
        "coverage_rate": coverage_rate(dc, approx_doc_chars),
        "boundary_alignment": boundary_alignment(dc),
        "cohesion_vs_doc_mean": cohesion_vs_doc_mean(dc),
        "redundancy_p95": redundancy_p95(dc),
    }
