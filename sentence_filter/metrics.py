# sentence_filter/metrics.py
# -*- coding: utf-8 -*-
"""
Métricas básicas para la etapa Sentence/Filter
"""

from __future__ import annotations
from typing import Dict, Any
from statistics import mean
from .schemas import DocumentSentences


def sentence_length_stats(ds: DocumentSentences) -> Dict[str, Any]:
    lens = [len(s.text) for s in ds.sentences]
    if not lens:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}
    lens_sorted = sorted(lens)
    p95 = lens_sorted[int(0.95 * (len(lens_sorted) - 1))]
    return {
        "count": len(lens),
        "avg": int(mean(lens)),
        "min": min(lens),
        "max": max(lens),
        "p95": p95,
    }


def unique_ratio(ds: DocumentSentences) -> float:
    uniq = len({s.text for s in ds.sentences})
    total = len(ds.sentences)
    return (uniq / total) if total else 0.0
