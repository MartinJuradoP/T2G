# chunker/metrics.py
"""
Métricas básicas de calidad para HybridChunker.
"""

from __future__ import annotations
from typing import Dict, Any
from statistics import mean

from parser.schemas import DocumentChunks


def chunk_length_stats(dc: DocumentChunks) -> Dict[str, Any]:
    lens = [len(c.text) for c in dc.chunks]
    if not lens:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}

    def pctl(p: float) -> int:
        idx = max(0, min(len(lens) - 1, int(round(p * (len(lens) - 1)))))
        return sorted(lens)[idx]

    return {
        "count": len(lens),
        "avg": int(mean(lens)),
        "min": min(lens),
        "max": max(lens),
        "p95": pctl(0.95),
    }

def percent_within_threshold(dc: DocumentChunks, min_chars: int, max_chars: int) -> float:
    if not dc.chunks:
        return 0.0
    ok = sum(1 for c in dc.chunks if min_chars <= len(c.text) <= max_chars)
    return ok / len(dc.chunks)

def table_mix_ratio(dc: DocumentChunks) -> Dict[str, float]:
    total = len(dc.chunks) or 1
    t = sum(1 for c in dc.chunks if c.type == "table")
    m = sum(1 for c in dc.chunks if c.type == "mixed")
    x = sum(1 for c in dc.chunks if c.type == "text")
    return {"text": x/total, "mixed": m/total, "table": t/total}
