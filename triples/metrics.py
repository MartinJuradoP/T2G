# triples/metrics.py
# -*- coding: utf-8 -*-
"""
Métricas operables para la etapa Triples (dep.).
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple

try:
    from sentence_filter.schemas import DocumentSentences  # type: ignore
except Exception:
    # Tipado laxo si no está disponible el contrato real.
    from pydantic import BaseModel
    class DocumentSentences(BaseModel):  # type: ignore
        doc_id: str
        sentences: list
        meta: dict = {}

from .schemas import DocumentTriples


def triple_counts(ds: DocumentSentences, dt: DocumentTriples) -> Dict[str, float]:
    """Resumen de conteos clave y promedios."""
    total_sents = len(getattr(ds, "sentences", []))
    counters = dt.meta.get("counters", {})
    used_sents = int(counters.get("used_sents", 0))
    total_triples = int(counters.get("total_triples", len(dt.triples)))
    avg_per_used = float(counters.get("avg_triples_per_used_sent", total_triples / max(1, used_sents)))
    return {
        "total_sents": float(total_sents),
        "used_sents": float(used_sents),
        "total_triples": float(total_triples),
        "avg_triples_per_used_sent": round(avg_per_used, 3),
    }


def relation_distribution(dt: DocumentTriples, top_k: int = 20) -> Dict[str, int]:
    """Distribución de relaciones extraídas (top-k)."""
    c = Counter(t.relation for t in dt.triples)
    return dict(c.most_common(top_k))


def unique_ratio(dt: DocumentTriples) -> float:
    """Proporción de triples únicos exactos (S,R,O)."""
    seen = set()
    for t in dt.triples:
        seen.add((t.subject.lower(), t.relation.lower(), t.object.lower()))
    return round(len(seen) / max(1, len(dt.triples)), 4)
