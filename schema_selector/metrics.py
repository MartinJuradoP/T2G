# -*- coding: utf-8 -*-
"""
metrics.py — Métricas de calidad para la selección adaptativa.

Se exponen helpers para:
- cobertura (% chunks con top_domain)
- tasa de ambigüedad (doc/chunk)
- shrink del espacio de entidades (proxy de ahorro)
"""
from __future__ import annotations
from typing import Dict, Any
from .schemas import SchemaSelection


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
    Aproximación simple: toma el promedio de #entity_types propuestos
    en top-1 dominio por chunk y compara contra baseline_types.
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
