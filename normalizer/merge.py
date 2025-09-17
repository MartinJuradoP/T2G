# -*- coding: utf-8 -*-
"""
normalizer/merge.py
===================

Estrategias de *merge/dedupe* para entidades normalizadas.

Flujo lógico:
1) La normalización produce *prototipos* (NormalizedEntity) por mención.
2) Aquí consolidamos prototipos que representan la misma entidad lógica
   usando una *clave canónica* (org_key, person_key, email_norm, url_norm, value_iso, etc.).
3) Si hay conflicto de tipos (ej. misma clave aparece como ORG y LOC),
   resolvemos con una *preferencia de tipo* (_type_rank) y confianza.

Este módulo es agnóstico del dominio; la clave canónica la define el caller a través
de `key_fn(proto) -> str`. La política de preferencia de tipos se puede ajustar
editando `PREFER_ORDER`.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Callable

from .schemas import NormalizedEntity

# ---------------------------------------------------------------------------
# Preferencia de tipos cuando hay conflicto para la misma clave
# (mayor puntaje = más fuerte/preferido)
# ---------------------------------------------------------------------------
PREFER_ORDER = {
    # Valores inequívocos primero
    "MONEY":   100,
    "DATE":    100,
    "EMAIL":   95,
    "URL":     95,
    "ID":      90,
    # Nominales con mayor poder semántico
    "ORG":     80,   # ORG por encima de LOC — (ej. "Beta Ltd" como ORG vs LOC)
    "PERSON":  70,
    "LOC":     60,
    "TITLE":   50,
    "DEGREE":  45,
    "PRODUCT": 40,
    # Fallback
    "OTHER":   10,
}

def _type_rank(t: str) -> int:
    """Devuelve el puntaje de preferencia del tipo de entidad."""
    return PREFER_ORDER.get(t, 0)


def _merge_group(group: List[NormalizedEntity]) -> NormalizedEntity:
    """
    Funde un grupo de prototipos con la misma clave:
    - Ordena por (preferencia de tipo, confianza) desc.
    - El primer elemento se toma como *head* (representante).
    - Concatena y deduplica `mentions` en el head.
    - Mantiene attrs/name/value del head (estrategia simple y efectiva).
      (Si quieres combinaciones más finas, aquí es el punto para hacerlo.)
    """
    if not group:
        raise ValueError("Grupo vacío en merge")

    group.sort(key=lambda x: (_type_rank(x.type), x.conf), reverse=True)
    head = group[0]

    merged_mentions = []
    for e in group:
        merged_mentions.extend(e.mentions)

    # Deduplicación estable de trazabilidad
    head.mentions = list(dict.fromkeys(merged_mentions))
    return head


def merge_by_key(entities: List[NormalizedEntity], key_fn: Callable[[NormalizedEntity], str]) -> List[NormalizedEntity]:
    """
    Fusión por *clave canónica*:

    Args:
        entities: lista de prototipos a consolidar.
        key_fn:   función que, dada una entidad, devuelve la clave canónica (p. ej.):
                  - ORG     -> e.attrs["org_key"]
                  - PERSON  -> e.attrs["person_key"]
                  - DATE    -> e.attrs["value_iso"]
                  - MONEY   -> f"{value}::{currency}"
                  - EMAIL   -> e.attrs["email_norm"]
                  - URL     -> e.attrs["url_norm"]
                  Si la clave sale vacía, usamos "__raw::<name|value>" como fallback.

    Returns:
        Lista consolidada (una entidad por clave).
    """
    if not entities:
        return []

    buckets: Dict[str, List[NormalizedEntity]] = defaultdict(list)
    for e in entities:
        try:
            key = key_fn(e) or ""
        except Exception:
            key = ""
        if not key:
            # Fallback crudo (lo suficientemente estable para reducir duplicados evidentes)
            key = f"__raw::{(e.name or e.value or '').strip().lower()}"
        buckets[key].append(e)

    merged: List[NormalizedEntity] = []
    for _, group in buckets.items():
        merged.append(_merge_group(group) if len(group) > 1 else group[0])

    return merged
