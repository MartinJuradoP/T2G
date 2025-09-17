# -*- coding: utf-8 -*-
"""
normalizer/metrics.py
=====================

Métricas para salidas de la etapa de Normalización (DocumentEntities).
Este módulo es **puro**: no escribe archivos, no imprime por consola;
solo calcula y retorna estructuras de datos listas para ser usadas
(en CLI, notebooks o tests).

Principales KPIs:
- merge_savings: ahorro por deduplicación (input_mentions vs entities).
- type_distribution: distribución por tipo de entidad.
- date_precision: desglose de precisión (year/month/day/unknown).
- money_currency: desglose por moneda y estadísticas de valores.
- email_domains / url_domains: top dominios (auditoría de normalización).
- person_single_token: sospechosos PERSON de un solo token (ruido).
- person_org_conflicts: (post-normalización) cuántas claves coinciden entre PERSON/ORG.
- key_uniqueness: cuán única es la clave canónica por tipo.

Todas las funciones aceptan un `DocumentEntities` ya parseado a dict
o al modelo Pydantic; internamente se trabaja sobre dict.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import math

try:
    # Si Pydantic está disponible y usan el modelo
    from .schemas import DocumentEntities  # type: ignore
except Exception:
    DocumentEntities = Any  # fallback typing


def _to_dict(doc: Any) -> Dict[str, Any]:
    """Admite Pydantic o dict plano."""
    if hasattr(doc, "model_dump"):
        return doc.model_dump(mode="json")
    if isinstance(doc, dict):
        return doc
    raise TypeError("DocumentEntities debe ser dict o Pydantic con model_dump().")


# ---------------------------------------------------------------------------
# KPIs básicos
# ---------------------------------------------------------------------------

def merge_savings(de_doc: Any) -> Dict[str, Any]:
    """
    Ahorro por deduplicación: compara menciones de entrada vs entidades finales.
    Requiere que meta.counters tenga 'input_mentions' y 'entities'.
    """
    d = _to_dict(de_doc)
    cnt = d.get("meta", {}).get("counters", {}) or {}
    input_mentions = int(cnt.get("input_mentions", 0))
    entities = int(cnt.get("entities", len(d.get("entities", []))))
    saved = max(0, input_mentions - entities)
    rate = (saved / input_mentions) if input_mentions else 0.0
    return {
        "input_mentions": input_mentions,
        "entities": entities,
        "saved": saved,
        "saved_rate": round(rate, 4),
    }


def type_distribution(de_doc: Any) -> Dict[str, int]:
    """Conteo por `entity.type`."""
    d = _to_dict(de_doc)
    c = Counter()
    for e in d.get("entities", []):
        c[str(e.get("type") or "OTHER")] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Desgloses por tipo
# ---------------------------------------------------------------------------

def date_precision_breakdown(de_doc: Any) -> Dict[str, int]:
    """Precisión de fechas: year/month/day/unknown."""
    d = _to_dict(de_doc)
    c = Counter()
    for e in d.get("entities", []):
        if e.get("type") == "DATE":
            prec = str(e.get("attrs", {}).get("precision") or "unknown").lower()
            c[prec] += 1
    return dict(c)


def money_currency_stats(de_doc: Any) -> Dict[str, Any]:
    """
    Dinero por moneda: cuenta, min/max/avg (ignorando None).
    Retorna: {currency: {"count": n, "min": x, "max": y, "avg": z}}
    """
    d = _to_dict(de_doc)
    buckets: Dict[str, List[float]] = defaultdict(list)
    for e in d.get("entities", []):
        if e.get("type") == "MONEY":
            cur = str(e.get("attrs", {}).get("currency") or "UNK")
            val = e.get("attrs", {}).get("normalized_value", None)
            if isinstance(val, (int, float)) and not math.isnan(val):
                buckets[cur].append(float(val))
    out: Dict[str, Any] = {}
    for cur, vals in buckets.items():
        if not vals:
            out[cur] = {"count": 0, "min": None, "max": None, "avg": None}
        else:
            out[cur] = {
                "count": len(vals),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "avg": float(sum(vals)/len(vals)),
            }
    return out


def email_domains_top(de_doc: Any, k: int = 10) -> List[Tuple[str, int]]:
    """Top dominios de email normalizados."""
    d = _to_dict(de_doc)
    c = Counter()
    for e in d.get("entities", []):
        if e.get("type") == "EMAIL":
            dom = str(e.get("attrs", {}).get("domain") or "").lower()
            if dom:
                c[dom] += 1
    return c.most_common(k)


def url_domains_top(de_doc: Any, k: int = 10) -> List[Tuple[str, int]]:
    """Top dominios de URL normalizados."""
    d = _to_dict(de_doc)
    c = Counter()
    for e in d.get("entities", []):
        if e.get("type") == "URL":
            host = str(e.get("attrs", {}).get("host") or "").lower()
            if host:
                c[host] += 1
    return c.most_common(k)


# ---------------------------------------------------------------------------
# Señales de ruido / calidad
# ---------------------------------------------------------------------------

def person_single_token_suspects(de_doc: Any) -> List[str]:
    """
    Lista de PERSON con nombre de un solo token (sospechosos de ser ORG/OTHER).
    Retorna las superficies `name` (o attrs.raw si no hay name).
    """
    d = _to_dict(de_doc)
    out: List[str] = []
    for e in d.get("entities", []):
        if e.get("type") == "PERSON":
            name = str(e.get("name") or e.get("attrs", {}).get("raw") or "").strip()
            if name and len(name.split()) == 1:
                out.append(name)
    return out


def person_org_conflicts(de_doc: Any) -> int:
    """
    Post-normalización: si aún coexistieran PERSON y ORG con la misma clave,
    contamos cuántas claves canónicas (slug) aparecen en ambos tipos.
    (Debería tender a 0 con el resolver de conflictos del normalizer).
    """
    d = _to_dict(de_doc)
    people_keys = set()
    org_keys = set()
    for e in d.get("entities", []):
        t = e.get("type")
        a = e.get("attrs", {}) or {}
        if t == "PERSON":
            k = str(a.get("person_key") or (e.get("name") or "").lower())
            if k:
                people_keys.add(k)
        elif t == "ORG":
            k = str(a.get("org_key") or a.get("org_core") or (e.get("name") or "").lower())
            if k:
                org_keys.add(k)
    return len(people_keys.intersection(org_keys))


def key_uniqueness(de_doc: Any) -> Dict[str, float]:
    """
    Proporción de **claves únicas** por tipo (1.0 = todas las claves distintas).
    Útil para revisar si la canonicalización está siendo efectiva.
    """
    d = _to_dict(de_doc)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for e in d.get("entities", []):
        t = str(e.get("type") or "OTHER")
        a = e.get("attrs", {}) or {}
        key = None
        if t == "EMAIL":
            key = a.get("email_norm")
        elif t == "URL":
            key = a.get("url_norm")
        elif t == "DATE":
            key = a.get("value_iso")
        elif t == "MONEY":
            key = f"{a.get('normalized_value','NA')}::{a.get('currency','UNK')}"
        elif t == "ID":
            key = a.get("normalized_id")
        elif t == "ORG":
            key = a.get("org_key") or a.get("org_core") or (e.get("name") or "").lower()
        elif t == "PERSON":
            key = a.get("person_key") or (e.get("name") or "").lower()
        elif t == "LOC":
            key = a.get("loc_key") or (e.get("name") or "").lower()
        elif t in ("TITLE", "DEGREE"):
            key = (e.get("value") or e.get("name") or "").lower()
        else:
            key = (e.get("name") or "").lower()
        if key:
            buckets[t].append(str(key))

    out: Dict[str, float] = {}
    for t, keys in buckets.items():
        if not keys:
            out[t] = 1.0
        else:
            uniq = len(set(keys)) / float(len(keys))
            out[t] = round(uniq, 4)
    return out


def summary(de_doc: Any) -> Dict[str, Any]:
    """
    Resumen compacto para dashboards/CLI.
    """
    return {
        "merge_savings": merge_savings(de_doc),
        "type_distribution": type_distribution(de_doc),
        "date_precision": date_precision_breakdown(de_doc),
        "money_currency": money_currency_stats(de_doc),
        "email_domains_top5": email_domains_top(de_doc, 5),
        "url_domains_top5": url_domains_top(de_doc, 5),
        "person_single_token_suspects": person_single_token_suspects(de_doc),
        "person_org_conflicts": person_org_conflicts(de_doc),
        "key_uniqueness": key_uniqueness(de_doc),
    }
