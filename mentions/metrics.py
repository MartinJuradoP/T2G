# -*- coding: utf-8 -*-
"""
mentions/metrics.py

Métricas prácticas para auditar la etapa Mentions.
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd


def entities_summary(doc_mentions: Dict[str, Any]) -> pd.DataFrame:
    """Conteo de entidades por etiqueta (usa canonical_label si existe)."""
    ents = doc_mentions.get("entities", [])
    if not ents:
        return pd.DataFrame(columns=["label", "count"]).set_index("label")
    df = pd.json_normalize(ents)
    lbl = df.get("canonical_label").fillna(df.get("label"))
    return lbl.value_counts().rename_axis("label").to_frame("count")


def relations_summary(doc_mentions: Dict[str, Any]) -> pd.DataFrame:
    """Conteo de relaciones por etiqueta."""
    rels = doc_mentions.get("relations", [])
    if not rels:
        return pd.DataFrame(columns=["label", "count"]).set_index("label")
    df = pd.json_normalize(rels)
    lbl = df.get("canonical_label").fillna(df.get("label"))
    return lbl.value_counts().rename_axis("label").to_frame("count")


def confidence_stats(doc_mentions: Dict[str, Any]) -> Dict[str, float]:
    """Describe de 'conf' en entidades y relaciones."""
    out: Dict[str, float] = {}
    for key in ("entities", "relations"):
        items = doc_mentions.get(key, [])
        if not items:
            continue
        conf = pd.Series([i.get("conf", 0.0) for i in items], dtype=float)
        out[f"{key}_count"] = int(conf.shape[0])
        out[f"{key}_conf_mean"] = float(conf.mean())
        out[f"{key}_conf_p50"] = float(conf.quantile(0.50))
        out[f"{key}_conf_p75"] = float(conf.quantile(0.75))
    return out
