# -*- coding: utf-8 -*-
"""
mentions/metrics.py — Métricas automáticas del subsistema Mentions (T2G)
=========================================================================

Calcula métricas por documento para integrarlas directamente en la salida JSON
de cada archivo generado por `llm_extractor.py`.

Las métricas se agregan como clave `"metrics"` dentro del JSON final.
"""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import math
import json
from schema_selector.registry import REGISTRY


# ============================================================
# 🔹 Utilidades básicas
# ============================================================
def entropy(values: list[str]) -> float:
    """Entropía de Shannon normalizada (0–1)."""
    total = len(values)
    if total == 0:
        return 0.0
    counts = Counter(values)
    probs = [v / total for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs) / math.log2(len(probs))


# ============================================================
# 📊 Función principal
# ============================================================
def compute_doc_metrics(mentions: list[dict]) -> dict:
    """Calcula métricas principales de un documento basado en las menciones extraídas."""
    if not mentions:
        return {
            "n_mentions": 0,
            "avg_confidence": 0.0,
            "repetition_rate": 0.0,
            "lexical_diversity": 0.0,
            "errors_rate": 0.0,
            "coverage_domains": 0.0,
            "top_entities": [],
            "top_domains": []
        }

    confs = [m.get("confidence", 0.0) for m in mentions]
    types = [m.get("type", "Unknown") for m in mentions]
    domains = [m.get("domain", "generic") for m in mentions]
    texts = [m.get("text", "").lower().strip() for m in mentions if m.get("text")]

    n_total = len(mentions)
    avg_conf = sum(confs) / max(1, len(confs))
    repetition_rate = 1 - (len(set(texts)) / max(1, len(texts)))
    lexical_div = entropy(texts)
    errors_rate = len([t for t in types if t == "Unknown"]) / max(1, n_total)

    # Cobertura de dominios definidos en registry
    registry_domains = {d.domain for d in REGISTRY.domains}
    coverage_domains = len(set(domains) & registry_domains) / len(registry_domains)

    return {
        "n_mentions": n_total,
        "avg_confidence": round(avg_conf, 3),
        "repetition_rate": round(repetition_rate, 3),
        "lexical_diversity": round(lexical_div, 3),
        "errors_rate": round(errors_rate, 3),
        "coverage_domains": round(coverage_domains, 3),
        "top_entities": Counter(types).most_common(5),
        "top_domains": Counter(domains).most_common(5)
    }


# ============================================================
# 🚀 Integración simple
# ============================================================
def attach_metrics_to_output(doc_json: dict) -> dict:
    """Adjunta métricas calculadas directamente al JSON final."""
    mentions = doc_json.get("mentions", [])
    metrics = compute_doc_metrics(mentions)
    doc_json["metrics"] = metrics
    return doc_json
