# graph_builder/metrics.py
# -*- coding: utf-8 -*-
"""
metrics.py — Registro de métricas del Graph Builder (T2G)
=========================================================

Funcionalidad:
--------------
- Asegura la existencia del archivo de métricas (outputs_graph/graph_metrics.json).
- Appendea un registro por documento procesado.
- Mantiene trazabilidad histórica de la ejecución.
- Incluye timestamp ISO y campos estandarizados para análisis futuro.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("graph_builder.metrics")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _load_existing_metrics(path: Path) -> List[Dict[str, Any]]:
    """Carga métricas existentes o devuelve una lista vacía."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # versión previa con dict -> migrar a lista
            return [data]
    except Exception as e:
        logger.warning("Error leyendo métricas previas (%s): %s", path, repr(e))
    return []


def append_metrics_record(path: Path, record: Dict[str, Any]) -> None:
    """
    Appendea un nuevo registro de métricas al archivo graph_metrics.json.

    Args:
        path: Ruta del archivo JSON de métricas.
        record: Diccionario con métricas del documento procesado.
    """
    try:
        existing = _load_existing_metrics(path)
        record.setdefault("timestamp", datetime.utcnow().isoformat())
        existing.append(record)

        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("[METRICS] Registrado doc_id=%s", record.get("doc_id"))
    except Exception as e:
        logger.error("Error escribiendo métricas: %s", repr(e))
