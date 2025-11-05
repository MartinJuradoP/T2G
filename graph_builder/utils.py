# graph_builder/utils.py
# -*- coding: utf-8 -*-
"""
utils.py — Utilidades generales para el Graph Builder (T2G)
===========================================================

Incluye:
- Lectura robusta de archivos JSON.
- Validaciones básicas y helpers.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("graph_builder.utils")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def load_json_safe(path: str | Path) -> Dict[str, Any]:
    """
    Carga un archivo JSON con manejo seguro de errores.
    Devuelve {} si no es legible o está corrupto.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Archivo JSON no encontrado: %s", path)
        return {}

    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("JSON inválido (%s): %s", path, repr(e))
    except Exception as e:
        logger.error("Error leyendo %s: %s", path, repr(e))
    return {}
