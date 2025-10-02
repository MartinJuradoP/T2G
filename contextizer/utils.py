"""
utils.py — Utilidades para Contextizer
--------------------------------------
Incluye helpers para:
- Normalizar texto
- Semillas reproducibles
- Caché de embeddings (joblib)
- Escritura atómica de JSON
"""

from __future__ import annotations
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from pydantic import BaseModel

try:
    import joblib  # opcional, mejora performance de caching
except Exception:
    joblib = None

# ---------------------------------------------------------------------------
# Regex comunes
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalización de texto
# ---------------------------------------------------------------------------
def prepare_text_for_topic(text: str) -> str:
    """Normaliza texto para topic modeling:
    - Quita guiones y bullets
    - Colapsa espacios
    - Normaliza comillas y apóstrofes
    """
    t = text.replace("\u00AD", "-")
    t = _WHITESPACE_RE.sub(" ", t).strip()
    t = re.sub(r"(?m)^(?:[-•*]\s+|\d+[\.)]\s+)", "", t)
    t = re.sub(r"[“”«»]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    return t


# ---------------------------------------------------------------------------
# Semillas reproducibles
# ---------------------------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    """Fija semillas globales en random, numpy y torch (si está disponible)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Caché de embeddings
# ---------------------------------------------------------------------------
def _hash_key(model_name: str, texts: Sequence[str]) -> str:
    """Genera un hash SHA256 a partir del modelo y los textos."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for t in texts:
        h.update(t[:2000].encode("utf-8"))  # límite defensivo
    return h.hexdigest()


def embeddings_cache_path(cache_dir: Path, model_name: str, texts: Sequence[str]) -> Path:
    """Devuelve path de caché para embeddings basado en hash SHA256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_key(model_name, texts)
    return cache_dir / f"emb_{key}.joblib"


def try_load_embeddings(cache_path: Path) -> Optional[np.ndarray]:
    """Intenta cargar embeddings desde caché.
    Retorna None si no existe o falla.
    """
    if joblib is None or not cache_path.exists():
        return None
    try:
        return joblib.load(cache_path)
    except Exception:
        return None


def save_embeddings(cache_path: Path, arr: np.ndarray) -> None:
    """Guarda embeddings en caché."""
    if joblib is None:
        return
    try:
        joblib.dump(arr, cache_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Escritura segura de JSON
# ---------------------------------------------------------------------------
def atomic_write_json(obj: BaseModel | dict, out_path: Path) -> None:
    """Escribe JSON atómicamente (safe write con tempfile + os.replace)."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(out_path.parent)) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, out_path)
