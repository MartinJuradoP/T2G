# -*- coding: utf-8 -*-
"""
utils.py — Utilidades para el Adaptive Schema Selector.

Incluye:
- Similitud coseno y normalización de embeddings.
- Funciones de extracción y normalización de tokens/keywords.
- Cálculo de centroides de documentos y chunks.
- Selección top-k (coseno).
- Helpers de logging/debug.
"""

from __future__ import annotations
import re
import numpy as np
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# 🔹 Similitud y vectores
# ---------------------------------------------------------------------------

def cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    """Calcula similitud coseno entre dos vectores."""
    if vec_a is None or vec_b is None:
        return 0.0
    a, b = np.array(vec_a, dtype=float), np.array(vec_b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_vector(vec: List[float]) -> List[float]:
    """Normaliza un vector a norma 1."""
    v = np.array(vec, dtype=float)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else v.tolist()


def topk_cosine(query_vec: List[float], candidates: List[List[float]], k: int = 5) -> List[Tuple[int, float]]:
    """
    Retorna los top-k candidatos más similares a un vector query.
    Devuelve lista de (índice, score).
    """
    if not candidates:
        return []
    scores = [(i, cosine_sim(query_vec, c)) for i, c in enumerate(candidates)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def topk(query_vec: List[float], candidates: List[List[float]], k: int = 5) -> List[Tuple[int, float]]:
    """
    Alias de `topk_cosine` para compatibilidad retro con selector.py.
    """
    return topk_cosine(query_vec, candidates, k=k)

# ---------------------------------------------------------------------------
# 🔹 Normalización de texto y tokens
# ---------------------------------------------------------------------------

def normalize_keyword(text: str) -> str:
    """Normaliza palabra clave (lowercase, sin espacios extra)."""
    return text.strip().lower() if text else ""


def normalize_tokens(text: str) -> List[str]:
    """
    Normaliza y tokeniza texto en palabras.
    - Lowercase
    - Elimina caracteres no alfanuméricos (excepto acentos/ñ)
    """
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^0-9a-záéíóúüñ\s]", " ", text)
    return [t for t in text.split() if t.strip()]


def keyword_match(text: str, keywords: List[str]) -> bool:
    """Devuelve True si alguna keyword aparece en el texto."""
    norm_text = normalize_keyword(text)
    return any(k in norm_text for k in map(normalize_keyword, keywords))

# ---------------------------------------------------------------------------
# 🔹 Keywords desde doc/chunk
# ---------------------------------------------------------------------------

def get_doc_keywords(doc: Dict[str, Any]) -> List[str]:
    """
    Extrae keywords globales de un documento enriquecido por Contextizer.
    Busca en meta.topics_doc y meta.topics_chunks.
    """
    kws: List[str] = []
    if "meta" in doc:
        if "topics_doc" in doc["meta"]:
            kws.extend(doc["meta"]["topics_doc"].get("keywords_global", []))
        if "topics_chunks" in doc["meta"]:
            kws.extend(doc["meta"]["topics_chunks"].get("keywords_global", []))
    return sorted(set(normalize_keyword(k) for k in kws if k))


def get_chunk_keywords(chunk: Dict[str, Any]) -> List[str]:
    """
    Extrae keywords locales de un chunk.
    Busca en chunk['topic'] y meta.topics_chunk.keywords.
    """
    kws: List[str] = []
    if "topic" in chunk:
        kws.extend(chunk["topic"].get("keywords", []))
    if "meta" in chunk and "topics_chunk" in chunk["meta"]:
        kws.extend(chunk["meta"]["topics_chunk"].get("keywords", []))
    return sorted(set(normalize_keyword(k) for k in kws if k))

# ---------------------------------------------------------------------------
# 🔹 Embeddings centroid (doc y chunk)
# ---------------------------------------------------------------------------

def get_doc_centroid(doc: Dict[str, Any]) -> List[float]:
    """
    Calcula el embedding centroide de un documento.
    Usa embeddings en meta.topics_doc o en chunks.
    """
    embeddings: List[List[float]] = []
    if "meta" in doc and "topics_doc" in doc["meta"]:
        embeddings.extend(doc["meta"]["topics_doc"].get("embeddings", []))
    if "chunks" in doc:
        for ch in doc["chunks"]:
            if "meta" in ch and "embedding" in ch["meta"]:
                embeddings.append(ch["meta"]["embedding"])
    if not embeddings:
        return []
    arr = np.array(embeddings, dtype=float)
    centroid = np.mean(arr, axis=0)
    return normalize_vector(centroid.tolist())


def get_chunk_centroid(chunk: Dict[str, Any]) -> List[float]:
    """
    Calcula el embedding centroide de un chunk.
    Usa embedding en meta o promedio de embeddings de oraciones.
    """
    if "meta" in chunk and "embedding" in chunk["meta"]:
        return normalize_vector(chunk["meta"]["embedding"])
    embeddings: List[List[float]] = []
    if "sentences" in chunk:
        for sent in chunk["sentences"]:
            if "embedding" in sent.get("meta", {}):
                embeddings.append(sent["meta"]["embedding"])
    if not embeddings:
        return []
    arr = np.array(embeddings, dtype=float)
    centroid = np.mean(arr, axis=0)
    return normalize_vector(centroid.tolist())

# ---------------------------------------------------------------------------
# 🔹 Logging / Debug
# ---------------------------------------------------------------------------

def log_evidence(msg: str, evidence: dict) -> str:
    """Formatea evidencias en logs legibles."""
    return f"{msg} | evidence={evidence}"
