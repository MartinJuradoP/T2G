# -*- coding: utf-8 -*-
"""
contextizer.hybrid.metrics_ext — Métricas extendidas para el Contextizer Híbrido
===============================================================================

Resumen
-------
Este módulo define **métricas complementarias** a `contextizer/metrics.py` para
evaluar la calidad de los tópicos producidos por el **modo híbrido** (o por el
modelo clásico), con foco en **robustez**, **diversidad** y **coherencia**.

Principios
----------
- **No intrusivo**: opera sobre los contratos existentes (`TopicsDocMeta`,
  `TopicItem`) sin modificar su forma.
- **Opcionalmente semántico**: si se provee un `embedder` (p. ej. SentenceTransformer),
  algunas métricas usan similitudes coseno sobre embeddings; si no, caen a
  estimadores léxicos/estadísticos.
- **Interpretables**: valores acotados y documentados, útiles para umbrales.

Métricas incluidas
------------------
- `entropy_topics(dt)` → dispersión de masa entre tópicos (0..1, mayor= más mezcla)
- `redundancy_score(dt)` → redundancia léxica promedio por tópico (0..1, menor= mejor)
- `keywords_diversity_ext(dt)` → diversidad de keywords global (0..1, mayor= mejor)
- `semantic_variance(dt, embedder=None)` → varianza de embeddings de *exemplars*
- `coherence_semantic(dt, embedder=None)` → coherencia intra‑tópico basada en keywords

Notas
-----
- **No reemplazan** a las métricas existentes; son *additive*.
- Están pensadas para *monitoring/QA* y ajuste de umbrales en producción.

Ejemplo rápido
--------------
>>> from contextizer.hybrid.metrics_ext import entropy_topics, redundancy_score
>>> H = entropy_topics(dt)
>>> R = redundancy_score(dt)
>>> H, R
(0.72, 0.18)
"""

from __future__ import annotations
from typing import List, Iterable, Tuple, Optional
import numpy as np

from ..schemas import TopicsDocMeta, TopicItem

# ──────────────────────────────────────────────────────────────────────────────
# Helpers básicos
# ──────────────────────────────────────────────────────────────────────────────

def _topic_items(dt: TopicsDocMeta) -> List[TopicItem]:
    return list(dt.topics or [])

def _all_keywords(items: Iterable[TopicItem]) -> List[str]:
    kws: List[str] = []
    for t in items:
        kws.extend((t.keywords or []))
        # Si existen keywords post‑MMR en meta, pueden añadirse:
        mmr_kws = (t.meta or {}).get("mmr_keywords") if isinstance(t.meta, dict) else None
        if mmr_kws:
            kws.extend(mmr_kws)
    return [k for k in kws if isinstance(k, str) and k]

# ──────────────────────────────────────────────────────────────────────────────
# 1) Dispersión de masa entre tópicos (entropía normalizada)
# ──────────────────────────────────────────────────────────────────────────────

def entropy_topics(dt: TopicsDocMeta) -> float:
    """Entropía normalizada de la distribución de tamaños de tópicos.

    Rango: 0..1. Valores altos implican **distribución más uniforme** (más mezcla).
    Si hay 1 o 0 tópicos válidos, retorna 0.0.
    """
    sizes = np.array([int(t.count or 0) for t in _topic_items(dt) if int(t.topic_id) != -1], dtype=float)
    if sizes.size <= 1 or sizes.sum() == 0:
        return 0.0
    p = sizes / sizes.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    H_max = np.log(len(p))
    return float(H / (H_max + 1e-12))

# ──────────────────────────────────────────────────────────────────────────────
# 2) Redundancia léxica promedio por tópico
# ──────────────────────────────────────────────────────────────────────────────

def redundancy_score(dt: TopicsDocMeta) -> float:
    """Mide redundancia léxica por tópico (0..1, **menor es mejor**).

    Definición: 1 − (unique/total) promediado sobre tópicos con keywords.
    - 0.0 → sin redundancia (todos los términos únicos)
    - 1.0 → máxima redundancia (todas las palabras iguales)
    """
    items = [t for t in _topic_items(dt) if (t.keywords or [])]
    if not items:
        return 0.0
    vals: List[float] = []
    for t in items:
        kws = list(t.keywords or [])
        if not kws:
            continue
        unique = len(set(kws))
        total = len(kws)
        vals.append(1.0 - (unique / max(1, total)))
    return float(np.mean(vals)) if vals else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 3) Diversidad global de keywords (extensión)
# ──────────────────────────────────────────────────────────────────────────────

def keywords_diversity_ext(dt: TopicsDocMeta) -> float:
    """Proporción de keywords **únicas** sobre el total (0..1, mayor= mejor).

    Similar a `metrics.keywords_diversity` pero considerando posibles keywords
    post‑MMR almacenadas en `TopicItem.meta.mmr_keywords`.
    """
    kws = _all_keywords(_topic_items(dt))
    if not kws:
        return 0.0
    return float(len(set(kws)) / len(kws))

# ──────────────────────────────────────────────────────────────────────────────
# 4) Varianza semántica de *exemplars*
# ──────────────────────────────────────────────────────────────────────────────

def semantic_variance(dt: TopicsDocMeta, embedder=None) -> float:
    """Varianza de embeddings de los `exemplar` por tópico (0..∞, mayor= más dispersión).

    Si no se provee `embedder` o faltan *exemplars*, retorna 0.0.
    """
    exemplars = [t.exemplar for t in _topic_items(dt) if isinstance(t.exemplar, str) and t.exemplar]
    if not exemplars or embedder is None:
        return 0.0
    try:
        E = np.array(embedder.encode(exemplars, show_progress_bar=False))
    except Exception:
        return 0.0
    if E.ndim != 2 or len(E) < 2:
        return 0.0
    mu = E.mean(axis=0, keepdims=True)
    diff = E - mu
    return float((diff * diff).sum(axis=1).mean())

# ──────────────────────────────────────────────────────────────────────────────
# 5) Coherencia semántica intra‑tópico basada en keywords
# ──────────────────────────────────────────────────────────────────────────────

def coherence_semantic(dt: TopicsDocMeta, embedder=None) -> float:
    """Coherencia intra‑tópico promediada (0..1, mayor= mejor).

    Método: para cada tópico, se vectorizan sus keywords (o mmr_keywords si están
    disponibles) y se calcula la **similitud coseno media** entre pares.

    Consideraciones
    ---------------
    - Requiere `embedder`. Si no hay, retorna 0.0.
    - Para tópicos con menos de 2 keywords, se ignora el cálculo.
    """
    if embedder is None:
        return 0.0

    from sklearn.metrics.pairwise import cosine_similarity

    items = _topic_items(dt)
    vals: List[float] = []
    for t in items:
        cand = (t.meta or {}).get("mmr_keywords") if isinstance(t.meta, dict) else None
        kws = cand or (t.keywords or [])
        kws = [k for k in kws if isinstance(k, str) and k]
        if len(kws) < 2:
            continue
        try:
            K = np.array(embedder.encode(kws, show_progress_bar=False))
        except Exception:
            continue
        if K.ndim != 2 or len(K) < 2:
            continue
        sim = cosine_similarity(K)
        # media de la parte superior de la matriz sin diagonal
        m = (np.sum(sim) - np.trace(sim)) / max(1, (sim.shape[0] * (sim.shape[1] - 1)))
        vals.append(float(m))
    return float(np.mean(vals)) if vals else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 6) Balance de tópicos (uniformidad de tamaños)
# ──────────────────────────────────────────────────────────────────────────────

def topic_balance(dt: TopicsDocMeta) -> float:
    """Uniformidad del tamaño de tópicos (0..1, mayor= más equilibrado).

    Calcula 1 - (std / mean) de los tamaños de cluster (count por tópico).
    - 1.0 → todos los tópicos tienen tamaño similar.
    - 0.0 → un tópico domina o los tamaños son muy desiguales.
    """
    sizes = np.array([int(t.count or 0) for t in _topic_items(dt)], dtype=float)
    if len(sizes) <= 1 or sizes.mean() == 0:
        return 0.0
    return float(1 - (sizes.std() / (sizes.mean() + 1e-9)))


# ──────────────────────────────────────────────────────────────────────────────
# 7) Redundancia global de keywords (0..1, menor= mejor)
# ──────────────────────────────────────────────────────────────────────────────

def keyword_redundancy_rate(dt: TopicsDocMeta) -> float:
    """Detecta solapamientos globales entre keywords de todos los tópicos.

    0.0 → sin duplicados; 1.0 → todas las palabras se repiten.
    """
    kws = _all_keywords(_topic_items(dt))
    if not kws:
        return 0.0
    unique = len(set(kws))
    total = len(kws)
    return float(1 - (unique / max(1, total)))


# ──────────────────────────────────────────────────────────────────────────────
# 8) Proporción de tópicos válidos (con keywords no vacías)
# ──────────────────────────────────────────────────────────────────────────────

def informativeness_ratio(dt: TopicsDocMeta) -> float:
    """Fracción de tópicos con keywords válidas (0..1, mayor= mejor)."""
    items = [t for t in _topic_items(dt) if t.keywords]
    if not _topic_items(dt):
        return 0.0
    return float(len(items) / len(_topic_items(dt)))


# ──────────────────────────────────────────────────────────────────────────────
# 9) Alineación contextual chunk↔tópico heredado
# ──────────────────────────────────────────────────────────────────────────────

def context_alignment(chunks: List[dict]) -> float:
    """Mide la afinidad semántica promedio entre chunks y sus topic_hints.

    Requiere que cada chunk tenga los embeddings:
      - `emb_chunk`
      - `emb_hint` (promedio del tópico heredado)
    Si faltan, retorna 0.0.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sims = []
    for ch in chunks:
        emb_c = np.array(ch.get("emb_chunk"))
        emb_h = np.array(ch.get("emb_hint"))
        if emb_c.ndim == 1 and emb_h.ndim == 1:
            sims.append(float(cosine_similarity([emb_c], [emb_h])[0][0]))
    return float(np.mean(sims)) if sims else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 10) Penalización por redundancia entre chunks
# ──────────────────────────────────────────────────────────────────────────────

def redundancy_penalty(embeddings: np.ndarray, threshold: float = 0.85) -> float:
    """Penaliza similitud alta entre embeddings de chunks (0..1, menor= mejor).

    - Calcula la proporción de pares con similitud coseno > threshold.
    - Ideal < 0.1. Si > 0.5 indica repetición o fragmentación excesiva.
    """
    if embeddings is None or len(embeddings) < 2:
        return 0.0
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embeddings)
    iu = np.triu_indices_from(sim, k=1)
    mask = (sim[iu] > threshold).astype(float)
    return float(mask.mean() if mask.size else 0.0)
