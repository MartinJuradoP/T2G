# -*- coding: utf-8 -*-
"""
contextizer.hybrid.mmr — Maximal Marginal Relevance para keywords de tópicos
==========================================================================

Resumen
-------
Este módulo implementa **MMR (Maximal Marginal Relevance)** para seleccionar un
subconjunto de *keywords* **relevante y diverso** a partir de una lista de
candidatas. MMR reduce la **redundancia semántica** (sinónimos, variaciones) y
favorece la **cobertura informativa** del tópico.

Se usa inmediatamente después de la generación de keywords (TF‑IDF + KeyBERT
opcional), dentro del Contextizer Híbrido. La salida es una lista de keywords
ordenada por utilidad global para representar el tópico.

Garantías y diseño
------------------
- **Determinista** dado un mismo embedding.
- **Independiente** de BERTopic; funciona con *sentence embeddings*.
- **Additive**: no altera contratos; solo produce una lista refinada
  (`mmr_keywords`) que puede adjuntarse opcionalmente a `TopicItem`.

Fórmula (intuición)
-------------------
Seleccionamos de forma codiciosa el siguiente término `i` que **maximiza**:

    score(i) = λ · sim(i, centroide) − (1−λ) · max_{j∈S} sim(i, j)

donde `S` es el conjunto ya seleccionado, `sim` es similitud coseno, y `λ` (
`lambda_diversity`) controla el equilibrio **relevancia** (→1) vs **diversidad** (→0).

Referencias
-----------
- Carbonell, J. & Goldstein, J. (1998). *The use of MMR for diversity‑based IR*.
- Reimers & Gurevych (2019). *Sentence‑BERT* (embeddings para similitud semántica).

Ejemplo rápido
--------------
>>> from contextizer.hybrid.mmr import mmr_filter
>>> import numpy as np
>>> candidates = ["earnings", "profits", "utilities", "zacks", "forecast"]
>>> emb = np.random.RandomState(42).randn(len(candidates), 8)  # demo
>>> mmr_filter(candidates, candidate_emb=emb, top_k=3, lambda_diversity=0.7)
['earnings', 'utilities', 'forecast']
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────────────────────

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    """Normaliza por fila para estabilidad numérica en cosenos.

    Si alguna fila es cero, la deja como está para evitar divisiones por cero.
    """
    if M is None or M.ndim != 2:
        return M
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return M / norms


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

def mmr_filter(
    candidates: List[str],
    candidate_emb: Optional[np.ndarray] = None,
    *,
    embedder=None,
    centroid: Optional[np.ndarray] = None,
    lambda_diversity: float = 0.7,
    top_k: int = 10,
    normalize: bool = True,
) -> List[str]:
    """Selecciona un subconjunto de `candidates` con **máxima relevancia** y **mínima**
    **redundancia semántica** mediante MMR.

    Parámetros
    ----------
    candidates : list[str]
        Lista de palabras/frases candidatas ordenadas (p. ej. por TF‑IDF/KeyBERT).
    candidate_emb : np.ndarray | None
        Embeddings de `candidates` con forma (n, dim). Si es `None`, se intenta
        obtener con `embedder.encode(candidates)`.
    embedder : Any, opcional
        Modelo de *sentence embeddings* con método `.encode(list[str])`.
    centroid : np.ndarray | None
        Vector de referencia para relevancia. Por defecto, el **centroide** de
        `candidate_emb` (media por dimensión).
    lambda_diversity : float, por defecto 0.7
        Peso de relevancia (→1) vs diversidad (→0). Valores típicos: [0.6, 0.8].
    top_k : int, por defecto 10
        Tamaño del subconjunto final de keywords. Si `top_k >= len(candidates)`,
        retorna `candidates` como están.
    normalize : bool, por defecto True
        Si `True`, normaliza embeddings para cosenos estables.

    Retorna
    -------
    list[str]
        Lista de *keywords* seleccionadas por MMR (ordenadas por selección).

    Notas
    -----
    - El primer término seleccionado es el más **relevante** (máxima similitud
      con el centroide). A partir de ahí se penaliza la similitud con palabras
      ya elegidas para promover **diversidad**.
    - Si no hay embeddings viables, se retorna la lista original truncada.
    """
    n = len(candidates)
    if n == 0:
        return []
    if top_k >= n:
        return list(candidates)

    # Embeddings
    E: Optional[np.ndarray] = candidate_emb
    if E is None and embedder is not None:
        try:
            E = embedder.encode(candidates, show_progress_bar=False)
        except Exception:
            E = None

    if E is None or not isinstance(E, np.ndarray) or E.ndim != 2 or E.shape[0] != n:
        # Fallback defensivo: sin embeddings, devolvemos truncado
        return list(candidates[:top_k])

    if normalize:
        E = _l2_normalize(E)

    # Centroide (referencia de relevancia)
    if centroid is None:
        c = E.mean(axis=0, keepdims=True)
        c = _l2_normalize(c)
    else:
        c = centroid.reshape(1, -1)
        if normalize:
            c = _l2_normalize(c)

    # Similitudes
    sim_to_centroid = cosine_similarity(E, c).reshape(-1)  # (n,)
    sim_matrix = cosine_similarity(E, E)                   # (n, n)

    # Selección codiciosa
    selected: list[int] = []
    candidates_idx = list(range(n))

    # 1) Arrancar con el más relevante (máx sim al centroide)
    first = int(np.argmax(sim_to_centroid))
    selected.append(first)
    candidates_idx.remove(first)

    # 2) Iterar mientras falten posiciones
    while len(selected) < top_k and candidates_idx:
        scores = []
        for i in candidates_idx:
            rel = sim_to_centroid[i]
            div = max(sim_matrix[i, j] for j in selected) if selected else 0.0
            score = lambda_diversity * rel - (1.0 - lambda_diversity) * div
            scores.append((score, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        best_idx = scores[0][1]
        selected.append(best_idx)
        candidates_idx.remove(best_idx)

    return [candidates[i] for i in selected]
