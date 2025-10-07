# -*- coding: utf-8 -*-
"""
contextizer.hybrid.keyword_fusion — Fusión de keywords (TF‑IDF + KeyBERT opcional)
=================================================================================

Resumen
-------
Este módulo implementa la **extracción y fusión de palabras clave** para el
Contextizer Híbrido. Combina un enfoque **léxico estadístico (TF‑IDF)** con un
enfoque **semántico (KeyBERT, opcional)** para producir listas de keywords
**contextuales y no redundantes** que describen cada tópico/cluster.

El resultado está diseñado para alimentar el filtro de **Maximal Marginal
Relevance (MMR)**, de modo que las keywords finales sean informativas y
diversas, evitando sinónimos/clones.

Garantías
---------
- **Sin dependencias rígidas**: KeyBERT es opcional. Si no está disponible, se
  usa TF‑IDF puro y embeddings de oraciones para MMR.
- **Reproducible y ligero**: TF‑IDF interno (micro‑implementación) apto para
  lotes pequeños y textos cortos (reseñas, posts, noticias).
- **Compatibilidad**: Interfaz estable y simple:

  ```python
  merged_keywords, keybert_only, emb_matrix = fuse_keywords(
      texts, embedder, top_k=10, min_len=3, use_keybert=True
  )
  ```

Referencias
-----------
- Grootendorst (2022) — BERTopic (uso original de c-TF-IDF).
- Ganesan (2020) — KeyBERT.
- Carbonell & Goldstein (1998) — MMR (aplicado aguas abajo en `mmr.py`).

Ejemplo rápido
--------------
>>> merged, keybert_only, kw_emb = fuse_keywords(
...     ["Southern Co. earnings beat expectations", "Utilities sector outlook"],
...     embedder, top_k=8, use_keybert=True
... )
>>> merged[:4]
['earnings', 'utilities', 'outlook', 'southern']
"""

from __future__ import annotations
from typing import Iterable, List, Tuple
from collections import Counter
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# TF‑IDF ligero (micro‑implementación)
# ──────────────────────────────────────────────────────────────────────────────

def _tfidf_top(texts: Iterable[str], top_k: int = 10, min_len: int = 3) -> List[str]:
    """Calcula un ranking simple TF‑IDF y devuelve las `top_k` keywords.

    Notas
    -----
    - Diseñado para **lotes pequeños**: documentos o clusters con pocos textos.
    - Se aplica tokenización por espacios con filtro de longitud mínima.
    - No realiza stemming/lemmatización para evitar dependencias pesadas.
    """
    docs = [" ".join((t or "").lower().split()) for t in texts if t]
    if not docs:
        return []

    vocab = Counter()
    df = Counter()
    for d in docs:
        toks = [w for w in d.split(" ") if len(w) >= min_len]
        vocab.update(toks)
        for t in set(toks):
            df[t] += 1

    N = len(docs)
    scores = {}
    for term, tf in vocab.items():
        # Smoothing para estabilidad con lotes muy pequeños
        idf = np.log((N + 1) / (df[term] + 1)) + 1.0
        scores[term] = tf * idf

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [w for w, _ in top]

# ──────────────────────────────────────────────────────────────────────────────
# Fusión TF‑IDF + KeyBERT (opcional) y embeddings para MMR
# ──────────────────────────────────────────────────────────────────────────────

def fuse_keywords(
    texts: Iterable[str],
    embedder,
    top_k: int = 10,
    min_len: int = 3,
    use_keybert: bool = True,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Genera y fusiona keywords **léxicas** y **semánticas**; devuelve embeddings.

    Parámetros
    ----------
    texts : Iterable[str]
        Colección de textos (documentos o cluster de chunks) para extraer keywords.
    embedder : Any
        Modelo de *sentence embeddings* (p. ej., `SentenceTransformer`). Se usa
        para generar embeddings de keywords y permitir MMR aguas abajo.
    top_k : int, por defecto 10
        Número máximo de keywords finales deseadas.
    min_len : int, por defecto 3
        Longitud mínima de token para TF‑IDF.
    use_keybert : bool, por defecto True
        Si es posible, intenta usar KeyBERT para enriquecer con señales semánticas.

    Retorna
    -------
    merged_keywords : list[str]
        Lista fusionada y **deduplicada** (orden estable) de tamaño ≤ `top_k`.
    keybert_only : list[str]
        Lista de keywords sugeridas por KeyBERT (puede estar vacía si no disponible).
    emb_matrix : np.ndarray
        Embeddings de las keywords fusionadas (o de la mejor alternativa); sirven
        como entrada a MMR para seleccionar un subconjunto **no redundante**.
    """
    # Normalización mínima y filtrado de vacíos
    docs = [t for t in texts if t and t.strip()]

    # 1) TF‑IDF (siempre disponible)
    tfidf_kw = _tfidf_top(docs, top_k=top_k, min_len=min_len)

    # 2) KeyBERT (opcional)
    keybert_only: List[str] = []
    if use_keybert:
        try:
            from keybert import KeyBERT
            kb = KeyBERT(model=embedder)
            # Concatenamos para maximizar contexto; KeyBERT internamente chunkeará
            kb_pairs = kb.extract_keywords("\n".join(docs), top_n=top_k)
            keybert_only = [w for (w, _s) in kb_pairs]
        except Exception:
            keybert_only = []

    # 3) Fusión y deduplicación estable
    merged: List[str] = []
    seen = set()
    for w in (tfidf_kw + keybert_only):
        wl = (w or "").strip().lower()
        if wl and wl not in seen:
            merged.append(wl)
            seen.add(wl)
        if len(merged) >= top_k:
            break

    # 4) Embeddings para MMR (si no hay merged, usar el mejor fallback)
    baseline = merged or tfidf_kw or keybert_only or ["topic"]
    try:
        emb_matrix = embedder.encode(baseline, show_progress_bar=False)
    except Exception:
        # Fallback defensivo: vector unitario para evitar romper flujos aguas abajo
        emb_matrix = np.ones((len(baseline), 1), dtype=float)

    return merged, keybert_only, emb_matrix
