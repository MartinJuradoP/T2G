# -*- coding: utf-8 -*-
"""
contextizer.hybrid.density_clustering — Clustering semántico por densidad
========================================================================

Resumen
-------
Implementa un **clustering alternativo basado en densidad semántica (DBSCAN)** 
para escenarios donde BERTopic falla o resulta costoso debido a pocos fragmentos 
o alta heterogeneidad temática. 

El objetivo es detectar **subtemas locales** de manera robusta sin depender de 
UMAP ni de reducción de dimensionalidad agresiva, preservando la estructura 
semántica original de los embeddings.

Principios de diseño
--------------------
- **Determinismo:** resultados reproducibles a partir de embeddings fijos.
- **Eficiencia:** usa `DBSCAN` (cosine distance) directamente sobre embeddings.
- **Compatibilidad:** genera estructuras equivalentes a los `TopicItem` 
  esperados por el pipeline T2G.
- **Auto-contenido:** no depende de BERTopic ni de componentes externos.

Referencias
-----------
- Ester et al. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases*.
- Grootendorst, M. (2022). *BERTopic* — baseline conceptual framework.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.

Ejemplo de uso
--------------
>>> from contextizer.hybrid.density_clustering import cluster_by_density, build_topic_items
>>> embeddings = embedder.encode(["A", "B", "C"])
>>> labels, n_clusters = cluster_by_density(embeddings)
>>> topics = build_topic_items(texts=["A", "B", "C"], labels=labels, embedder=embedder)
>>> topics[0]["keywords"]
["stock", "earnings", "forecast"]

Salida esperada (ejemplo simplificado)
-------------------------------------
{
  "topic_id": 0,
  "count": 3,
  "exemplar": "Southern Co. stock forecast...",
  "keywords": ["stock", "earnings", "forecast"],
  "meta": {"cluster_method": "dbscan", "eps": 0.25}
}
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIGURACIÓN BASE
# -----------------------------------------------------------------------------
DEFAULT_EPS = 0.25
DEFAULT_MIN_SAMPLES = 2

# -----------------------------------------------------------------------------
# FUNCIÓN: cluster_by_density
# -----------------------------------------------------------------------------
def cluster_by_density(
    embeddings: np.ndarray,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES
) -> Tuple[np.ndarray, int]:
    """
    Aplica clustering por densidad sobre los embeddings.

    Parámetros
    ----------
    embeddings : np.ndarray
        Matriz de embeddings (n_samples, dim).
    eps : float
        Radio máximo de vecindad en el espacio coseno.
    min_samples : int
        Mínimo de puntos requeridos para formar un cluster.

    Retorna
    -------
    labels : np.ndarray
        Etiquetas de cluster asignadas a cada muestra (-1 = ruido).
    n_clusters : int
        Número total de clusters detectados (sin contar ruido).
    """
    if embeddings is None or len(embeddings) == 0:
        return np.array([]), 0

    # DBSCAN con métrica coseno (robusto a magnitud)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters

# -----------------------------------------------------------------------------
# FUNCIÓN: build_topic_items
# -----------------------------------------------------------------------------
def build_topic_items(
    texts: List[str],
    labels: np.ndarray,
    embedder,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Construye objetos tipo `TopicItem` equivalentes, a partir de clusters DBSCAN.

    Parámetros
    ----------
    texts : list[str]
        Fragmentos o chunks textuales.
    labels : np.ndarray
        Etiquetas de cluster (-1 = ruido).
    embedder :
        Modelo SentenceTransformer (o compatible) para embeddings.
    top_k : int
        Número máximo de keywords a generar por cluster.

    Retorna
    -------
    list[dict]
        Lista de diccionarios con estructura compatible con `TopicItem`.

    Ejemplo de salida
    -----------------
    [
      {
        "topic_id": 0,
        "count": 3,
        "exemplar": "Southern Co. closed at $95.49...",
        "keywords": ["southern", "utilities", "earnings"],
        "meta": {"cluster_method": "dbscan", "eps": 0.25}
      }
    ]
    """
    from .keyword_fusion import fuse_keywords
    from .mmr import mmr_filter

    topics: List[Dict[str, Any]] = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 0:
        # Caso sin clusters válidos
        merged, keybert_only, emb = fuse_keywords(texts, embedder, top_k=top_k)
        merged = mmr_filter(merged, emb, top_k=top_k)
        return [{
            "topic_id": 0,
            "count": len(texts),
            "exemplar": texts[0] if texts else "",
            "keywords": merged,
            "meta": {"cluster_method": "none", "reason": "no-valid-clusters"}
        }]

    for topic_id in sorted(set(labels)):
        if topic_id == -1:
            continue
        cluster_texts = [t for t, l in zip(texts, labels) if l == topic_id]
        cluster_emb = embedder.encode(cluster_texts, show_progress_bar=False)
        merged, keybert_only, emb_kw = fuse_keywords(cluster_texts, embedder, top_k=top_k)
        merged = mmr_filter(merged, emb_kw, top_k=top_k)

        exemplar_idx = _find_exemplar(cluster_emb)
        topics.append({
            "topic_id": int(topic_id),
            "count": len(cluster_texts),
            "exemplar": cluster_texts[exemplar_idx],
            "keywords": merged,
            "meta": {
                "cluster_method": "dbscan",
                "eps": DEFAULT_EPS,
                "min_samples": DEFAULT_MIN_SAMPLES
            }
        })
    return topics

# -----------------------------------------------------------------------------
# FUNCIÓN: _find_exemplar
# -----------------------------------------------------------------------------
def _find_exemplar(embeddings: np.ndarray) -> int:
    """
    Encuentra el texto más representativo de un cluster 
    (mínima distancia media al resto de embeddings).

    Retorna
    -------
    int : índice del texto más representativo.
    """
    if len(embeddings) == 0:
        return 0
    sim_matrix = cosine_similarity(embeddings)
    mean_sim = sim_matrix.mean(axis=1)
    return int(np.argmax(mean_sim))
