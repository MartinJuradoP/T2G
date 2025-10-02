# -*- coding: utf-8 -*-
"""
contextizer.py — Contextualización semántica (doc-level y chunk-level)
======================================================================

Este módulo implementa la contextualización de documentos y chunks para el
pipeline T2G usando BERTopic + SentenceTransformers, con fallbacks robustos.

Etapas soportadas
-----------------
- run_contextizer_on_doc(ir_path, cfg, outdir):
    Enriquecer un DocumentIR (outputs_ir/*.json) con tópicos globales y
    guardarlo en `outdir` como *_doc_topics.json. Inyecta resultados en
    data["meta"]["topics_doc"] sin alterar el resto.

- run_contextizer_on_chunks(chunk_path, cfg):
    Enriquecer un DocumentChunks (outputs_chunks/*.json) con tópicos locales.
    Escribe de vuelta en el mismo archivo, agregando:
      - Por chunk:  chunk["topic"] = {topic_id, keywords, prob}
      - Global:     data["meta"]["topics_chunks"] = {...}

Principios de diseño
--------------------
- Resiliencia:
  * Si el texto es escaso (1–2 bloques o 1–2 chunks) o BERTopic falla, usamos
    un fallback de keywords por frecuencia (sin dependencias externas).
  * Umbral dinámico para UMAP: n_neighbors = min(cfg.umap_n_neighbors, max(2, n_samples-1))

- Calidad de keywords:
  * Limpieza de stopwords (ES/EN), número configurable de keywords por tópico.
  * Pre-procesamiento homogéneo (utils.prepare_text_for_topic).

- Compatibilidad:
  * No rompe los contratos del Parser ni del Chunker.
  * Metadatos adicionales no afectan consumidores existentes.

Requisitos
----------
- pip: sentence-transformers, bertopic, umap-learn, numpy
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Iterable, Tuple
from collections import Counter

import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from .schemas import (
    TopicModelConfig,
    TopicItem,
    TopicsDocMeta,
    ChunkTopic,
    TopicsChunksMeta,
)
from .utils import (
    prepare_text_for_topic,
    atomic_write_json,
    set_global_seeds,
    embeddings_cache_path,
    try_load_embeddings,
    save_embeddings,
)

logger = logging.getLogger("contextizer")
logger.setLevel(logging.INFO)


# ============================================================================
# Utilidades de texto (tokenización simple + stopwords + TF fallback)
# ============================================================================
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}")

def _default_stopwords_es_en() -> set[str]:
    """Stopwords básicas ES/EN (reducidas, editables)."""
    es = {
        "el","la","los","las","un","una","unos","unas",
        "de","del","al","a","y","o","u","en","por","para","con",
        "se","su","sus","es","son","ser","era","fue","han","que",
        "como","más","menos","entre","sobre","ya","si","sí","no",
        "lo","le","les","este","esta","estos","estas","ese","esa",
        "eso","esas","esos","esto","muy","también","porque"
    }
    en = {
        "the","a","an","and","or","of","in","on","for","to","by",
        "is","are","be","was","were","as","at","from","that","this",
        "these","those","with","it","its","into","over","under"
    }
    return es | en

def _tokens(text: str, min_len: int) -> List[str]:
    """Tokenizador simple (sin dependencias externas)."""
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if len(t) >= min_len]

def _top_keywords_freq(
    texts: Iterable[str],
    stopwords: set[str],
    top_k: int,
    min_len: int,
) -> List[str]:
    """Fallback por frecuencia: cuenta tokens y devuelve top_k (sin stopwords)."""
    counter = Counter()
    for t in texts:
        t_norm = prepare_text_for_topic(t)
        for tok in _tokens(t_norm, min_len):
            if tok not in stopwords:
                counter[tok] += 1
    return [w for (w, _) in counter.most_common(top_k)]

def _clean_keywords(words: List[str], stopwords: set[str], top_k: int, min_len: int) -> List[str]:
    """Filtra stopwords y tokens muy cortos; trunca a top_k sin alterar el orden original."""
    seen = []
    for w in words:
        wl = (w or "").strip().lower()
        if len(wl) < min_len or wl in stopwords:
            continue
        if wl not in seen:
            seen.append(wl)
        if len(seen) >= top_k:
            break
    return seen


# ============================================================================
# DOC-LEVEL CONTEXTUALIZATION
# ============================================================================
def run_contextizer_on_doc(
    ir_path: str,
    cfg: TopicModelConfig,
    outdir: Path | str = "outputs_doc_topics",
) -> None:
    """
    Enriquecer un DocumentIR con tópicos globales y guardar *_doc_topics.json.

    Estructura de salida (inyectada):
      data["meta"]["topics_doc"] = TopicsDocMeta(...)
    """
    set_global_seeds(cfg.seed)
    p = Path(ir_path)
    if not p.exists():
        logger.error("[Contextizer-DOC] No existe el archivo IR: %s", ir_path)
        return

    # 1) Carga del IR y extracción de textos por bloque
    data = json.loads(p.read_text(encoding="utf-8"))
    blocks = [
        blk.get("text", "").strip()
        for page in data.get("pages", [])
        for blk in page.get("blocks", [])
        if isinstance(blk, dict) and isinstance(blk.get("text"), str)
    ]
    texts = [t for t in blocks if t]
    n_samples = len(texts)

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # Config stopwords
    stopwords = set(cfg.stopwords) if cfg.stopwords else _default_stopwords_es_en()

    def _inject_and_save(reason: str, topics_items: List[TopicItem], keywords_global: List[str], outlier_ratio: float | None = None):
        data.setdefault("meta", {})
        payload = TopicsDocMeta(
            reason=reason,
            created_at=datetime.utcnow().isoformat(),
            n_samples=n_samples,
            n_topics=len(topics_items),
            keywords_global=keywords_global,
            topics=topics_items,
            outlier_ratio=outlier_ratio
        )
        data["meta"]["topics_doc"] = payload.model_dump(mode="json")
        out_path = outdir_p / p.name.replace(".json", "_doc_topics.json")
        atomic_write_json(data, out_path)
        logger.info("[CONTEXTIZE-DOC OK] → %s", out_path)

    if n_samples == 0:
        logger.warning("[Contextizer-DOC] Sin texto procesable en: %s", ir_path)
        _inject_and_save("doc-fallback-small", [], [])
        return

    # 2) Si hay muy pocas muestras, usar fallback por frecuencia
    if n_samples < 3:
        logger.info("[Contextizer-DOC] Pocas muestras (n=%d). Usando fallback de frecuencia.", n_samples)
        kws = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
        topics = [TopicItem(topic_id=0, count=n_samples, exemplar=texts[0], keywords=kws)]
        _inject_and_save("doc-fallback-small", topics, kws)
        return

    # 3) Embeddings + BERTopic con UMAP ajustado al tamaño del corpus
    try:
        logger.info("[Contextizer-DOC] Generando embeddings con %s", cfg.embedding_model)
        embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)

        # Caché opcional de embeddings
        emb = None
        if cfg.cache_dir:
            cache_path = embeddings_cache_path(Path(cfg.cache_dir), cfg.embedding_model, [prepare_text_for_topic(t) for t in texts])
            emb = try_load_embeddings(cache_path)

        if emb is None:
            emb = embedder.encode([prepare_text_for_topic(t) for t in texts], show_progress_bar=False)
            if cfg.cache_dir:
                save_embeddings(cache_path, emb)

        # UMAP con vecinos dinámicos: evita errores con corpus pequeño
        umap_neighbors = min(cfg.umap_n_neighbors, max(2, n_samples - 1))
        umap_model = UMAP(
            n_neighbors=umap_neighbors,
            n_components=cfg.umap_n_components,
            random_state=cfg.seed,
            metric=cfg.hdbscan_metric,
        )

        nr_topics = cfg.nr_topics if (cfg.nr_topics and cfg.nr_topics > 0) else None
        topic_model = BERTopic(
            embedding_model=embedder,
            nr_topics=nr_topics,
            min_topic_size=cfg.min_topic_size,
            umap_model=umap_model,
            verbose=False,
        )

        logger.info("[Contextizer-DOC] Descubriendo tópicos con BERTopic...")
        topics_idx, probs = topic_model.fit_transform(texts, emb)
        topics_idx = np.array(topics_idx)

        # 4) Construcción de salida por tópico (excluyendo outliers -1)
        uniq = sorted(set(int(t) for t in topics_idx if int(t) != -1))
        topic_items: List[TopicItem] = []
        for t_id in uniq:
            mask = (topics_idx == t_id)
            count = int(mask.sum())
            exemplar_idx = int(np.where(mask)[0][0])
            raw_words = [w for (w, _s) in (topic_model.get_topic(t_id) or [])]
            clean_words = _clean_keywords(raw_words, stopwords, cfg.max_keywords_per_topic, cfg.min_token_len)
            topic_items.append(TopicItem(
                topic_id=t_id,
                count=count,
                exemplar=texts[exemplar_idx],
                keywords=clean_words
            ))

        # Si nada pasó el filtro (todo outlier), hacer fallback
        if not topic_items:
            logger.warning("[Contextizer-DOC] Todos los bloques quedaron como outliers. Fallback de frecuencia.")
            kws = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
            topics = [TopicItem(topic_id=0, count=n_samples, exemplar=texts[0], keywords=kws)]
            _inject_and_save("doc-fallback-small", topics, kws)
        else:
            global_keywords = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
            outlier_ratio = float(np.mean(topics_idx == -1))
            _inject_and_save("doc-level", topic_items, global_keywords, outlier_ratio=outlier_ratio)

    except Exception as e:
        logger.error("[Contextizer-DOC] BERTopic falló: %s", e)
        kws = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
        topics = [TopicItem(topic_id=0, count=n_samples, exemplar=texts[0], keywords=kws)]
        _inject_and_save("doc-fallback-error", topics, kws)


# ============================================================================
# CHUNK-LEVEL CONTEXTUALIZATION
# ============================================================================
def run_contextizer_on_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """
    Enriquecer un DocumentChunks con tópicos locales.

    Para cada chunk se agrega:
      chunk["topic"] = { "topic_id": int, "keywords": [...], "prob": float | None }

    Y a nivel global:
      data["meta"]["topics_chunks"] = TopicsChunksMeta(...)
    """
    set_global_seeds(cfg.seed)

    p = Path(chunk_path)
    if not p.exists():
        logger.error("[Contextizer-CHUNKS] No existe el archivo: %s", chunk_path)
        return

    data = json.loads(p.read_text(encoding="utf-8"))
    chunks = data.get("chunks", []) or []
    texts = [str(c.get("text", "")).strip() for c in chunks if c.get("text")]
    n_samples = len(texts)

    # Config stopwords
    stopwords = set(cfg.stopwords) if cfg.stopwords else _default_stopwords_es_en()

    def _inject_and_persist(reason: str, topics_items: List[TopicItem], keywords_global: List[str]):
        data.setdefault("meta", {})
        payload = TopicsChunksMeta(
            reason=reason,
            created_at=datetime.utcnow().isoformat(),
            n_samples=n_samples,
            n_topics=len(topics_items),
            keywords_global=keywords_global,
            topics=topics_items
        )
        data["meta"]["topics_chunks"] = payload.model_dump(mode="json")
        # Persistir cambios en el mismo archivo
        atomic_write_json(data, p)
        logger.info("[CONTEXTIZE-CHUNKS OK] %s", chunk_path)

    if n_samples == 0:
        logger.warning("[Contextizer-CHUNKS] Sin texto útil en: %s", chunk_path)
        _inject_and_persist("chunk-fallback-small", [], [])
        return

    # Caso 1–2 chunks: fallback robusto (evita edge-cases de UMAP/HDBSCAN)
    if n_samples < 3:
        logger.info("[Contextizer-CHUNKS] Pocas muestras (n=%d). Fallback de frecuencia.", n_samples)
        keywords = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
        for c in chunks:
            if c.get("text"):
                c["topic"] = ChunkTopic(topic_id=0, keywords=keywords, prob=1.0).model_dump(mode="json")
        topics = [TopicItem(topic_id=0, count=n_samples, exemplar=texts[0], keywords=keywords)]
        _inject_and_persist("chunk-fallback-small", topics, keywords)
        return

    # Caso n>=3: intentar BERTopic
    try:
        logger.info("[Contextizer-CHUNKS] Generando embeddings con %s", cfg.embedding_model)
        embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)

        # Caché opcional
        emb = None
        if cfg.cache_dir:
            cache_path = embeddings_cache_path(Path(cfg.cache_dir), cfg.embedding_model, [prepare_text_for_topic(t) for t in texts])
            emb = try_load_embeddings(cache_path)

        if emb is None:
            emb = embedder.encode([prepare_text_for_topic(t) for t in texts], show_progress_bar=False)
            if cfg.cache_dir:
                save_embeddings(cache_path, emb)

        umap_neighbors = min(cfg.umap_n_neighbors, max(2, n_samples - 1))
        umap_model = UMAP(
            n_neighbors=umap_neighbors,
            n_components=cfg.umap_n_components,
            random_state=cfg.seed,
            metric=cfg.hdbscan_metric,
        )

        nr_topics = cfg.nr_topics if (cfg.nr_topics and cfg.nr_topics > 0) else None
        topic_model = BERTopic(
            embedding_model=embedder,
            nr_topics=nr_topics,
            min_topic_size=cfg.min_topic_size,
            umap_model=umap_model,
            verbose=False,
            calculate_probabilities=True,
        )

        topics_idx, probs = topic_model.fit_transform(texts, emb)
        topics_idx = np.array(topics_idx)

        # Palabras por tópico (limpias)
        uniq = sorted(set(int(t) for t in topics_idx if int(t) != -1))
        topic_keywords_cache: Dict[int, List[str]] = {}
        for t_id in uniq:
            raw_words = [w for (w, _s) in (topic_model.get_topic(t_id) or [])]
            topic_keywords_cache[t_id] = _clean_keywords(
                raw_words, stopwords, cfg.max_keywords_per_topic, cfg.min_token_len
            )

        # Asignación por chunk
        for i, c in enumerate(chunks):
            if not c.get("text"):
                continue
            t_id = int(topics_idx[i])
            prob = float(np.max(probs[i])) if (probs is not None and len(probs.shape) == 2 and probs.shape[1] > 0) else None
            if t_id == -1:
                # outlier → keywords locales fallback
                kw = _top_keywords_freq([texts[i]], stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
                c["topic"] = ChunkTopic(topic_id=-1, keywords=kw, prob=prob).model_dump(mode="json")
            else:
                kw = topic_keywords_cache.get(t_id, [])
                c["topic"] = ChunkTopic(topic_id=t_id, keywords=kw, prob=prob).model_dump(mode="json")

        # Resumen global
        topics_summary: List[TopicItem] = []
        for t_id in uniq:
            mask = (topics_idx == t_id)
            count = int(mask.sum())
            exemplar_idx = int(np.where(mask)[0][0])
            topics_summary.append(
                TopicItem(topic_id=t_id, count=count, exemplar=texts[exemplar_idx], keywords=topic_keywords_cache.get(t_id, []))
            )

        global_keywords = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
        _inject_and_persist("chunk-level", topics_summary, global_keywords)

    except Exception as e:
        logger.error("[Contextizer-CHUNKS] BERTopic falló: %s", e)
        keywords = _top_keywords_freq(texts, stopwords, cfg.fallback_max_keywords, cfg.min_token_len)
        for c in chunks:
            if c.get("text"):
                c["topic"] = ChunkTopic(topic_id=0, keywords=keywords, prob=1.0).model_dump(mode="json")
        topics = [TopicItem(topic_id=0, count=n_samples, exemplar=texts[0], keywords=keywords)]
        _inject_and_persist("chunk-fallback-error", topics, keywords)
