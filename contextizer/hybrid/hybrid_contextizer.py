# -*- coding: utf-8 -*-
"""
contextizer.hybrid.hybrid_contextizer — Ejecutor principal del Contextizer Híbrido
=================================================================================

Resumen
-------
Este módulo integra todos los componentes híbridos (analyzers, clustering,
keywords, MMR) para reemplazar o complementar a BERTopic en los casos donde
el sistema detecta documentos o chunks pequeños, ruidosos o multitema.

Propósito
---------
Garantizar que incluso en contextos **de baja densidad semántica** (pocos
fragmentos, textos cortos, mezcla de tópicos) el pipeline T2G produzca una
contextualización coherente, estable y explicativa, sin romper contratos.

Compatibilidad
--------------
✔ Misma estructura de salida que el Contextizer clásico (`topics_doc` o `topics_chunks`).  
✔ Se inyectan metadatos extra en `meta.reason = 'doc-hybrid'` o `'chunk-hybrid'`.  
✔ Integración no intrusiva con `contextizer/contextizer.py`.

Referencias
-----------
- Grootendorst (2022) — BERTopic.
- Carbonell & Goldstein (1998) — Maximal Marginal Relevance.
- Reimers & Gurevych (2019) — Sentence-BERT embeddings.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

from .density_clustering import cluster_by_density, build_topic_items
from .keyword_fusion import fuse_keywords
from .mmr import mmr_filter
from .analyzers import should_use_hybrid_doc, should_use_hybrid_chunks

from ..schemas import TopicModelConfig, TopicsDocMeta, TopicsChunksMeta
from ..utils import atomic_write_json

logger = logging.getLogger("contextizer.hybrid")
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# DOC-LEVEL HYBRID CONTEXTUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def run_hybrid_contextizer_doc(ir_path: str, cfg: TopicModelConfig, outdir: Path | str = "outputs_doc_topics") -> None:
    """Ejecuta el modo híbrido de contextualización a nivel documento."""
    p = Path(ir_path)
    if not p.exists():
        logger.error("[Hybrid-DOC] Archivo no encontrado: %s", ir_path)
        return

    data = json.loads(p.read_text(encoding="utf-8"))

    # Extracción básica de textos
    texts: List[str] = [
        blk.get("text", "").strip()
        for page in data.get("pages", [])
        for blk in page.get("blocks", [])
        if isinstance(blk, dict) and isinstance(blk.get("text"), str)
    ]
    texts = [t for t in texts if t]
    n = len(texts)

    if n == 0:
        logger.warning("[Hybrid-DOC] Documento vacío o sin texto utilizable.")
        return

    if cfg.use_hybrid:
        # Embeddings y clustering activos
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
        emb = embedder.encode(texts, show_progress_bar=False)
        labels, n_clusters = cluster_by_density(emb)
        topics = build_topic_items(texts, labels, embedder, top_k=cfg.max_keywords_per_topic)
    else:
        # Modo sin embeddings ni clustering
        embedder = None
        emb = None
        labels, n_clusters = [0] * n, 1
        topics = []
        logger.info("[Hybrid-DOC] Modo ligero activo: sin embeddings ni clustering.")


    # Keywords globales (TF-IDF + KeyBERT)
    use_kb = (cfg.use_keybert and embedder is not None)
    merged_kw, keybert_kw, emb_kw = fuse_keywords(
        texts, embedder, top_k=cfg.fallback_max_keywords, use_keybert=use_kb
    )
    if cfg.enable_mmr:
        mmr_kw = mmr_filter(merged_kw, emb_kw, top_k=cfg.max_keywords_per_topic)
    else:
        mmr_kw = merged_kw


    # Empaquetado final
    data.setdefault("meta", {})
    meta = TopicsDocMeta(
        reason="doc-hybrid",
        created_at=datetime.utcnow().isoformat(),
        n_samples=n,
        n_topics=len(topics),
        keywords_global=mmr_kw,
        topics=topics,
        outlier_ratio=None
    )

    # ──────────────────────────────────────────────────────────────
    # MÉTRICAS EXTENDIDAS — inyección directa al JSON (para Selector)
    # ──────────────────────────────────────────────────────────────
    from .metrics_ext import (
        entropy_topics,
        redundancy_score,
        keywords_diversity_ext,
        semantic_variance,
        coherence_semantic,
        topic_balance,
        keyword_redundancy_rate,
        informativeness_ratio,
    )

    metrics_ext = {}
    try:
        metrics_ext = {
            "entropy_topics": entropy_topics(meta),
            "redundancy_score": redundancy_score(meta),
            "keywords_diversity_ext": keywords_diversity_ext(meta),
            "semantic_variance": semantic_variance(meta, embedder),
            "coherence_semantic": coherence_semantic(meta, embedder),
            "topic_balance": topic_balance(meta),
            "keyword_redundancy_rate": keyword_redundancy_rate(meta),
            "informativeness_ratio": informativeness_ratio(meta),
        }
    except Exception as e:
        logger.warning(f"[Hybrid-DOC] Error calculando métricas extendidas: {e}")

    # Inyecta las métricas extendidas en el contrato JSON
    meta_dict = meta.model_dump(mode="json")
    meta_dict["metrics_ext"] = metrics_ext
    data["meta"]["topics_doc"] = meta_dict
    meta_dict["config_used"] = {
    "use_hybrid": cfg.use_hybrid,
    "use_keybert": cfg.use_keybert,
    "enable_mmr": cfg.enable_mmr,
    "hybrid_eps": cfg.hybrid_eps,
    "hybrid_min_samples": cfg.hybrid_min_samples,
    }

    # Guardado
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    out_path = outdir_p / p.name.replace(".json", "_doc_topics.json")
    atomic_write_json(data, out_path)
    logger.info("[HYBRID-DOC OK] %s (topics=%d)", out_path, len(topics))
    logger.info(
    "[HYBRID CONFIG] use_hybrid=%s | use_keybert=%s | enable_mmr=%s",
    cfg.use_hybrid,
    cfg.use_keybert,
    cfg.enable_mmr
    )

# ──────────────────────────────────────────────────────────────────────────────
# CHUNK-LEVEL HYBRID CONTEXTUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def run_hybrid_contextizer_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """Ejecuta contextualización híbrida sobre chunks locales."""
    p = Path(chunk_path)
    if not p.exists():
        logger.error("[Hybrid-CHUNKS] No existe el archivo: %s", chunk_path)
        return

    data = json.loads(p.read_text(encoding="utf-8"))
    chunks = data.get("chunks", []) or []
    texts = [str(c.get("text", "")).strip() for c in chunks if c.get("text")]
    n = len(texts)

    if n == 0:
        logger.warning("[Hybrid-CHUNKS] Sin texto válido.")
        return

    
    if cfg.use_hybrid:
        # Embeddings + clustering
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
        emb = embedder.encode(texts, show_progress_bar=False)
        labels, n_clusters = cluster_by_density(emb)
        topics = build_topic_items(texts, labels, embedder, top_k=cfg.max_keywords_per_topic)
    else:
        # Sin embeddings ni clustering
        embedder = None
        emb = None
        labels, n_clusters = [0] * n, 1
        topics = []
        logger.info("[Hybrid-CHUNKS] Modo ligero activo: sin embeddings ni clustering.")


    # Asignación por chunk
    for i, c in enumerate(chunks):
        if not c.get("text"):
            continue
        lbl = int(labels[i]) if i < len(labels) else 0
        c["topic"] = {
            "topic_id": lbl,
            "keywords": topics[lbl % len(topics)]["keywords"] if topics else [],
            "prob": 1.0,
        }

    # Meta global (summary)
    use_kb = (cfg.use_keybert and embedder is not None)
    merged_kw, keybert_kw, emb_kw = fuse_keywords(
        texts, embedder, top_k=cfg.fallback_max_keywords, use_keybert=use_kb
    )
    
    if cfg.enable_mmr:
        mmr_kw = mmr_filter(merged_kw, emb_kw, top_k=cfg.max_keywords_per_topic)
    else:
        mmr_kw = merged_kw


    meta = TopicsChunksMeta(
        reason="chunk-hybrid",
        created_at=datetime.utcnow().isoformat(),
        n_samples=n,
        n_topics=len(topics),
        keywords_global=mmr_kw,
        topics=topics,
    )
    data.setdefault("meta", {})

    from .metrics_ext import context_alignment, redundancy_penalty

    metrics_ext = {}
    try:
        metrics_ext = {
            "context_alignment": context_alignment(chunks),
            "redundancy_penalty": redundancy_penalty(emb),
        }
    except Exception as e:
        logger.warning(f"[Hybrid-CHUNKS] Error calculando métricas extendidas: {e}")

    meta_dict = meta.model_dump(mode="json")
    meta_dict["metrics_ext"] = metrics_ext
    data["meta"]["topics_chunks"] = meta_dict
    meta_dict["config_used"] = {
    "use_hybrid": cfg.use_hybrid,
    "use_keybert": cfg.use_keybert,
    "enable_mmr": cfg.enable_mmr,
    "hybrid_eps": cfg.hybrid_eps,
    "hybrid_min_samples": cfg.hybrid_min_samples,
    }


    # Guardado
    atomic_write_json(data, p)
    logger.info("[HYBRID-CHUNKS OK] %s (topics=%d)", chunk_path, len(topics))
    logger.info(
    "[HYBRID CONFIG] use_hybrid=%s | use_keybert=%s | enable_mmr=%s",
    cfg.use_hybrid,
    cfg.use_keybert,
    cfg.enable_mmr
    )

