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
    """Ejecuta el modo híbrido de contextualización a nivel documento.

    Este reemplazo se activa si `should_use_hybrid_doc()` devuelve True.
    Opera sin UMAP ni BERTopic, usando DBSCAN + TF-IDF/KeyBERT + MMR.
    """
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

    # Embeddings globales
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
    emb = embedder.encode(texts, show_progress_bar=False)

    # Clustering semántico
    labels, n_clusters = cluster_by_density(emb)
    topics = build_topic_items(texts, labels, embedder, top_k=cfg.max_keywords_per_topic)

    # Keywords globales (TF-IDF + KeyBERT)
    merged_kw, keybert_kw, emb_kw = fuse_keywords(texts, embedder, top_k=cfg.fallback_max_keywords)
    mmr_kw = mmr_filter(merged_kw, emb_kw, top_k=cfg.max_keywords_per_topic)

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
    data["meta"]["topics_doc"] = meta.model_dump(mode="json")

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    out_path = outdir_p / p.name.replace(".json", "_doc_topics.json")
    atomic_write_json(data, out_path)
    logger.info("[HYBRID-DOC OK] %s (topics=%d)", out_path, len(topics))


# ──────────────────────────────────────────────────────────────────────────────
# CHUNK-LEVEL HYBRID CONTEXTUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def run_hybrid_contextizer_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """Ejecuta contextualización híbrida sobre chunks locales.

    Combina DBSCAN + fusión de keywords + MMR para enriquecer cada chunk con
    `topic_id`, `keywords`, y `prob≈1.0`. Mantiene compatibilidad completa con
    `topics_chunks`.
    """
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

    # Embeddings + clustering
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
    emb = embedder.encode(texts, show_progress_bar=False)

    labels, n_clusters = cluster_by_density(emb)
    topics = build_topic_items(texts, labels, embedder, top_k=cfg.max_keywords_per_topic)

    # Asignación por chunk (simplificada: cluster más cercano)
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
    merged_kw, keybert_kw, emb_kw = fuse_keywords(texts, embedder, top_k=cfg.fallback_max_keywords)
    mmr_kw = mmr_filter(merged_kw, emb_kw, top_k=cfg.max_keywords_per_topic)

    meta = TopicsChunksMeta(
        reason="chunk-hybrid",
        created_at=datetime.utcnow().isoformat(),
        n_samples=n,
        n_topics=len(topics),
        keywords_global=mmr_kw,
        topics=topics,
    )
    data.setdefault("meta", {})
    data["meta"]["topics_chunks"] = meta.model_dump(mode="json")

    atomic_write_json(data, p)
    logger.info("[HYBRID-CHUNKS OK] %s (topics=%d)", chunk_path, len(topics))
