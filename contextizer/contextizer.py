# -*- coding: utf-8 -*-
"""
contextizer/contextizer.py — Router adaptativo entre BERTopic y modo híbrido
==========================================================================

Versión final
-------------
Este archivo amplía el Contextizer original para incluir el **modo híbrido**,
que decide dinámicamente si utilizar BERTopic o el nuevo subsistema híbrido
en función del contenido del documento o de los chunks.

Objetivo
--------
Garantizar robustez en todos los contextos (documentos extensos, breves,
ruidosos o multitema) sin alterar contratos ni romper el flujo original del
pipeline T2G.

Principales cambios
-------------------
-  Detección adaptativa: usa `should_use_hybrid_doc` / `should_use_hybrid_chunks`.
-  Modo híbrido automático: ejecuta `run_hybrid_contextizer_doc` / `run_hybrid_contextizer_chunks`.
-  Compatibilidad completa: `TopicsDocMeta` y `TopicsChunksMeta` sin cambios.
-  Logging informativo y trazabilidad en `meta.reason`.

Ejemplo de integración
----------------------
El sistema selecciona el modo adecuado sin intervención del usuario:

```python
# dentro de la CLI o router YAML
cfg = TopicModelConfig()
if should_use_hybrid_doc(texts, cfg, emb):
    run_hybrid_contextizer_doc(ir_path, cfg, outdir)
else:
    run_contextizer_on_doc(ir_path, cfg, outdir)
```

Autor: T2G — Equipo de IA Semántica
"""

from __future__ import annotations
import logging
from pathlib import Path

from .schemas import TopicModelConfig


# Importamos heurísticas y modos híbridos
from .hybrid.analyzers import should_use_hybrid_doc, should_use_hybrid_chunks
from .hybrid.hybrid_contextizer import run_hybrid_contextizer_doc, run_hybrid_contextizer_chunks

logger = logging.getLogger("contextizer.router")
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# DOC-LEVEL ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def route_contextizer_doc(ir_path: str, cfg: TopicModelConfig, outdir: Path | str = "outputs_doc_topics") -> None:
    """Router adaptativo doc-level: decide entre BERTopic o modo híbrido.

    Flujo:
     1 Carga los textos del documento (bloques IR).
    2️ Evalúa heurísticas → decide modo.
    3️ Ejecuta el motor correspondiente y persiste salida.
    """
    import json
    from sentence_transformers import SentenceTransformer

    p = Path(ir_path)
    if not p.exists():
        logger.error("[Router-DOC] Archivo no encontrado: %s", ir_path)
        return

    data = json.loads(p.read_text(encoding="utf-8"))
    texts = [
        blk.get("text", "").strip()
        for page in data.get("pages", [])
        for blk in page.get("blocks", [])
        if isinstance(blk, dict) and isinstance(blk.get("text"), str)
    ]
    texts = [t for t in texts if t]

    if not texts:
        logger.warning("[Router-DOC] Documento vacío, se omite.")
        return

    embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
    emb = embedder.encode(texts, show_progress_bar=False)

    # Decisión adaptativa
    if should_use_hybrid_doc(texts, cfg, emb):
        logger.info("[Router-DOC] Activando modo híbrido (heurísticas positivas).")
        run_hybrid_contextizer_doc(ir_path, cfg, outdir)
    else:
        logger.info("[Router-DOC] Modo estándar BERTopic.")
        run_contextizer_on_doc(ir_path, cfg, outdir)

# ──────────────────────────────────────────────────────────────────────────────
# CHUNK-LEVEL ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def route_contextizer_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """Router adaptativo chunk-level: decide entre BERTopic o modo híbrido."""
    import json
    from sentence_transformers import SentenceTransformer

    p = Path(chunk_path)
    if not p.exists():
        logger.error("[Router-CHUNKS] No existe el archivo: %s", chunk_path)
        return

    data = json.loads(p.read_text(encoding="utf-8"))
    chunks = data.get("chunks", []) or []
    texts = [str(c.get("text", "")).strip() for c in chunks if c.get("text")]

    if not texts:
        logger.warning("[Router-CHUNKS] Sin texto utilizable.")
        return

    embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)
    emb = embedder.encode(texts, show_progress_bar=False)

    # Decisión adaptativa
    if should_use_hybrid_chunks(texts, cfg, emb):
        logger.info("[Router-CHUNKS] Activando modo híbrido.")
        run_hybrid_contextizer_chunks(chunk_path, cfg)
    else:
        logger.info("[Router-CHUNKS] Modo estándar BERTopic.")
        run_contextizer_on_chunks(chunk_path, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "route_contextizer_doc",
    "route_contextizer_chunks",
]

logger.info("Contextizer adaptativo cargado correctamente ")