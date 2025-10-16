# -*- coding: utf-8 -*-
"""
contextizer/contextizer.py — Router adaptativo entre BERTopic y modo híbrido
==========================================================================

Este archivo controla el enrutamiento entre los modos de contextualización:

- **Modo completo (híbrido)**: usa embeddings, TF-IDF, KeyBERT y clustering por densidad (DBSCAN).
- **Modo ligero (light)**: usa únicamente TF-IDF y, opcionalmente, KeyBERT (sin embeddings ni clustering).

Objetivo:
---------
Garantizar robustez y flexibilidad para documentos extensos o breves, 
según la configuración definida en `TopicModelConfig`.



Características clave
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
import json

logger = logging.getLogger("contextizer.router")
logger.setLevel(logging.INFO)
# ──────────────────────────────────────────────────────────────────────────────
# Light Contextizer (sin embeddings ni clustering)
# ──────────────────────────────────────────────────────────────────────────────
def run_light_contextizer_doc(ir_path: str, cfg: TopicModelConfig, outdir: Path | str) -> None:
    """Modo ligero: solo TF-IDF y KeyBERT, sin embeddings ni clustering."""
    from .hybrid.hybrid_contextizer import run_hybrid_contextizer_doc
    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.use_hybrid = False
    cfg_copy.cache_embeddings = False
    cfg_copy.hybrid_min_samples = 0
    cfg_copy.hybrid_eps = 0.0
    run_hybrid_contextizer_doc(ir_path, cfg_copy, outdir)
# ──────────────────────────────────────────────────────────────────────────────
# Light Contextizer (sin embeddings ni clustering)
# ──────────────────────────────────────────────────────────────────────────────
def run_light_contextizer_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """Modo ligero: sin embeddings ni clustering, solo TF-IDF/KeyBERT."""
    from .hybrid.hybrid_contextizer import run_hybrid_contextizer_chunks
    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.use_hybrid = False
    cfg_copy.cache_embeddings = False
    cfg_copy.hybrid_min_samples = 0
    cfg_copy.hybrid_eps = 0.0
    run_hybrid_contextizer_chunks(chunk_path, cfg_copy)

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
    # ─────────────────────────────────────────────
    # Validación lógica rápida de configuración
    # ─────────────────────────────────────────────
    if cfg.enable_mmr and not cfg.use_keybert:
        logger.warning("[Router-DOC] Ignorando enable_mmr=True porque use_keybert=False.")
        cfg.enable_mmr = False
    
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

    if not cfg.use_hybrid:
        logger.info("[Router-DOC] Ejecutando modo LIGHT (sin embeddings ni clustering).")
        run_light_contextizer_doc(ir_path, cfg, outdir)
        return

    logger.info("[Router-DOC] Ejecutando modo COMPLETO (embeddings + TF-IDF + KeyBERT + clustering).")
    run_hybrid_contextizer_doc(ir_path, cfg, outdir)



# ──────────────────────────────────────────────────────────────────────────────
# CHUNK-LEVEL ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def route_contextizer_chunks(chunk_path: str, cfg: TopicModelConfig) -> None:
    """Router adaptativo chunk-level: decide entre BERTopic o modo híbrido."""
    # ─────────────────────────────────────────────
    # Validación lógica rápida de configuración
    # ─────────────────────────────────────────────
    if cfg.enable_mmr and not cfg.use_keybert:
        logger.warning("[Router-CHUNKS] Ignorando enable_mmr=True porque use_keybert=False.")
        cfg.enable_mmr = False

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

    if not cfg.use_hybrid:
        logger.info("[Router-CHUNKS] Ejecutando modo LIGHT (sin embeddings ni clustering).")
        run_light_contextizer_chunks(chunk_path, cfg)
        return

    logger.info("[Router-CHUNKS] Ejecutando modo COMPLETO (embeddings + TF-IDF + KeyBERT + clustering).")
    run_hybrid_contextizer_chunks(chunk_path, cfg)



# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "route_contextizer_doc",
    "route_contextizer_chunks",
]

logger.info("Contextizer adaptativo cargado correctamente ")