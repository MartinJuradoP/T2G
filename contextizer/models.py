# -*- coding: utf-8 -*-
"""
models.py — Wrapper de Contextualización (OOP)
==============================================

Rol
---
Ofrece una interfaz orientada a objetos para consumir la contextualización,
evitando duplicar la lógica procedural de contextizer.py.

Uso
---
from .schemas import TopicModelConfig
from .models import ContextizerModel

ctx = ContextizerModel(TopicModelConfig())
ctx.run_on_doc_path("outputs_ir/XXX.json", outdir="outputs_doc_topics")
ctx.run_on_chunks_path("outputs_chunks/XXX.json")
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from .schemas import TopicModelConfig
from .contextizer import run_contextizer_on_doc, run_contextizer_on_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContextizerModel:
    """Fachada amigable: delega en las funciones procedurales."""

    def __init__(self, config: TopicModelConfig):
        self.cfg = config

    # ---------------------------
    # Doc-level (por ruta)
    # ---------------------------
    def run_on_doc_path(self, ir_path: str, outdir: str = "outputs_doc_topics") -> None:
        """
        Ejecuta doc-level contextizer sobre una ruta IR (.json) y persiste en `outdir`.
        """
        run_contextizer_on_doc(ir_path, self.cfg, outdir=outdir)

    # ---------------------------
    # Chunk-level (por ruta)
    # ---------------------------
    def run_on_chunks_path(self, chunk_path: str) -> None:
        """
        Ejecuta chunk-level contextizer sobre una ruta de DocumentChunks (.json) y persiste en sitio.
        """
        run_contextizer_on_chunks(chunk_path, self.cfg)
