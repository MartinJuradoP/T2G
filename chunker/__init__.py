# -*- coding: utf-8 -*-
"""
chunker — HybridChunker (DocumentIR+Topics → DocumentChunks)

Expone la API de alto nivel del subsistema:
- chunker.run()
- chunker.load_ir_with_topics()
- chunker.save_chunks()
"""
from .chunker import run, load_ir_with_topics, save_chunks
from .schemas import (
    Chunk,
    ChunkingConfig,
    DocumentChunks,
    ChunkSourceSpan,
    TopicHints,
)
