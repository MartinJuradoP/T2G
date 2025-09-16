# -*- coding: utf-8 -*-
"""
Paquete Mentions (NER/RE) para T2G.
Expone los contratos y el extractor principal.
"""
from .schemas import EntityMention, RelationMention, DocumentMentions
from .ner_re import NERREExtractor, MentionConfig

__all__ = [
    "EntityMention",
    "RelationMention",
    "DocumentMentions",
    "NERREExtractor",
    "MentionConfig",
]
