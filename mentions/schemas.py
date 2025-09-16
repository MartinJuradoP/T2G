# -*- coding: utf-8 -*-
"""
mentions/schemas.py

Esquemas Pydantic para la etapa Mentions (NER/RE).
- Comentarios en ES para claridad; identificadores en inglés por coherencia del repo.
- Entrada: DocumentSentences (sentence_filter.schemas)
- Salida: DocumentMentions (este archivo)

Trazabilidad y normalización:
- Cada mención incluye offsets y meta.
- 'norm' y 'norm_hints' son placeholders para la etapa de Normalización.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple


class EntityMention(BaseModel):
    """
    Mención puntual de una ENTIDAD dentro de una oración.
    """
    id: str
    text: str
    label: str                              # PERSON, ORG, LOC, DATE, MONEY, EMAIL, URL, TITLE, DEGREE, PRODUCT, ...
    canonical_label: Optional[str] = None   # ES/EN → forma canónica
    sentence_idx: int
    sentence_id: str
    char_span_in_sentence: Tuple[int, int]  # [start, end)
    token_span_in_sentence: Optional[Tuple[int, int]] = None
    char_span_in_doc: Optional[Tuple[int, int]] = None
    conf: float = 0.0
    source: str = "regex|spacy|phrase|transformer|triple"
    lang: str = "und"
    lang_source: Optional[str] = None
    norm: Optional[str] = None              # reservado para Normalización
    entity_subtype: Optional[str] = None    # ej. ORG:company|university
    meta: Dict[str, Any] = Field(default_factory=dict)  # chunk_id, page_span, filtros, etc.
    norm_hints: Dict[str, Any] = Field(default_factory=dict)


class RelationMention(BaseModel):
    """
    Mención de RELACIÓN intra-oración conectando dos EntityMention por ID.
    """
    id: str
    subj_entity_id: str
    obj_entity_id: str
    label: str
    canonical_label: Optional[str] = None
    sentence_idx: int
    sentence_id: str
    char_span_in_sentence: Tuple[int, int]
    conf: float = 0.0
    source: str = "regex|dep|matcher|transformer|triple"
    lang: str = "und"
    surface: Optional[str] = None            # fragmento soporte en la oración
    evidence: Optional[str] = None           # texto de evidencia (opcional)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DocumentMentions(BaseModel):
    """
    Contenedor de menciones para un documento.
    """
    doc_id: str
    entities: List[EntityMention]
    relations: List[RelationMention]
    meta: Dict[str, Any] = Field(default_factory=dict)  # params, counters, timing, version
