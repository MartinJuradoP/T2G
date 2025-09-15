# triples/schemas.py
# -*- coding: utf-8 -*-
"""
Contratos Pydantic v2 para la etapa Triples (dep.).
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class TripleIR(BaseModel):
    """Triple ligero (S, R, O) con metadatos de trazabilidad."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="sha1 corto sobre doc_id + sent_idx + offsets + rule + SRO")
    subject: str
    relation: str
    object: str
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "{ sentence_id, sentence_idx, char_span_in_sentence:[start,end], "
            "dep_rule, conf, lang }"
        ),
    )


class DocumentTriples(BaseModel):
    """Salida por documento para triples extraídos por dependencias/heurísticas."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    strategy: Literal["dep-triples-v1"] = "dep-triples-v1"
    version: Literal["1.0"] = "1.0"
    triples: List[TripleIR] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="{ counters:{...}, params:{...} }",
    )
