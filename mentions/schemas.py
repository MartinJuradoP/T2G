# -*- coding: utf-8 -*-
from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class Mention(BaseModel):
    text: str = Field(..., description="Texto exacto de la mención")
    type: str = Field(..., description="Tipo de entidad (Person, Date, Contract, etc.)")
    domain: str = Field(default="generic", description="Dominio: legal, financial, etc.")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Confianza [0,1]")
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    source_chunk: Optional[str] = None

    @validator("text")
    def clean_text(cls, v):
        return v.strip()

    @validator("type")
    def upper_first(cls, v):
        return v.strip()


class MentionsOutput(BaseModel):
    doc_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    mentions: List[Mention]
    meta: Dict[str, Any]


class MentionsConfig(BaseModel):
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 4096
    confidence_threshold: float = 0.25
    outdir: str = "outputs_mentions"
