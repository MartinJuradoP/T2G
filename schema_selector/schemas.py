# -*- coding: utf-8 -*-
"""
schemas.py — Contratos Pydantic para Adaptive Schema Selector.

Este módulo define los contratos de datos para la etapa de selección de esquemas
dentro del pipeline T2G (Knowledge Graph from Documents).

Incluye:
- Definiciones de ontología (atributos, entidades, relaciones, dominios).
- Evidencia y scores (EntityTypeScore, DomainScore).
- Selección de esquemas a nivel de chunk y documento.
- Selección final (SchemaSelection).
- Configuración del Adaptive Schema Selector (SelectorConfig).
- Métodos utilitarios: validación, serialización, resumen.

Compatible con **Pydantic v2** y diseñado para ser:
- **Auditables**: cada score se justifica con evidencias.
- **Extensibles**: se pueden añadir dominios, entidades, atributos y metadatos.
- **Robustos**: validadores internos y defaults seguros.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

Number = Union[int, float]


# ---------------------------------------------------------------------------
# 🔹 Ontología (sin cambios de fondo; se añaden validadores ligeros)
# ---------------------------------------------------------------------------

class AttributeDef(BaseModel):
    name: str = Field(..., description="Nombre lógico del atributo.")
    type: str = Field("string", description="string|number|date|id|code")
    required: bool = False
    description: Optional[str] = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {"string", "number", "date", "id", "code"}
        if v not in allowed:
            raise ValueError(f"Tipo inválido '{v}'. Debe ser uno de {allowed}")
        return v


class EntityTypeDef(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    attributes: List[AttributeDef] = Field(default_factory=list)
    description: Optional[str] = None


class RelationTypeDef(BaseModel):
    name: str
    head: str
    tail: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class OntologyDomain(BaseModel):
    domain: str
    aliases: List[str] = Field(default_factory=list)
    entity_types: List[EntityTypeDef] = Field(default_factory=list)
    relation_types: List[RelationTypeDef] = Field(default_factory=list)
    # Opcional: prototipos semánticos del dominio (si están disponibles)
    label_vecs: Dict[str, List[Number]] = Field(default_factory=dict)


class OntologyRegistry(BaseModel):
    domains: List[OntologyDomain] = Field(default_factory=list)

    def get_domain(self, name: str) -> Optional[OntologyDomain]:
        for d in self.domains:
            if d.domain.lower() == name.lower():
                return d
        return None


# ---------------------------------------------------------------------------
# 🔹 Evidencias, trazabilidad y scoring
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Evidencia auditable que respalda un score."""
    kind: str = Field(..., description="keyword|embedding|topic|context|prior")
    detail: Dict[str, Any] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    """
    Rastro de decisión cuantitativo por dominio:
    - Sub-scores por señal (K/E/T/C/P)
    - Pesos efectivos
    - Contribuciones al score final
    """
    domain: str
    keyword_score: float = 0.0  # K
    embedding_score: float = 0.0  # E
    topic_score: float = 0.0  # T (topic_affinity)
    context_score: float = 0.0  # C (cohesion, health, novelty, redundancy, richness)
    prior: float = 0.0  # P
    weights: Dict[str, float] = Field(default_factory=dict)  # {"alpha_kw":..., ...}
    contributions: Dict[str, float] = Field(default_factory=dict)  # {"K": α*K, ...}
    final_score: float = 0.0
    used_signals: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class EntityTypeScore(BaseModel):
    type_name: str
    score: float
    evidence: List[Evidence] = Field(default_factory=list)


class DomainScore(BaseModel):
    domain: str
    score: float
    entity_type_scores: List[EntityTypeScore] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    decision_trace: Optional[DecisionTrace] = None


# ---------------------------------------------------------------------------
# 🔹 Selecciones por Chunk y por Documento
# ---------------------------------------------------------------------------

class ChunkSchemaSelection(BaseModel):
    chunk_id: str
    domain_scores: List[DomainScore] = Field(default_factory=list)
    top_domain: Optional[str] = None
    selected_schema: Optional[str] = None
    schema_confidence: float = 0.0
    explanation: Optional[str] = None
    signals_used: List[str] = Field(default_factory=list)
    weights_used: Dict[str, float] = Field(default_factory=dict)
    ambiguous: bool = False


class DocSchemaSelection(BaseModel):
    doc_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    domain_scores: List[DomainScore] = Field(default_factory=list)
    top_domains: List[str] = Field(default_factory=list)
    selected_schema: Optional[str] = None
    schema_confidence: float = 0.0
    explanation: Optional[str] = None
    signals_used: List[str] = Field(default_factory=list)
    weights_used: Dict[str, float] = Field(default_factory=dict)
    ambiguous: bool = False
    


class SchemaSelection(BaseModel):
    doc: DocSchemaSelection
    chunks: List[ChunkSchemaSelection] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Doc={self.doc.doc_id}, TopDomains={self.doc.top_domains}, "
            f"Schema={self.doc.selected_schema}, Confidence={self.doc.schema_confidence:.2f}, "
            f"Chunks={len(self.chunks)}"
        )


# ---------------------------------------------------------------------------
# 🔹 Configuración del Selector (v2)
# ---------------------------------------------------------------------------

class SelectorConfig(BaseModel):
    # Dominios
    always_include: List[str] = Field(default_factory=lambda: ["generic"])
    max_domains: int = 5
    topk: int = 5
    allow_fallback_generic: bool = True

    # Pesos (α, β, γ, δ, ε) para K, E, C, T, P respectivamente
    alpha_kw: float = 0.30
    beta_emb: float = 0.20
    gamma_ctx: float = 0.25
    delta_top: float = 0.20  # topic_affinity
    epsilon_prior: float = 0.05

    # Umbrales y temperatura softmax
    ambiguity_threshold: float = 0.12
    fallback_threshold: float = 0.40
    softmax_temperature: float = 0.85  # T < 1 endurece diferencias

    # Heurísticas de textos cortos
    min_doc_tokens_for_domain: int = 14  # por debajo, favorece 'generic' salvo evidencia fuerte

    # Versión
    version: str = "selector.v2"

    def weights(self) -> Dict[str, float]:
        return {
            "alpha_kw": self.alpha_kw,
            "beta_emb": self.beta_emb,
            "gamma_ctx": self.gamma_ctx,
            "delta_top": self.delta_top,
            "epsilon_prior": self.epsilon_prior,
        }