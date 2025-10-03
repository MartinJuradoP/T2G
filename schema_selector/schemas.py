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
import json


# ---------------------------------------------------------------------------
# Tipos básicos
# ---------------------------------------------------------------------------

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Ontología
# ---------------------------------------------------------------------------

class AttributeDef(BaseModel):
    """Definición de un atributo dentro de una entidad."""
    name: str = Field(..., description="Nombre lógico del atributo (ej. 'fecha_nacimiento').")
    type: str = Field("string", description="Tipo de dato: string | number | date | id | code.")
    required: bool = Field(False, description="Si el atributo es obligatorio en todas las instancias.")
    description: Optional[str] = Field(None, description="Descripción semántica del atributo.")

    @field_validator("type")
    def check_valid_type(cls, v):
        allowed = {"string", "number", "date", "id", "code"}
        if v not in allowed:
            raise ValueError(f"Tipo no válido: {v}. Debe estar en {allowed}")
        return v


class EntityTypeDef(BaseModel):
    """Definición de un tipo de entidad dentro de un dominio."""
    name: str
    aliases: List[str] = Field(default_factory=list)
    attributes: List[AttributeDef] = Field(default_factory=list)
    description: Optional[str] = None


class RelationTypeDef(BaseModel):
    """Definición de una relación entre entidades dentro de un dominio."""
    name: str
    head: str
    tail: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class OntologyDomain(BaseModel):
    """Dominio ontológico con sus entidades, relaciones y alias."""
    domain: str
    aliases: List[str] = Field(default_factory=list)
    entity_types: List[EntityTypeDef] = Field(default_factory=list)
    relation_types: List[RelationTypeDef] = Field(default_factory=list)
    label_vecs: Dict[str, List[Number]] = Field(default_factory=dict)

    def list_entities(self) -> List[str]:
        """Devuelve los nombres de las entidades definidas en este dominio."""
        return [et.name for et in self.entity_types]


class OntologyRegistry(BaseModel):
    """Registro global de dominios disponibles para selección adaptativa."""
    domains: List[OntologyDomain] = Field(default_factory=list)

    def get_domain(self, name: str) -> Optional[OntologyDomain]:
        """Devuelve un dominio por nombre (case-insensitive)."""
        for d in self.domains:
            if d.domain.lower() == name.lower():
                return d
        return None


# ---------------------------------------------------------------------------
# Evidencia y Scoring
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Evidencia que justifica una puntuación (para auditoría)."""
    kind: str = Field(..., description="Tipo: keyword | embedding | stat | heuristic.")
    detail: Dict[str, Union[str, Number]] = Field(default_factory=dict)

    def __str__(self):
        return f"[{self.kind}] {self.detail}"


class EntityTypeScore(BaseModel):
    """Puntuación de un tipo de entidad dentro de un dominio."""
    type_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)

    @field_validator("type_name")
    def strip_name(cls, v):
        return v.strip()

    def summary(self) -> str:
        return f"{self.type_name}: {self.score:.2f}"


class DomainScore(BaseModel):
    """Puntuación global de un dominio, incluyendo breakdown por entidades."""
    domain: str
    score: float = Field(..., ge=0.0, le=1.0)
    entity_type_scores: List[EntityTypeScore] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    def top_entity_types(self, k: int = 3) -> List[str]:
        """Devuelve los top-k entity types mejor puntuados del dominio."""
        sorted_types = sorted(self.entity_type_scores, key=lambda x: x.score, reverse=True)
        return [et.type_name for et in sorted_types[:k]]


# ---------------------------------------------------------------------------
# Selección de Esquemas (Chunk / Documento / Final)
# ---------------------------------------------------------------------------

class ChunkSchemaSelection(BaseModel):
    """Selección de esquemas aplicada a un chunk."""
    chunk_id: str
    domain_scores: List[DomainScore] = Field(default_factory=list)
    top_domain: Optional[str] = None
    ambiguous: bool = False


class DocSchemaSelection(BaseModel):
    """Selección de esquemas a nivel de documento completo."""
    doc_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    domain_scores: List[DomainScore] = Field(default_factory=list)
    top_domains: List[str] = Field(default_factory=list)
    ambiguous: bool = False


class SchemaSelection(BaseModel):
    """Selección final de esquemas (documento + chunks)."""
    doc: DocSchemaSelection
    chunks: List[ChunkSchemaSelection] = Field(default_factory=list)
    meta: Dict[str, Union[str, int, float, List[str]]] = Field(
        default_factory=dict,
        description="Metadatos: hiperparámetros, versión, dominios incluidos, tiempos, etc."
    )

    def to_json(self, indent: int = 2) -> str:
        """Serializa el objeto a JSON con indentación."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a dict estándar."""
        return self.model_dump()

    def summary(self) -> str:
        """Devuelve un resumen en texto plano."""
        return (
            f"Doc={self.doc.doc_id}, "
            f"TopDomains={self.doc.top_domains}, "
            f"Chunks={len(self.chunks)}, "
            f"Ambiguous={self.doc.ambiguous}"
        )


# ---------------------------------------------------------------------------
# Configuración del Adaptive Schema Selector
# ---------------------------------------------------------------------------

class SelectorConfig(BaseModel):
    """
    Configuración del Adaptive Schema Selector.

    Hiperparámetros:
    - always_include: dominios que siempre deben incluirse (ej. 'generic').
    - min_topic_conf: confianza mínima para considerar un tópico.
    - max_domains: máximo de dominios seleccionables.
    - allow_fallback_generic: si True, siempre incluir 'generic'.
    - alpha_kw, beta_emb, gamma_prior: pesos para combinación de scores.
    - ambiguity_threshold: umbral para marcar ambigüedad.
    - topk: número máximo de dominios a devolver.
    - version: para trazabilidad.
    """
    always_include: List[str] = Field(default_factory=lambda: ["generic"])
    min_topic_conf: float = Field(0.2, ge=0.0, le=1.0)
    max_domains: int = Field(5, ge=1)
    allow_fallback_generic: bool = Field(True)

    alpha_kw: float = Field(0.6, ge=0.0, le=1.0)
    beta_emb: float = Field(0.3, ge=0.0, le=1.0)
    gamma_prior: float = Field(0.1, ge=0.0, le=1.0)

    ambiguity_threshold: float = Field(0.15, ge=0.0, le=1.0)
    topk: int = Field(5, ge=1)

    version: str = Field("selector.v1")

    def __str__(self) -> str:
        return (
            f"SelectorConfig("
            f"always_include={self.always_include}, "
            f"min_topic_conf={self.min_topic_conf}, "
            f"max_domains={self.max_domains}, "
            f"allow_fallback_generic={self.allow_fallback_generic}, "
            f"alpha_kw={self.alpha_kw}, beta_emb={self.beta_emb}, gamma_prior={self.gamma_prior}, "
            f"ambiguity_threshold={self.ambiguity_threshold}, topk={self.topk}, "
            f"version={self.version})"
        )


# ---------------------------------------------------------------------------
# Reconstrucción de Modelos (obligatorio Pydantic v2)
# ---------------------------------------------------------------------------

SchemaSelection.model_rebuild()
DocSchemaSelection.model_rebuild()
ChunkSchemaSelection.model_rebuild()
DomainScore.model_rebuild()
EntityTypeScore.model_rebuild()
Evidence.model_rebuild()
OntologyDomain.model_rebuild()
EntityTypeDef.model_rebuild()
RelationTypeDef.model_rebuild()
AttributeDef.model_rebuild()
OntologyRegistry.model_rebuild()
SelectorConfig.model_rebuild()
