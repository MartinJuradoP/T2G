# -*- coding: utf-8 -*-
"""
schema_selector package — Selección adaptativa de esquemas de entidades.

Este paquete implementa la Etapa 5 del pipeline T2G:
- Dado un DocumentChunks (IR + topics heredados y locales),
  selecciona dinámica y contextualmente dominios y tipos de entidades
  para la etapa siguiente (Mentions/NER-RE).

Arquitectura:
- schemas.py     : contratos Pydantic v2
- registry.py    : ontologías/domains iniciales (extensibles)
- utils.py       : utilidades puras (normalización, similitud, etc.)
- metrics.py     : métricas de calidad y ambigüedad
- selector.py    : lógica de selección (keywords + embeddings opcionales)
"""
from .schemas import (
    SchemaSelection, DocSchemaSelection, ChunkSchemaSelection,
    OntologyRegistry, OntologyDomain, EntityTypeDef, RelationTypeDef, AttributeDef
)
from .registry import REGISTRY  # Registro por defecto
from .selector import select_schemas
