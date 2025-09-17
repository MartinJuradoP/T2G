# -*- coding: utf-8 -*-
"""
normalizer/schemas.py
=====================

Contratos Pydantic para la etapa de **Normalización** (Mentions → Entities).

Diseño:
- `NormalizedEntity`: entidad consolidada y normalizada a partir de una o más *mentions*.
- `EntitiesMeta`: metadatos de la corrida (contadores y configuración efectiva).
- `DocumentEntities`: contenedor por documento (lista de entidades + metadatos).

Notas clave:
- Compatibilidad: Pydantic v2.7+ (según tu requirements).
- `attrs` permite valores de tipos primitivos y listas de strings; además, aceptamos `None`
  como `Optional` por robustez, aunque en la implementación normal nosotros limpiamos `None`.
- `meta` está modelado con `EntitiesMeta` para mantener estructura clara:
  - `counters`: números (int/float) de uso general (ej. entidades, prototipos, etc.)
  - `config`: mapa flexible (str/bool/float/int/list/dict) para volcar configuración efectiva
    sin pelear con validación estricta (útil cuando incluye strings como "auto" o "MXN").

Ejemplo de salida (resumen):
----------------------------
{
  "doc_id": "DOC-123",
  "entities": [
    {
      "id": "ENT-a1b2c3d4e5",
      "type": "ORG",
      "name": "acme",
      "value": null,
      "attrs": {
        "org_core": "acme",
        "org_suffix": ["S.A. de C.V."],
        "org_key": "acme",
        "raw": "Acme, S.A. de C.V."
      },
      "mentions": ["E12","E34"],
      "conf": 0.93
    },
    {
      "id": "ENT-f6g7h8i9j0",
      "type": "DATE",
      "name": null,
      "value": "2023-01-12",
      "attrs": {"value_iso":"2023-01-12", "precision":"day", "raw":"12 ene 2023"},
      "mentions": ["E2"],
      "conf": 0.88
    }
  ],
  "meta": {
    "counters": {"input_mentions": 42, "kept_protos": 33, "entities": 27},
    "config": {
      "date_locale": "auto",
      "min_conf_keep": 0.66,
      "merge_threshold": 0.92,
      "canonicalize": true,
      "default_currency": "MXN"
    }
  }
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tipos y alias de utilidad
# ---------------------------------------------------------------------------

#: Tipos primitivos aceptados en atributos normalizados
AttrPrimitive = Union[str, float, int, bool]

#: Valor de atributo: primitivo o lista de strings (p. ej. org_suffix)
AttrValue = Optional[Union[AttrPrimitive, List[str]]]

#: Enumeración de tipos de entidad soportados por la Normalización
EntityType = Literal[
    "PERSON", "ORG", "LOC", "DATE", "MONEY", "EMAIL", "URL",
    "TITLE", "DEGREE", "PRODUCT", "ID", "OTHER"
]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class NormalizedEntity(BaseModel):
    """
    Entidad *normalizada* y *fusionada* a partir de una o más mentions.

    Campos:
    - id:     ID estable (hash determinista, ej. "ENT-abc123def4").
    - type:   Tipo de entidad (PERSON, ORG, DATE, ...).
    - name:   Nombre canónico legible (p. ej. "acme" para ORG o "Ana López" para PERSON).
    - value:  Valor principal cuando aplica (ISO para DATE, url/email para URL/EMAIL, etc.).
    - attrs:  Atributos tipados y normalizados (depende del tipo).
              Ejemplos:
                * DATE  -> {"value_iso":"2023-01-12","precision":"day","raw":"12 ene 2023"}
                * MONEY -> {"normalized_value":1250.0,"currency":"USD","currency_source":"code","raw":"US$ 1,250.00"}
                * ORG   -> {"org_core":"acme","org_suffix":["S.A. de C.V."],"org_key":"acme","raw":"Acme, S.A. de C.V."}
                * EMAIL -> {"email_norm":"user@dominio.com","user":"user","domain":"dominio.com"}
                * URL   -> {"url_norm":"http://acme.com","scheme":"http","host":"acme.com","path":""}
    - mentions: Lista de IDs de mentions que dan origen a esta entidad (trazabilidad 1–N).
    - conf:     Confianza agregada de la entidad (típicamente el máximo de las menciones).
    """
    id: str = Field(..., description="ID estable (ej. ENT-abc123def4)")
    type: EntityType = Field(..., description="Tipo de entidad normalizada")
    name: Optional[str] = Field(
        default=None,
        description="Nombre canónico para tipos nominales (PERSON/ORG/PRODUCT)."
    )
    value: Optional[str] = Field(
        default=None,
        description="Valor principal si aplica (ISO date, email, url, id, etc.)."
    )
    attrs: Dict[str, AttrValue] = Field(
        default_factory=dict,
        description="Atributos tipados normalizados; puede incluir listas (p. ej. org_suffix)."
    )
    mentions: List[str] = Field(
        default_factory=list,
        description="IDs de mentions fuente (trazabilidad 1–N)."
    )
    conf: float = Field(
        default=0.0,
        description="Confianza agregada de la entidad (ej. max de sus menciones)."
    )


class EntitiesMeta(BaseModel):
    """
    Metadatos de la salida por documento.

    - counters: métricas numéricas (int/float) útiles para auditoría:
        * input_mentions: total de mentions de entrada
        * kept_protos:    cuántas mentions superaron el umbral y generaron protos
        * entities:       total de entidades tras fusión/dedupe
    - config:   configuración efectiva utilizada (mezcla de tipos; p. ej. strings como 'auto' o 'MXN').
                Ejemplo:
                {
                  "date_locale": "auto",
                  "min_conf_keep": 0.66,
                  "merge_threshold": 0.92,
                  "canonicalize": true,
                  "default_currency": "MXN"
                }
    """
    counters: Dict[str, Union[int, float]] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class DocumentEntities(BaseModel):
    """
    Contenedor por documento para entidades normalizadas.

    - doc_id:   Identificador del documento (consistentemente el usado en otras etapas).
    - entities: Lista de `NormalizedEntity` para el documento.
    - meta:     `EntitiesMeta` con contadores y configuración aplicada.
    """
    doc_id: str = Field(..., description="ID del documento fuente (coincide con otras etapas)")
    entities: List[NormalizedEntity] = Field(
        default_factory=list,
        description="Entidades normalizadas resultantes para el documento."
    )
    meta: EntitiesMeta = Field(
        default_factory=EntitiesMeta,
        description="Metainformación: contadores y configuración efectiva."
    )


# ---------------------------------------------------------------------------
# Export explícito (opcional)
# ---------------------------------------------------------------------------

__all__ = [
    "AttrPrimitive",
    "AttrValue",
    "EntityType",
    "NormalizedEntity",
    "EntitiesMeta",
    "DocumentEntities",
]
