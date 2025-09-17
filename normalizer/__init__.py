"""Etapa de Normalización (Mentions -> Entities).

Expone:
- NormalizeConfig: configuración tipada de la etapa
- Normalizer: orquestador principal
- DocumentEntities / NormalizedEntity: contratos de salida
"""

from .config import NormalizeConfig
from .normalizer import Normalizer
from .schemas import DocumentEntities, NormalizedEntity

__all__ = ["NormalizeConfig", "Normalizer", "DocumentEntities", "NormalizedEntity","metrics"]
