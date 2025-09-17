"""Configuración tipada para Normalización."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class NormalizeConfig:
    date_locale: str = "auto"        # auto|es|en
    min_conf_keep: float = 0.66
    merge_threshold: float = 0.92    # similitud para unir PERSON/ORG por nombre
    canonicalize: bool = True
    default_currency: str = "MXN"    # $ ambiguo → MXN por contexto
