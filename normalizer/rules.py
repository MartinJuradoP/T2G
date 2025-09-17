"""Recursos inmutables: sufijos de organizaciones, mapas de títulos/grados, meses ES/EN."""
from __future__ import annotations
from importlib.resources import files
import json

def load_json_resource(path: str) -> list | dict:
    """Carga un JSON embebido en `normalizer/resources`."""
    return json.loads(files("normalizer.resources").joinpath(path).read_text(encoding="utf-8"))

# Sufijos corporativos frecuentes (MX/EN). Editables en /resources
ORG_SUFFIXES_MX: list[str] = load_json_resource("org_suffixes.mx.json")
ORG_SUFFIXES_EN: list[str] = load_json_resource("org_suffixes.en.json")

# Mapas planos para normalizar títulos/cargos y grados académicos
TITLES_MAP: dict[str, str]   = load_json_resource("titles.map.json")
DEGREES_MAP: dict[str, str]  = load_json_resource("degrees.map.json")

# Abreviaturas de mes -> número (ambos idiomas)
MONTHS_ES = {
    "ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
    "jul":"07","ago":"08","sep":"09","oct":"10","nov":"11","dic":"12"
}
MONTHS_EN = {
    "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
    "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"
}
