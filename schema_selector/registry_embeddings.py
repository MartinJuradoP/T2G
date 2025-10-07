# -*- coding: utf-8 -*-
from __future__ import annotations
"""
registry_embeddings.py — Generación, validación y cacheo de embeddings de dominio (T2G)

Propósito
---------
Este módulo construye y gestiona los embeddings representativos de cada dominio
registrado en el `OntologyRegistry` (ver `registry.py`). Su misión es mantener
consistencia semántica entre:
  - Los embeddings generados por el Contextizer (documento/chunks)
  - Los prototipos de dominio usados por el Adaptive Schema Selector

Características
---------------
Coherencia semántica: usa el mismo modelo del Contextizer (`all-MiniLM-L6-v2`)
Cache persistente: guarda centroides de dominios en JSON (`registry_vectors.json`)
Resiliencia: si el cache no existe, lo reconstruye automáticamente
Auditable: estructura JSON legible y versionable
Modular: puede regenerarse sin alterar `registry.py`

Ubicación
---------
project_T2G/schema_selector/registry_embeddings.py

Dependencias
------------
- sentence-transformers
- numpy
- json
- registry.py (OntologyRegistry base)
"""

import os
import sys
import json
import numpy as np
import warnings
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Ajuste de rutas y warnings
# ---------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    # Permite ejecutar directamente: python schema_selector/registry_embeddings.py
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(package_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "schema_selector"

warnings.filterwarnings("ignore", message="found in sys.modules")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Importación del registry
# ---------------------------------------------------------------------------
from schema_selector.registry import REGISTRY

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILENAME = "registry_vectors.json"
CACHE_PATH = os.path.join(os.path.dirname(__file__), CACHE_FILENAME)


def _compute_domain_centroid(model, aliases: list[str]) -> list[float]:
    if not aliases:
        return []
    vectors = [model.encode(alias) for alias in aliases]
    centroid = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def build_label_vecs(registry=REGISTRY, rebuild: bool = False) -> Any:
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(CACHE_PATH) and not rebuild:
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded, skipped = 0, 0
            for d in registry.domains:
                if d.domain in data:
                    d.label_vecs = data[d.domain]
                    loaded += 1
                else:
                    skipped += 1

            print(f"Embeddings cargados desde cache ({loaded} dominios, {skipped} faltantes).")
            print(f"Archivo: {CACHE_PATH}")
            return registry

        except Exception as e:
            print(f"Error al cargar cache: {e}. Se regenerarán todos los embeddings.")

    print("Recalculando embeddings de dominios...")
    vectors: Dict[str, Dict[str, list[float]]] = {}
    for domain in registry.domains:
        centroid = _compute_domain_centroid(model, domain.aliases)
        if centroid:
            domain.label_vecs = {"centroid": centroid}
            vectors[domain.domain] = {"centroid": centroid}
        else:
            print(f"Dominio '{domain.domain}' sin alias válidos; se omite embedding.")

    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(vectors, f, indent=2, ensure_ascii=False)
        print(f"Embeddings de dominios guardados en {CACHE_PATH}")
    except Exception as e:
        print(f"No se pudo escribir el cache: {e}")

    return registry


def validate_registry_embeddings(registry=REGISTRY) -> Dict[str, Any]:
    report = {"total": len(registry.domains), "valid": 0, "invalid": [], "dim": None}
    ref_dim = None
    for d in registry.domains:
        vec = d.__dict__.get("label_vecs", {}).get("centroid")
        if not vec:
            report["invalid"].append(d.domain)
            continue
        dim = len(vec)
        if ref_dim is None:
            ref_dim = dim
        if dim != ref_dim:
            report["invalid"].append(d.domain)
            continue
        report["valid"] += 1
    report["dim"] = ref_dim
    return report


if __name__ == "__main__":
    registry = build_label_vecs(rebuild=True)
    summary = validate_registry_embeddings(registry)
    print(f"Validación: {summary['valid']}/{summary['total']} dominios con embeddings ({summary['dim']} dim).")
