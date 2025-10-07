# -*- coding: utf-8 -*-
"""
contextizer.hybrid.__init__ — Punto de entrada del modo híbrido del Contextizer T2G
==============================================================================

Resumen
-------
Este submódulo habilita un **Contextizer Híbrido** que complementa a BERTopic en
textos cortos, ruidosos o multitema (p. ej., reseñas, posts y noticias
financieras). Su objetivo es **aumentar la robustez** y **conservar la
compatibilidad** con los contratos existentes del pipeline T2G.

Garantías de diseño
-------------------
- **Compatibilidad total** con `TopicItem`, `TopicsDocMeta`, `ChunkTopic` y
  `TopicsChunksMeta` (Pydantic con `extra="ignore"`).
- **Integración no intrusiva**: `contextizer/contextizer.py` decide si activar
  el modo híbrido usando heurísticas ligeras; cuando no aplica, BERTopic opera
  de forma normal.
- **Additive**: el híbrido puede añadir campos opcionales como
  `mmr_keywords` y `meta.prob_dist` sin romper consumidores existentes.
- **Trazabilidad**: cada decisión deja razón (`reason = "doc-hybrid"` o
  `"chunk-hybrid"`) y metadatos explicativos.

Arquitectura
------------
Este paquete expone dos funciones públicas para ejecución y dos funciones para
la decisión adaptativa:

- `should_use_hybrid_doc(texts, cfg, emb=None)`  → bool
- `should_use_hybrid_chunks(texts, cfg, emb=None)` → bool
- `run_hybrid_contextizer_doc(ir_path, cfg, outdir)` → None (inyecta en IR)
- `run_hybrid_contextizer_chunks(chunk_path, cfg)` → None (inyecta en place)

Internamente, el modo híbrido utiliza clustering por densidad (cosine) y
fusión de *keywords* (TF‑IDF + KeyBERT opcional) con filtrado MMR para reducir
redundancia léxica, respetando la estructura de salida actual.

Uso (ejemplo de alto nivel)
---------------------------
>>> from contextizer.hybrid import should_use_hybrid_doc, run_hybrid_contextizer_doc
>>> # En tu router (contextizer.py), tras extraer `texts` y `cfg`:
>>> if should_use_hybrid_doc(texts, cfg):
...     run_hybrid_contextizer_doc(ir_path, cfg, outdir)
... else:
...     # continuar con el flujo BERTopic clásico
...     ...

Notas
-----
- Este archivo implementa **carga perezosa** (lazy import) de los submódulos
  para evitar errores de importación mientras el paquete se construye
  incrementalmente.
- El contenido detallado de cada submódulo está documentado en su respectivo
  encabezado (ver: analyzers.py, density_clustering.py, keyword_fusion.py,
  mmr.py, hybrid_contextizer.py, metrics_ext.py).

Referencias
-----------
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with Class-based TF-IDF*.
- Carbonell, J., & Goldstein, J. (1998). *The use of MMR for diversity-based information retrieval*.
- Ganesan, A. (2020). *KeyBERT: Minimal keyword extraction with BERT*.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Decisión adaptativa
    "should_use_hybrid_doc",
    "should_use_hybrid_chunks",
    # Ejecución híbrida
    "run_hybrid_contextizer_doc",
    "run_hybrid_contextizer_chunks",
]

__version__ = "0.1.0"
__docformat__ = "restructuredtext"


# ---------------------------------------------------------------------------
# Carga perezosa (evita dependencias rotas durante la construcción incremental)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover — trivial loader
    """Resolución perezosa de símbolos públicos.

    Permite `from contextizer.hybrid import ...` incluso si aún no se han
    importado físicamente los submódulos, siempre que sus archivos existan en
    tiempo de ejecución.
    """
    if name in {"should_use_hybrid_doc", "should_use_hybrid_chunks"}:
        mod = import_module(".analyzers", __name__)
        return getattr(mod, name)
    if name in {"run_hybrid_contextizer_doc", "run_hybrid_contextizer_chunks"}:
        mod = import_module(".hybrid_contextizer", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover — trivial
    return sorted(list(globals().keys()) + __all__)
