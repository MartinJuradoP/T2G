# -*- coding: utf-8 -*-
"""
contextizer.hybrid.analyzers — Heurísticas adaptativas del Contextizer Híbrido T2G
=================================================================================

Resumen
-------
El propósito es reducir
fallos en documentos o fragmentos (chunks) cortos, ruidosos o con alta
diversidad temática, sin incurrir en costos innecesarios de procesamiento.

Principios de diseño
--------------------
- **Ligereza:** cálculos rápidos y deterministas sin dependencias pesadas.
- **Explicabilidad:** cada decisión de activación se basa en métricas
  interpretables (longitud promedio, diversidad léxica, dispersión semántica).
- **Escalabilidad:** las funciones pueden ejecutarse en lotes o por documento.
- **Compatibilidad:** su salida booleana (`True` → híbrido, `False` → BERTopic)
  permite integración directa en el router del Contextizer.

Funciones públicas
------------------
- `should_use_hybrid_doc(texts, cfg, emb=None)` → bool
- `should_use_hybrid_chunks(texts, cfg, emb=None)` → bool

Métricas internas utilizadas
----------------------------
1. **Longitud media de tokens (`avg_len`)** — mide densidad informativa.
2. **Diversidad léxica (`type_token_ratio`)** — evalúa dispersión de vocabulario.
3. **Varianza semántica (`semantic_variance`)** — estima heterogeneidad temática.

Criterios (activación del modo híbrido)
--------------------------------------
El modo híbrido se activa si al menos dos de las siguientes condiciones son
verdaderas:
- Pocos fragmentos (n < 5)
- Textos muy cortos (`avg_len` < 45)
- Alta diversidad léxica (TTR > 0.5)
- Alta dispersión semántica (varianza > 0.25)

Ejemplo de uso
--------------
>>> from contextizer.hybrid.analyzers import should_use_hybrid_doc
>>> texts = ["Diabetes Tipo 2: tratamiento", "Complicaciones clínicas"]
>>> should_use_hybrid_doc(texts, cfg={})
True

Referencias
-----------
- Grootendorst (2022) — BERTopic.
- Biber (1995) — Lexical diversity in text analysis.
- Mikolov et al. (2013) — Word2Vec: vector space semantics.
"""

from __future__ import annotations
from typing import List
import numpy as np
import re

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades internas (ligeras y sin dependencias externas)
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}")

def _rough_tokens(txt: str) -> list[str]:
    """Tokenizador ligero para estimar diversidad léxica sin spaCy ni NLTK."""
    return _TOKEN_RE.findall((txt or "").lower())

def _type_token_ratio(tokens: list[str]) -> float:
    """Calcula la relación tipo/token (TTR)."""
    return 0.0 if not tokens else len(set(tokens)) / float(len(tokens))

def _avg_len_tokens(texts: List[str]) -> float:
    """Promedio de tokens por fragmento de texto."""
    toks = sum(len(_rough_tokens(t)) for t in texts)
    return toks / max(1, len(texts))

def _semantic_variance(emb: np.ndarray | None) -> float:
    """Varianza promedio de los embeddings como indicador de dispersión semántica."""
    if emb is None or len(emb) < 2:
        return 0.0
    mu = emb.mean(axis=0, keepdims=True)
    dif = emb - mu
    return float((dif * dif).sum(axis=1).mean())

# ──────────────────────────────────────────────────────────────────────────────
# Reglas de decisión adaptativa
# ──────────────────────────────────────────────────────────────────────────────

def should_use_hybrid_doc(texts: List[str], cfg, emb: np.ndarray | None = None) -> bool:
    """Evalúa si un documento debe procesarse con el Contextizer Híbrido.

    Parámetros
    ----------
    texts : list[str]
        Lista de bloques o párrafos del documento.
    cfg : dict
        Configuración del contexto (no modificada aquí, usada para logs externos).
    emb : np.ndarray | None
        Embeddings opcionales (n_samples, dim) para medir dispersión semántica.

    Retorna
    -------
    bool
        `True` si se recomienda el modo híbrido, `False` si BERTopic estándar es suficiente.
    """
    n = len(texts)
    if n == 0:
        return False
    if n < 5:
        return True

    avg_len = _avg_len_tokens(texts)
    ttrs = [_type_token_ratio(_rough_tokens(t)) for t in texts[: min(32, n)]]
    ttr = float(np.mean(ttrs)) if ttrs else 0.0
    sem_var = _semantic_variance(emb)

    votes = [avg_len < 45, ttr > 0.5, sem_var > 0.25]
    return sum(votes) >= 2

def should_use_hybrid_chunks(texts: List[str], cfg, emb: np.ndarray | None = None) -> bool:
    """Evalúa si un conjunto de *chunks* requiere el modo híbrido.

    Lógica principal:
    - Si no hay texto → no aplica.
    - Si hay menos de 5 muestras → híbrido por defecto.
    - Si la longitud media < 60 tokens → híbrido.
    - Si los embeddings muestran varianza alta (> 0.25) → híbrido.
    """
    n = len(texts)
    if n == 0:
        return False
    if n < 5:
        return True

    avg_len = _avg_len_tokens(texts)
    if avg_len < 60:
        return True

    return _semantic_variance(emb) > 0.25 if emb is not None else False
