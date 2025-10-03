# -*- coding: utf-8 -*-
"""
selector.py — Adaptive Schema Selector

Lógica:
- Calcula puntuaciones de dominios con dos señales principales:
  (1) overlap de keywords (documento/chunk ↔ aliases del dominio)
  (2) similitud de embeddings (centroide ↔ label_vecs del dominio)
- Priors opcionales (si hubiera): ponderados vía gamma_prior.
- Devuelve selección a nivel documento + por chunk, con evidencias auditables.

Dependencias esperadas (ya implementadas en tu proyecto):
- schema_selector.utils: cosine_sim, get_doc_keywords, get_chunk_keywords,
                         get_doc_centroid, get_chunk_centroid
- schema_selector.registry: REGISTRY con dominios y entity_types
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .schemas import (
    SchemaSelection, DocSchemaSelection, ChunkSchemaSelection,
    SelectorConfig, DomainScore, EntityTypeScore, Evidence
)
from .utils import (
    cosine_sim, normalize_keyword, normalize_tokens,
    get_doc_keywords, get_chunk_keywords,
    get_doc_centroid, get_chunk_centroid
)
from .registry import REGISTRY


# ---------------------------------------------------------------------------
# Helpers de scoring
# ---------------------------------------------------------------------------

def _keyword_overlap_score(doc_kws: List[str], aliases: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Score por keywords: fracción de keywords del doc presentes en los alias del dominio.
    """
    if not doc_kws:
        return 0.0, {"overlap": 0, "kw_count": 0}
    aliases_norm = {normalize_keyword(a) for a in aliases}
    overlap = sum(1 for kw in doc_kws if kw in aliases_norm)
    score = overlap / max(1, len(doc_kws))
    return float(score), {"overlap": overlap, "kw_count": len(doc_kws)}


def _embedding_score(vec: List[float], label_vecs: Dict[str, List[float]]) -> Tuple[float, Dict[str, Any]]:
    """
    Score por embeddings: máximo coseno entre el vector del doc/chunk
    y cualquiera de los vectores de etiqueta del dominio.
    """
    if not vec or not label_vecs:
        return 0.0, {"emb_score": 0.0, "labels": 0}
    best = 0.0
    best_label = None
    for label, lvec in label_vecs.items():
        cs = cosine_sim(vec, lvec)
        if cs > best:
            best, best_label = cs, label
    return float(best), {"emb_score": best, "best_label": best_label, "labels": len(label_vecs)}


def _score_domain(dom, doc_kw: List[str], doc_vec: List[float], prior: float, cfg: SelectorConfig) -> Tuple[float, List[Evidence]]:
    """
    Score de dominio combinando keywords, embeddings y prior.
    """
    kw_score, kw_ev = _keyword_overlap_score(doc_kw, dom.aliases)
    emb_score, emb_ev = _embedding_score(doc_vec, getattr(dom, "label_vecs", {}))

    score = cfg.alpha_kw * kw_score + cfg.beta_emb * emb_score + cfg.gamma_prior * float(prior or 0.0)

    evidence = [
        Evidence(kind="keyword", detail={"kw_score": kw_score, **kw_ev}),
        Evidence(kind="embedding", detail={"emb_score": emb_score, **emb_ev}),
        Evidence(kind="stat", detail={"prior": float(prior or 0.0), "alpha_kw": cfg.alpha_kw, "beta_emb": cfg.beta_emb, "gamma_prior": cfg.gamma_prior})
    ]
    return float(score), evidence


def _score_entity_types(text_tokens: List[str], entity_types, cfg: SelectorConfig) -> List[EntityTypeScore]:
    """
    Score por tipo de entidad: fracción simple de aliases presentes en el texto tokenizado.
    (Ligero pero útil para priorizar etiquetas en NER/RE posteriores).
    """
    if not entity_types:
        return []

    token_set = set(text_tokens or [])
    scored: List[EntityTypeScore] = []
    for et in entity_types:
        aliases_norm = {normalize_keyword(a) for a in getattr(et, "aliases", [])}
        overlap = sum(1 for a in aliases_norm if a in token_set)
        # Normalizamos respecto al # de aliases para evitar sesgos
        base = overlap / max(1, len(aliases_norm)) if aliases_norm else 0.0
        ev = [
            Evidence(kind="keyword", detail={"et": et.name, "overlap": overlap, "aliases": len(aliases_norm)}),
        ]
        scored.append(EntityTypeScore(type_name=et.name, score=float(base), evidence=ev))
    # Devuelve en orden descendente (opcional)
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Selección principal
# ---------------------------------------------------------------------------

def select_schemas(doc: Dict[str, Any],
                   registry=REGISTRY,
                   config: SelectorConfig = SelectorConfig(),
                   priors: Dict[str, float] | None = None) -> SchemaSelection:
    """
    Selección adaptativa de esquemas (documento + chunks).
    - registry: OntologyRegistry con dominios y entidades.
    - config: SelectorConfig con pesos y umbrales.
    - priors: dict opcional {domain: prior in [0,1]} para sesgos organizacionales.
    """
    priors = priors or {}

    # -----------------------------
    # 1) Señales del documento
    # -----------------------------
    doc_id = doc.get("doc_id", "UNKNOWN")
    doc_keywords = get_doc_keywords(doc)                  # lista normalizada (puede estar vacía)
    doc_centroid = get_doc_centroid(doc)                  # vector normalizado o []

    # Dominio → score
    doc_domain_scores: List[DomainScore] = []
    for dom in registry.domains:
        prior = priors.get(dom.domain, 0.0)
        s, ev = _score_domain(dom, doc_keywords, doc_centroid, prior, config)

        # Opcional: score por entity types usando tokens del documento (text global si lo tienes)
        # Aquí usamos keywords como proxy de tokens globales (simple y barato)
        text_tokens = doc_keywords  # si tienes texto global, reemplázalo por normalize_tokens(global_text)
        et_scores = _score_entity_types(text_tokens, dom.entity_types, config)

        doc_domain_scores.append(DomainScore(
            domain=dom.domain,
            score=float(s),
            entity_type_scores=et_scores,
            evidence=ev
        ))

    # Ordenar dominios
    doc_domain_scores.sort(key=lambda d: d.score, reverse=True)

    # Top dominios (siempre incluye los de always_include)
    top_domains: List[str] = [d.domain for d in doc_domain_scores[:config.topk]]
    for must in (config.always_include or []):
        if must not in top_domains:
            top_domains.append(must)

    # Ambigüedad doc-level
    ambiguous_doc = False
    if len(doc_domain_scores) > 1:
        ambiguous_doc = abs(doc_domain_scores[0].score - doc_domain_scores[1].score) < config.ambiguity_threshold

    doc_sel = DocSchemaSelection(
        doc_id=doc_id,
        domain_scores=doc_domain_scores,
        top_domains=top_domains[:config.max_domains],
        ambiguous=ambiguous_doc
    )

    # -----------------------------
    # 2) Señales por chunk
    # -----------------------------
    chunk_selections: List[ChunkSchemaSelection] = []
    for ch in doc.get("chunks", []):
        ch_id = ch.get("chunk_id", "UNK")
        ch_kws = get_chunk_keywords(ch)
        ch_vec = get_chunk_centroid(ch)

        ch_scores: List[DomainScore] = []
        # prioridad: evaluar solo en top_domains para ahorrar costes (y siempre generic)
        eval_domains = {d for d in top_domains} | set(config.always_include or [])
        eval_domains = [d for d in registry.domains if d.domain in eval_domains] or registry.domains

        # tokens para score de entity types (rápido)
        text_tokens = ch_kws or normalize_tokens(ch.get("text", "") or "")

        for dom in eval_domains:
            prior = priors.get(dom.domain, 0.0)
            s, ev = _score_domain(dom, ch_kws, ch_vec, prior, config)
            et_scores = _score_entity_types(text_tokens, dom.entity_types, config)
            ch_scores.append(DomainScore(domain=dom.domain, score=float(s), entity_type_scores=et_scores, evidence=ev))

        ch_scores.sort(key=lambda d: d.score, reverse=True)
        top_dom = ch_scores[0].domain if ch_scores else None
        amb = False
        if len(ch_scores) > 1:
            amb = abs(ch_scores[0].score - ch_scores[1].score) < config.ambiguity_threshold

        chunk_selections.append(ChunkSchemaSelection(
            chunk_id=ch_id,
            domain_scores=ch_scores,
            top_domain=top_dom,
            ambiguous=amb
        ))

    # -----------------------------
    # 3) Ensamble final
    # -----------------------------
    meta = {
        "alpha_kw": config.alpha_kw,
        "beta_emb": config.beta_emb,
        "gamma_prior": config.gamma_prior,
        "ambiguity_threshold": config.ambiguity_threshold,
        "topk": config.topk,
        "always_include": config.always_include,
        "version": "selector.v1"
    }

    return SchemaSelection(doc=doc_sel, chunks=chunk_selections, meta=meta)
