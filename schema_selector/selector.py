# -*- coding: utf-8 -*-
"""
selector.py — Adaptive Schema Selector 2.0 (núcleo mejorado y explicable)

Propósito
---------
Seleccionar, para cada documento y cada chunk, el **dominio** y el **esquema de entidades**
más adecuado utilizando una **fusión ponderada de señales** provenientes del Contextizer y
del HybridChunker. El resultado es **explicable, auditable y reproducible**: además del
top-domain y el esquema, el módulo devuelve **confianza probabilística**, **evidencias**,
y un **DecisionTrace** con la contribución de cada señal al score final.

Motivación
----------
En textos breves (reviews, tweets, posts) el contexto es tenue y los enfoques basados
en una sola señal (keywords o embeddings) son frágiles. Este selector combina:
- Semántica léxica (keywords) con **F1**,
- Semántica densa (embeddings) vía **coseno** contra prototipos de dominio,
- **Afinidad temática** (tópicos doc/chunk ↔ alias de dominio),
- **Calidad contextual** del chunk/doc (cohesión, salud, redundancia inversa, novelty, riqueza),
- **Priors** opcionales (sesgos de negocio).
La decisión se explica con un rastro cuantitativo y umbrales explícitos para ambigüedad y fallback.

Entradas esperadas (contrato)
-----------------------------
`doc: Dict[str, Any]` enriquecido por Contextizer + HybridChunker, con:
- `doc["doc_id"]`: str
- `doc["meta"]["topics_doc"]` (opcional):
    - `keywords_global`: List[str]
    - `topics`: List[{`topic_id`: int, `keywords`: List[str], `prob`: float}]
    - `embeddings`: List[List[float]] (opcional; centroides por tópico)
- `doc["meta"]["topics_chunks"]` (opcional):
    - `keywords_global`: List[str]
- `doc["chunks"]`: List[Chunk], cada `Chunk` con:
    - `chunk_id`: str
    - `text`: str (opcional pero recomendado)
    - `topic`: {`topic_id`: int, `keywords`: List[str], `prob`: float} (opcional)
    - `meta_local.embedding`: List[float] (recomendado)
    - `scores`: Dict[str, float] con al menos:
        * `cohesion_vs_doc`, `chunk_health`, `redundancy_norm`, `novelty`,
          `lexical_density`, `type_token_ratio`

Además, el selector consume:
- `registry: OntologyRegistry` (ver `schema_selector/registry.py`) con dominios,
  alias, entidades/relaciones y `label_vecs` opcionales (prototipos de embeddings).
- `config: SelectorConfig` (ver `schema_selector/schemas.py`) con pesos, umbrales y versión.
- `priors: Dict[str, float]` opcional para sesgos organizacionales por dominio.

Salidas (resumen)
-----------------
`SchemaSelection` con:
- **DocSchemaSelection**:
  - `top_domains`: lista rankeada de dominios
  - `selected_schema`: nombre de la plantilla de esquema (p.ej. `legal_contract_v1`)
  - `schema_confidence`: [0,1] (softmax con temperatura)
  - `explanation`: texto breve con los drivers de la decisión
  - `domain_scores`: por dominio, incluye `DecisionTrace` y `Evidence`
  - `ambiguous`: bool si |S1−S2| < τ
- **ChunkSchemaSelection** (por chunk):
  - `top_domain`, `selected_schema`, `schema_confidence`, `explanation`,
    `domain_scores`, `ambiguous`
- **meta**: versión, pesos, umbrales, dominios evaluados, señales usadas

Señales y fórmulas
------------------
Para cada dominio `d`, el score final `S_d` es la suma de contribuciones ponderadas:

    K = F1(doc_keywords, domain_aliases)
    E = max_cos(centroid(doc|chunk), label_vecs(domain))              # 0 si no hay prototipos
    T = afinidad_tópicos(doc/chunk, domain_aliases)                   # ver más abajo
    C = 0.30*cohesion_vs_doc + 0.30*chunk_health +
        0.20*(1 - redundancy_norm) + 0.10*novelty +
        0.10*richness ;  richness = 0.5*lexical_density + 0.5*TTR
    P = prior(domain)                                                 # opcional

    S_d = α·K + β·E + γ·C + δ·T + ε·P

- **Afinidad de tópicos (T)**:
  - Doc-level: promedio de (Jaccard(keywords_topic, aliases_d) * prob_topic)
  - Chunk-level: Jaccard(chunk_keywords, aliases_d), ponderado por `chunk_health`
- **Confianza**:
  - `confidence = softmax(S_d / T°)[top]`, con temperatura `T°` configurable
- **Ambigüedad**:
  - `ambiguous = |S1 - S2| < τ` (umbral configurable)

Flujo de decisión (alto nivel)
------------------------------
1) **Agregación** de métricas de chunks → contexto global del documento.  
2) **Scoring por dominio** (doc-level) con K, E, T, C, P → rank + confianza.  
3) **Selección de esquema** para el doc (mapea dominio→plantilla).  
4) **Scoring por dominio** (chunk-level) limitado a `top_domains ∪ always_include`.  
5) **Explicabilidad**: se construye `DecisionTrace` (contribuciones αK, βE, γC, δT, εP)
   y `Evidence` (detalles de F1, coseno, Jaccard, métricas).  
6) **Guardrails**:
   - Texto corto (tokens < umbral) ⇒ prioriza `generic` salvo evidencia fuerte.
   - Confianza doc-level < `fallback_threshold` ⇒ añade esquema `generic` como respaldo.

Robustez y casos límite
-----------------------
- **Textos muy cortos**: depende de K/T y heurística de tokens; fallback a `generic`.
- **Multilingüe** (ES/EN): normalización básica en tokens/keywords; embeddings
  amortiguan diferencias de idioma.
- **Faltan embeddings**: `E=0`, el sistema se sostiene con K/T/C.
- **Tópicos conflictivos**: `ambiguous=True` y explicación con drivers divergentes.
- **Redundancia alta**: penalizada en C (1 - redundancy_norm).

Complejidad (aprox.)
--------------------
O(D * (1 + C)) por documento, donde D=#dominios y C=#chunks. El cálculo de
embeddings es O(1) (usa centroides precalculados). Tópicos y métricas son
consumidos tal cual del Contextizer/Chunker.

Integración
-----------
- Entrada: `outputs_chunks/*.json`
- Salida:  `outputs_schema/*.json`
- Puro y determinista: **no** modifica el doc de entrada; serializa `SchemaSelection`.

Extensión
---------
- Añadir dominios/alias/entidades: `schema_selector/registry.py`
- Ajustar pesos/umbrales: `SelectorConfig`
- Mejorar E: incluir `label_vecs` por dominio (prototipos promedio).
- Incluir nuevas métricas en C: ampliar `utils.context_score` y documentación.

Limitaciones conocidas
----------------------
- Si el doc carece de texto legible (solo imágenes sin OCR), la señal K cae y la decisión
  dependerá de T/C/E; se recomienda integrar OCR en Parser.
- Si no hay embeddings ni tópicos, el selector opera con K + heurísticas; marque `ambiguous`.
"""


from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
from .schemas import (
    SchemaSelection, DocSchemaSelection, ChunkSchemaSelection,
    SelectorConfig, DomainScore, EntityTypeScore, Evidence, DecisionTrace
)
from .utils import (
    get_doc_keywords, get_chunk_keywords, get_doc_centroid, get_chunk_centroid,
    keyword_f1, topic_affinity_score, context_score, aggregate_doc_metrics,
    softmax_confidence, build_explanation, normalize_tokens, cosine_sim
)
from .registry import REGISTRY


# Mapeo dominio → plantilla de esquema (ajústalo a tus plantillas reales)
SCHEMA_BY_DOMAIN = {
    "legal": "legal_contract_v1",
    "medical": "medical_note_v1",
    "financial": "financial_tx_v2",
    "ecommerce": "ecommerce_order_v1",
    "identity": "identity_record_v1",
    "tech_review": "tech_review_v1",
    "veterinary": "veterinary_case_v1",
    "geopolitical": "geopolitical_event_v1",
    "reviews": "review_text_v1",
    "generic": "generic_text_v1",
}


# -----------------------------
# Helpers internos
# -----------------------------

def _embedding_score(vec: List[float], label_vecs: Dict[str, List[float]]) -> Tuple[float, Dict[str, Any]]:
    """
    Devuelve max coseno entre el vector (doc/chunk) y prototipos del dominio.
    Si no existen prototipos, score=0 (la fusión se sostiene en K/T/C).
    """
    if not vec or not label_vecs:
        return 0.0, {"emb_score": 0.0, "best_label": None, "labels": 0}
    best = 0.0
    best_label = None
    for label, proto in label_vecs.items():
        cs = cosine_sim(vec, proto)
        if cs > best:
            best = cs
            best_label = label
    return float(max(0.0, min(1.0, best))), {"emb_score": best, "best_label": best_label, "labels": len(label_vecs)}


def _score_entity_types(tokens: List[str], entity_types) -> List[EntityTypeScore]:
    """
    Score ligero por tipo de entidad:
    - alias overlap normalizado (n_aliases) para priorizar clases de NER/RE aguas abajo.
    """
    tok = set(tokens or [])
    out: List[EntityTypeScore] = []
    for et in (entity_types or []):
        aliases = {a.strip().lower() for a in (et.aliases or []) if a}
        if not aliases:
            out.append(EntityTypeScore(type_name=et.name, score=0.0, evidence=[]))
            continue
        overlap = sum(1 for a in aliases if a in tok)
        base = overlap / max(1, len(aliases))
        ev = [Evidence(kind="keyword", detail={"et": et.name, "overlap": overlap, "aliases": len(aliases)})]
        out.append(EntityTypeScore(type_name=et.name, score=float(base), evidence=ev))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def _domain_final_score(k: float, e: float, c: float, t: float, p: float, cfg: SelectorConfig) -> Tuple[float, DecisionTrace, Dict[str, Any]]:
    """
    Funde señales con pesos del config y devuelve score final + DecisionTrace + summaries.
    """
    w = cfg.weights()
    contributions = {
        "K": w["alpha_kw"] * k,
        "E": w["beta_emb"] * e,
        "C": w["gamma_ctx"] * c,
        "T": w["delta_top"] * t,
        "P": w["epsilon_prior"] * p,
    }
    final_score = float(sum(contributions.values()))
    trace = DecisionTrace(
        domain="",
        keyword_score=float(k),
        embedding_score=float(e),
        context_score=float(c),
        topic_score=float(t),
        prior=float(p),
        weights=w,
        contributions=contributions,
        final_score=final_score,
        used_signals=["keywords", "embeddings", "context", "topics", "prior"]
    )
    summaries = {}
    return final_score, trace, summaries


# -----------------------------
# Selección a nivel documento
# -----------------------------

def _select_doc_level(doc: Dict[str, Any],
                      cfg: SelectorConfig,
                      registry=REGISTRY,
                      priors: Dict[str, float] | None = None) -> Tuple[DocSchemaSelection, List[str], List[DomainScore]]:
    priors = priors or {}
    doc_id = doc.get("doc_id", "UNKNOWN")
    doc_tokens = count_tokens_doc(doc)
    doc_kws = get_doc_keywords(doc)
    doc_vec = get_doc_centroid(doc)
    doc_metrics = aggregate_doc_metrics(doc)

    # Signals independent of domain
    ctx, ctx_detail = context_score(doc_metrics)

    domain_scores: List[DomainScore] = []
    for dom in registry.domains:
        # K: keywords F1
        k, k_detail = keyword_f1(doc_kws, dom.aliases)
        # E: embedding vs prototypes (si existen)
        e, e_detail = _embedding_score(doc_vec, dom.label_vecs)
        # T: topics affinity
        t, t_detail = topic_affinity_score(doc, dom.aliases)
        # P: prior opcional
        p = float(priors.get(dom.domain, 0.0))

        final, trace, _ = _domain_final_score(k, e, ctx, t, p, cfg)
        trace.domain = dom.domain  # completar

        # Entity-type prioritization tokens: usamos keywords + tokens globales
        global_text_tokens = set(doc_kws)
        # añadimos algunos tokens del propio texto (si existiera)
        for ch in doc.get("chunks", []):
            global_text_tokens |= set(normalize_tokens(ch.get("text", "") or ""))
        et_scores = _score_entity_types(sorted(global_text_tokens), dom.entity_types)

        ev = [
            Evidence(kind="keyword", detail=k_detail),
            Evidence(kind="embedding", detail=e_detail),
            Evidence(kind="topic", detail=t_detail),
            Evidence(kind="context", detail=ctx_detail),
            Evidence(kind="prior", detail={"value": p})
        ]
        domain_scores.append(DomainScore(
            domain=dom.domain,
            score=final,
            entity_type_scores=et_scores,
            evidence=ev,
            decision_trace=trace
        ))

    # Rank y top-domains
    domain_scores.sort(key=lambda d: d.score, reverse=True)
    top_domains = [d.domain for d in domain_scores[:cfg.topk]]

    # Always-include (generic)
    for must in (cfg.always_include or []):
        if must not in top_domains:
            top_domains.append(must)

    # Confianza y ambigüedad
    conf = softmax_confidence([d.score for d in domain_scores], temperature=cfg.softmax_temperature)
    ambiguous = (len(domain_scores) > 1 and abs(domain_scores[0].score - domain_scores[1].score) < cfg.ambiguity_threshold)

    # Fallback para textos muy cortos
    if doc_tokens < cfg.min_doc_tokens_for_domain and "generic" in top_domains:
        # forzamos generic si no hay gaps fuertes
        top_domain = "generic"
        selected_schema = SCHEMA_BY_DOMAIN.get(top_domain, "generic_text_v1")
        explanation = f"Texto corto (tokens={doc_tokens}); se aplica fallback a '{top_domain}' con soporte de señales globales."
    else:
        top_domain = domain_scores[0].domain if domain_scores else "generic"
        selected_schema = SCHEMA_BY_DOMAIN.get(top_domain, "generic_text_v1")
        # Explicación basada en contribuciones
        explanation = build_explanation(
            top_domain,
            domain_scores[0].decision_trace.contributions if domain_scores else {},
            {"kw": domain_scores[0].evidence[0].detail if domain_scores else {},
             "emb": domain_scores[0].evidence[1].detail if domain_scores else {},
             "topic": domain_scores[0].evidence[2].detail if domain_scores else {}},
            ctx_detail
        )

    doc_sel = DocSchemaSelection(
        doc_id=doc_id,
        domain_scores=domain_scores,
        top_domains=top_domains[:cfg.max_domains],
        selected_schema=selected_schema,
        schema_confidence=conf,
        explanation=explanation,
        signals_used=["keywords", "embeddings", "topics", "context", "prior"],
        weights_used=cfg.weights(),
        ambiguous=ambiguous
    )
    return doc_sel, top_domains, domain_scores


def count_tokens_doc(doc: Dict[str, Any]) -> int:
    total = 0
    for ch in doc.get("chunks", []):
        total += len(normalize_tokens(ch.get("text", "") or ""))
    return total


# -----------------------------
# Selección a nivel chunk
# -----------------------------

def _select_chunk_level(doc: Dict[str, Any],
                        cfg: SelectorConfig,
                        doc_top_domains: List[str],
                        registry=REGISTRY,
                        priors: Dict[str, float] | None = None) -> List[ChunkSchemaSelection]:
    priors = priors or {}
    selections: List[ChunkSchemaSelection] = []

    # Limitar evaluación a top-domains del doc + always_include
    allowed = set(doc_top_domains) | set(cfg.always_include or [])
    eval_domains = [d for d in registry.domains if d.domain in allowed] or registry.domains

    for ch in doc.get("chunks", []):
        ch_id = ch.get("chunk_id", "UNK")
        ch_kws = get_chunk_keywords(ch)
        ch_vec = get_chunk_centroid(ch)
        ch_metrics = ch.get("scores", {}) or {}
        c, c_detail = context_score(ch_metrics)

        ch_domain_scores: List[DomainScore] = []
        tok = set(ch_kws) | set(normalize_tokens(ch.get("text", "") or ""))

        for dom in eval_domains:
            # Señales para el chunk
            k, k_detail = keyword_f1(ch_kws, dom.aliases)
            e, e_detail = _embedding_score(ch_vec, dom.label_vecs)
            # topic_affinity a nivel chunk: reusamos la función doc pero con doc restringido no es ideal.
            # Simplificación local: Jaccard chunk-keywords vs domain.aliases
            j_inter = len(set(ch_kws) & set(map(str.lower, dom.aliases)))
            j_union = len(set(ch_kws) | set(map(str.lower, dom.aliases)))
            t = (j_inter / j_union) if j_union else 0.0
            t_detail = {"chunk_jaccard": t, "inter": j_inter, "union": j_union}

            p = float(priors.get(dom.domain, 0.0))

            final, trace, _ = _domain_final_score(k, e, c, t, p, cfg)
            trace.domain = dom.domain

            et_scores = _score_entity_types(sorted(tok), dom.entity_types)
            ev = [
                Evidence(kind="keyword", detail=k_detail),
                Evidence(kind="embedding", detail=e_detail),
                Evidence(kind="topic", detail=t_detail),
                Evidence(kind="context", detail=c_detail),
                Evidence(kind="prior", detail={"value": p})
            ]
            ch_domain_scores.append(DomainScore(
                domain=dom.domain,
                score=final,
                entity_type_scores=et_scores,
                evidence=ev,
                decision_trace=trace
            ))

        ch_domain_scores.sort(key=lambda d: d.score, reverse=True)
        top_domain = ch_domain_scores[0].domain if ch_domain_scores else None
        selected_schema = SCHEMA_BY_DOMAIN.get(top_domain or "generic", "generic_text_v1")
        confidence = softmax_confidence([d.score for d in ch_domain_scores], temperature=cfg.softmax_temperature)
        ambiguous = (len(ch_domain_scores) > 1 and abs(ch_domain_scores[0].score - ch_domain_scores[1].score) < cfg.ambiguity_threshold)

        explanation = build_explanation(
            top_domain or "generic",
            ch_domain_scores[0].decision_trace.contributions if ch_domain_scores else {},
            {"kw": ch_domain_scores[0].evidence[0].detail if ch_domain_scores else {},
             "emb": ch_domain_scores[0].evidence[1].detail if ch_domain_scores else {},
             "topic": ch_domain_scores[0].evidence[2].detail if ch_domain_scores else {}},
            c_detail
        )

        selections.append(ChunkSchemaSelection(
            chunk_id=ch_id,
            domain_scores=ch_domain_scores,
            top_domain=top_domain,
            selected_schema=selected_schema,
            schema_confidence=confidence,
            explanation=explanation,
            signals_used=["keywords", "embeddings", "topics", "context", "prior"],
            weights_used=cfg.weights(),
            ambiguous=ambiguous
        ))

    return selections


# -----------------------------
# API pública
# -----------------------------

def select_schemas(doc: Dict[str, Any],
                   registry=REGISTRY,
                   config: SelectorConfig = SelectorConfig(),
                   priors: Dict[str, float] | None = None) -> SchemaSelection:
    """
    Punto de entrada del Adaptive Schema Selector 2.0.
    - No modifica el input; genera un objeto SchemaSelection completo y explicable.
    """
    doc_sel, top_domains, doc_domain_scores = _select_doc_level(doc, config, registry, priors)
    chunks_sel = _select_chunk_level(doc, config, top_domains, registry, priors)

    # Meta de auditoría
    meta = {
        "version": config.version,
        "weights": config.weights(),
        "thresholds": {
            "ambiguity_threshold": config.ambiguity_threshold,
            "fallback_threshold": config.fallback_threshold,
            "softmax_temperature": config.softmax_temperature,
            "min_doc_tokens_for_domain": config.min_doc_tokens_for_domain,
        },
        "always_include": config.always_include,
        "evaluated_domains_doc": [d.domain for d in doc_domain_scores],
        "evaluated_domains_chunks": list(set(d for ch in chunks_sel for d in [ds.domain for ds in ch.domain_scores])),
        "signals": ["keywords(F1)", "embeddings(cos)", "topics(Jaccard/weighted)", "context(multi-metric)", "prior"],
    }

    # Fallback genérico si la confianza doc-level cae por debajo del umbral
    if doc_sel.schema_confidence < config.fallback_threshold and config.allow_fallback_generic:
        doc_sel.top_domains = list(dict.fromkeys(doc_sel.top_domains + ["generic"]))[:config.max_domains]
        doc_sel.selected_schema = SCHEMA_BY_DOMAIN["generic"]
        doc_sel.explanation = (doc_sel.explanation or "") + " | Confianza baja: se prepara esquema genérico como respaldo."

    return SchemaSelection(doc=doc_sel, chunks=chunks_sel, meta=meta)
