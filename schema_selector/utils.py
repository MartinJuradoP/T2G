# -*- coding: utf-8 -*-
"""
utils.py — Utilidades para Adaptive Schema Selector 2.0 (mejorado)

Incluye:
- Normalización y tokenización robusta (multi-idioma básico).
- Extracción de keywords a nivel doc/chunk (topics_doc/topics_chunks/topic).
- Cálculo de centroides de embeddings con degradación elegante.
- Cómputo de señales: keyword F1, topic_affinity, context_score (cohesión, salud, etc.).
- Fusión de señales y confianza softmax con temperatura.
- Generación de explicaciones con contribuciones dominantes.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Iterable
import numpy as np
import re
from math import exp


# ---------------------------------------------------------------------------
# 🔹 Texto y tokens
# ---------------------------------------------------------------------------

def normalize_keyword(text: str) -> str:
    return text.lower().strip() if text else ""


def normalize_tokens(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    t = re.sub(r"[^0-9a-záéíóúüñäëïöüçàèìòùâêîôû\s]", " ", t, flags=re.UNICODE)
    return [w for w in t.split() if w]


def unique_norm(it: Iterable[str]) -> List[str]:
    return sorted(set(normalize_keyword(x) for x in it if x))


# ---------------------------------------------------------------------------
# 🔹 Embeddings
# ---------------------------------------------------------------------------

def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def normalize_vector(v: List[float]) -> List[float]:
    arr = np.array(v, dtype=float)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def get_doc_centroid(doc: Dict[str, Any]) -> List[float]:
    embeddings: List[List[float]] = []
    # Preferimos embeddings de chunks (meta_local.embedding)
    for ch in doc.get("chunks", []):
        emb = ch.get("meta_local", {}).get("embedding")
        if emb:
            embeddings.append(emb)
    # Fallback: embeddings agregados en topics_doc (si existieran)
    tdoc = doc.get("meta", {}).get("topics_doc", {})
    embeddings.extend(tdoc.get("embeddings", []))
    if not embeddings:
        return []
    return normalize_vector(np.mean(np.array(embeddings, dtype=float), axis=0).tolist())


def get_chunk_centroid(ch: Dict[str, Any]) -> List[float]:
    emb = ch.get("meta_local", {}).get("embedding")
    if emb:
        return normalize_vector(emb)
    # Fallback: promedio de embeddings de oraciones (si existen)
    s_embs = []
    for s in ch.get("sentences", []):
        e = s.get("meta", {}).get("embedding")
        if e:
            s_embs.append(e)
    if not s_embs:
        return []
    return normalize_vector(np.mean(np.array(s_embs, dtype=float), axis=0).tolist())


# ---------------------------------------------------------------------------
# 🔹 Keywords de doc y chunk (topics)
# ---------------------------------------------------------------------------

def get_doc_keywords(doc: Dict[str, Any]) -> List[str]:
    kws: List[str] = []
    meta = doc.get("meta", {})
    tdoc = meta.get("topics_doc", {})
    if "keywords_global" in tdoc:
        kws.extend(tdoc.get("keywords_global", []))
    # También agregamos keywords globales de topics_chunks (si existen)
    tch = meta.get("topics_chunks", {})
    if "keywords_global" in tch:
        kws.extend(tch.get("keywords_global", []))
    return unique_norm(kws)


def get_chunk_keywords(ch: Dict[str, Any]) -> List[str]:
    kws: List[str] = []
    if "topic" in ch and isinstance(ch["topic"], dict):
        kws.extend(ch["topic"].get("keywords", []))
    # Si hay meta.topics_chunk
    mt = ch.get("meta", {}).get("topics_chunk", {})
    if "keywords" in mt:
        kws.extend(mt.get("keywords", []))
    return unique_norm(kws)


def count_doc_tokens(doc: Dict[str, Any]) -> int:
    total = 0
    for ch in doc.get("chunks", []):
        total += len(normalize_tokens(ch.get("text", "") or ""))
    return total


# ---------------------------------------------------------------------------
# 🔹 Señales: Keywords (F1), Topics, Context
# ---------------------------------------------------------------------------

def keyword_f1(doc_kws: List[str], domain_aliases: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    F1 entre keywords del doc y alias del dominio:
      P = matches / len(doc_kws)
      R = matches / len(domain_aliases)
      F1 = 2PR/(P+R)
    """
    if not doc_kws or not domain_aliases:
        return 0.0, {"matches": 0, "precision": 0.0, "recall": 0.0}
    a_doc = set(doc_kws)
    a_dom = set(unique_norm(domain_aliases))
    inter = a_doc & a_dom
    matches = len(inter)
    precision = matches / max(1, len(a_doc))
    recall = matches / max(1, len(a_dom))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return float(f1), {"matches": matches, "precision": precision, "recall": recall}


def topic_affinity_score(doc: Dict[str, Any], domain_aliases: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Mide la afinidad tema-dominio combinando:
    - topics_doc.topics: sum(prob * overlap_kw)
    - topics chunk-level: promedio de overlaps por chunk (ponderado por chunk_health)
    Cada overlap_kw es Jaccard simple entre keywords del topic y aliases del dominio.
    """
    aliases = set(unique_norm(domain_aliases))
    meta = doc.get("meta", {})
    score_doc = 0.0
    used = 0

    # Doc-level topics
    tdoc = meta.get("topics_doc", {})
    topics = tdoc.get("topics", [])  # [{'topic_id':..., 'keywords':[...], 'prob':...}]
    for t in topics or []:
        kws = set(unique_norm(t.get("keywords", [])))
        if not kws:
            continue
        inter = len(kws & aliases)
        union = len(kws | aliases)
        j = inter / union if union else 0.0
        score_doc += j * float(t.get("prob", 0.0))
        used += 1

    # Chunk-level topics (ponderado por salud)
    score_chunks = 0.0
    used_chunks = 0
    for ch in doc.get("chunks", []):
        ckws = set(get_chunk_keywords(ch))
        inter = len(ckws & aliases)
        union = len(ckws | aliases)
        j = inter / union if union else 0.0
        health = float(ch.get("scores", {}).get("chunk_health", 1.0))
        score_chunks += j * health
        used_chunks += 1

    # Normalizamos por contajes usados
    score = 0.0
    parts = 0
    if used > 0:
        score += score_doc / used
        parts += 1
    if used_chunks > 0:
        score += score_chunks / used_chunks
        parts += 1
    score = score / parts if parts > 0 else 0.0

    detail = {
        "doc_topics_used": used,
        "chunk_topics_used": used_chunks,
        "score_doc_topics": score_doc,
        "score_chunk_topics": score_chunks
    }
    return float(score), detail


def context_score(metrics: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Fusión de métricas contextuales a [0,1]:
      C = 0.30*cohesion_vs_doc + 0.30*chunk_health
        + 0.20*(1 - redundancy_norm) + 0.10*novelty
        + 0.10*richness
    donde richness = 0.5*lexical_density + 0.5*type_token_ratio
    """
    cvd = float(metrics.get("cohesion_vs_doc", 0.0))
    hlt = float(metrics.get("chunk_health", 0.0))
    red = float(metrics.get("redundancy_norm", 0.0))
    nov = float(metrics.get("novelty", 0.0))
    lex = float(metrics.get("lexical_density", 0.0))
    ttr = float(metrics.get("type_token_ratio", 0.0))
    richness = 0.5 * lex + 0.5 * ttr
    c = (0.30 * cvd) + (0.30 * hlt) + (0.20 * (1.0 - red)) + (0.10 * nov) + (0.10 * richness)
    return max(0.0, min(1.0, float(c))), {
        "cohesion_vs_doc": cvd, "chunk_health": hlt, "redundancy_norm": red,
        "novelty": nov, "lexical_density": lex, "type_token_ratio": ttr, "richness": richness
    }


def aggregate_doc_metrics(doc: Dict[str, Any]) -> Dict[str, float]:
    """Promedia métricas 'scores' de los chunks para obtener contexto global del documento."""
    keys = ["cohesion_vs_doc", "chunk_health", "redundancy_norm", "novelty", "lexical_density", "type_token_ratio"]
    acc: Dict[str, List[float]] = {k: [] for k in keys}
    for ch in doc.get("chunks", []):
        sc = ch.get("scores", {}) or {}
        for k in keys:
            v = sc.get(k)
            if isinstance(v, (int, float)):
                acc[k].append(float(v))
    out: Dict[str, float] = {}
    for k, vals in acc.items():
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# 🔹 Confianza (softmax con temperatura) y explicaciones
# ---------------------------------------------------------------------------

def softmax_confidence(scores: List[float], temperature: float = 1.0) -> float:
    """Confianza del top-score frente al resto con temperatura (T<1 endurece la distribución)."""
    if not scores:
        return 0.0
    scaled = [s / max(1e-6, temperature) for s in scores]
    exps = [exp(s) for s in scaled]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    return float(max(probs))


def build_explanation(domain: str,
                      contributions: Dict[str, float],
                      ev_summary: Dict[str, Any],
                      context_details: Dict[str, Any]) -> str:
    """
    Ensambla una explicación textual breve ponderada por contribuciones dominantes.
    contributions: {"K": αK, "E": βE, "C": γC, "T": δT, "P": εP}
    ev_summary: {"kw":{...}, "emb":{...}, "topic":{...}}
    context_details: métricas usadas en C
    """
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    drivers = [k for k, v in ranked if v > 0]
    pieces = []
    label = {
        "K": "afinidad léxica (keywords)",
        "E": "similitud semántica (embeddings)",
        "C": "calidad contextual (cohesión/salud/novelty/baja redundancia)",
        "T": "alineación con tópicos",
        "P": "prior organizacional"
    }
    for d in drivers[:3]:
        pieces.append(label[d])
    if not pieces:
        return f"Dominio '{domain}' seleccionado con señales débiles; se recomienda fallback genérico."
    # Añadimos un par de detalles numéricos útiles si existen
    dets = []
    if "kw" in ev_summary and "precision" in ev_summary["kw"]:
        dets.append(f"precisión_kw={ev_summary['kw']['precision']:.2f}")
    if "kw" in ev_summary and "recall" in ev_summary["kw"]:
        dets.append(f"recall_kw={ev_summary['kw']['recall']:.2f}")
    if "emb" in ev_summary and "best_label" in ev_summary["emb"]:
        dets.append(f"label_emb={ev_summary['emb']['best_label']}")
    if "context" in context_details:
        pass  # context_details ya viene desplegado

    base = f"Dominio '{domain}' seleccionado por " + ", ".join(pieces) + "."
    if dets:
        base += " (" + ", ".join(dets) + ")"
    return base
