# -*- coding: utf-8 -*-
"""
chunker.py — HybridChunker: segmentación semántica híbrida

Función:
- Convierte DocumentIR+Topics en chunks semánticos estables (≤ max_tokens aprox.).
- Preserva trazabilidad (source_spans) y hereda contexto (topic_hints).
- Calcula métricas de cohesión, redundancia y salud del chunk.
- Puede persistir embedding e idioma por chunk en meta_local.

Estrategia:
1) Pre-segmentación por estructura (headings/listas/bloques).
2) Segmentación por oraciones (spaCy) con fallback por puntuación/saltos.
3) Empaquetado greedy con límites min/max (tokens y caracteres).
4) Afinidad a tópicos globales con score híbrido (coseno + Jaccard).
5) Métricas internas: cohesión, redundancia (coseno+Jaccard), normalización y salud.

Robustez:
- Fallbacks si faltan deps (spaCy, SBERT) o texto.
- No muta topics_doc; lo transporta en meta.
- Salida JSON estable, apta para selector/mentions/grafo.
"""

from __future__ import annotations
import json
import re
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np

from .schemas import (
    Chunk, ChunkingConfig, DocumentChunks, ChunkSourceSpan,
    TopicHints, ChunkingMeta
)

# ----------------------- Logging -----------------------
logger = logging.getLogger("chunker")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------- Dependencias opcionales -----------------------
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_SPACY = False
_HAS_SKLEARN = False
_HAS_LANGDETECT = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    pass

try:
    import spacy  # noqa
    _HAS_SPACY = True
except Exception:
    pass

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    _HAS_SKLEARN = True
except Exception:
    pass

try:
    from langdetect import detect
    _HAS_LANGDETECT = True
except Exception:
    pass


# ----------------------- Utilidades de texto -----------------------
def _estimate_tokens(text: str) -> int:
    """Estimación simple (~4 chars/token) usada para límites de empaquetado."""
    return max(1, int(len(text) / 4))


def _hash_id(s: str) -> str:
    """Hash corto y estable para chunk_id."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10].upper()


def _iter_blocks(ir: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Devuelve lista de bloques textuales válidos: (page_number, block_idx, block_dict)."""
    out = []
    for page in ir.get("pages", []):
        pno = page.get("page_number", 1)
        for i, blk in enumerate(page.get("blocks", [])):
            t = (blk or {}).get("text", "")
            if isinstance(t, str) and t.strip():
                out.append((pno, i, blk))
    return out


def _choose_spacy_model(cfg: ChunkingConfig, lang: Optional[str]) -> str:
    """Elige modelo spaCy según idioma; si no se reconoce, usa el primario."""
    if not lang:
        return cfg.spacy_model
    lang = lang.lower()
    if lang.startswith("es"):
        return cfg.spacy_model
    if lang.startswith("en"):
        return cfg.spacy_model_alt
    return cfg.spacy_model


def _load_spacy(model_name: str):
    """Carga modelo spaCy; si falla, devuelve None (se usa fallback regex)."""
    if not _HAS_SPACY:
        return None
    try:
        import spacy
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer", "textcat"])
        return nlp
    except Exception as e:
        logger.warning(f"[chunker] spaCy no disponible: {e}")
        return None


def _sentence_split(text: str, nlp) -> List[str]:
    """Divide texto en oraciones. Usa spaCy si está; si no, regex robusta."""
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _is_heading(text: str, patterns: List[str]) -> bool:
    """Detecta encabezados por regex y bullets."""
    line = text.strip()
    if len(line) > 180:
        return False
    for rx in patterns:
        if re.compile(rx, re.IGNORECASE).match(line):
            return True
    if re.match(r"^\s*[-•\*]\s+\S+", line):
        return True
    return False


# ----------------------- Tokenización y similitudes léxicas -----------------------
_STOP_ES: Set[str] = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","y","o","u",
    "en","con","por","para","que","se","su","sus","es","son","ser","como","más",
    "no","si","sí","ya","lo","le","les","esto","esta","estas","estos","ese","esa",
    "eso","esas","esos","muy","pero","también","entre","sobre","sin"
}
_STOP_EN: Set[str] = {
    "the","a","an","and","or","of","to","in","on","for","with","as","by","is","are",
    "be","this","that","these","those","it","its","at","from","not","yes","no","very",
    "but","also","between","about","without","into","than","more","most","less","least"
}

def _stopwords(lang: Optional[str]) -> Set[str]:
    """Devuelve stopwords básicas por idioma (conjunto pequeño, embebido)."""
    if not lang:
        return _STOP_ES
    if lang.lower().startswith("en"):
        return _STOP_EN
    return _STOP_ES

def _tokens(text: str, lang: Optional[str]) -> List[str]:
    """Tokeniza en minúsculas, filtra no alfanum y stopwords."""
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-záéíóúüñ\s]", " ", text)
    toks = [t for t in text.split() if t.strip()]
    sw = _stopwords(lang)
    return [t for t in toks if t not in sw]

def _token_set(text: str, lang: Optional[str]) -> Set[str]:
    """Conjunto de tokens filtrados para medidas tipo Jaccard."""
    return set(_tokens(text, lang))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Similitud Jaccard entre dos conjuntos (0..1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0

def _lexical_intra_cohesion(sentences: List[str], lang: Optional[str]) -> float:
    """
    Cohesión léxica intra-chunk como promedio de Jaccard entre oraciones adyacentes.
    Si hay 1 oración, devuelve 1.0 por definición.
    """
    if not sentences:
        return 0.0
    if len(sentences) == 1:
        return 1.0
    sims = []
    prev = _token_set(sentences[0], lang)
    for s in sentences[1:]:
        cur = _token_set(s, lang)
        sims.append(_jaccard(prev, cur))
        prev = cur
    return float(np.mean(sims)) if sims else 0.0


# ----------------------- Embeddings / Vectorizadores -----------------------
class _Embedder:
    """Usa SBERT si está, si no HashingVectorizer; si no, vector trivial por longitud."""

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.kind = "none"
        self.model = None
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.kind = "sbert"
            except Exception as e:
                logger.warning(f"[chunker] SBERT no disponible: {e}")
        if self.model is None and _HAS_SKLEARN:
            self.model = HashingVectorizer(n_features=2**12, alternate_sign=False)
            self.kind = "hash"

    def encode(self, texts: List[str]) -> np.ndarray:
        """Devuelve matriz (n × d) de embeddings."""
        if self.kind == "sbert":
            return np.array(self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False))
        if self.kind == "hash":
            mat = self.model.transform(texts)  # sparse
            return mat.toarray().astype(np.float32)
        return np.array([[len(t)] for t in texts], dtype=np.float32)

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno promedio entre vectores, robusta a 1D/2D."""
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        den = np.where(den == 0, 1e-6, den)
        return float(np.mean(num / den))


def _topic_keyword_centroids(topics_doc: Optional[Dict[str, Any]], embedder: _Embedder) -> Dict[int, np.ndarray]:
    """Centroides de embeddings por topic global usando sus keywords."""
    if not topics_doc:
        return {}
    centroids: Dict[int, np.ndarray] = {}
    for t in topics_doc.get("topics", []):
        tid = int(t.get("topic_id"))
        kws = t.get("keywords", []) or []
        if not kws:
            continue
        vecs = embedder.encode(kws)
        centroids[tid] = vecs.mean(axis=0)
    return centroids


# ----------------------- Topic hints (coseno + Jaccard) -----------------------
def _build_topic_hints_fn(cfg: ChunkingConfig, topics_doc: Optional[Dict[str, Any]], embedder: _Embedder, lang: Optional[str]):
    """
    Devuelve función que asigna hints temáticos por chunk con score híbrido:
    score = blend * coseno(embedding_chunk, centroide_topic) + (1-blend) * Jaccard(tokens_chunk, keywords_topic)
    """
    centroids = _topic_keyword_centroids(topics_doc, embedder)

    # Prepara keyword-sets por topic para Jaccard:
    topic_kw_sets: Dict[int, Set[str]] = {}
    if topics_doc:
        for t in topics_doc.get("topics", []):
            tid = int(t.get("topic_id"))
            kws = [str(k) for k in (t.get("keywords", []) or [])]
            topic_kw_sets[tid] = set([k.lower() for k in kws if k])

    blend = float(cfg.topic_affinity_blend)

    def builder(text: str) -> TopicHints:
        if not topics_doc or (not centroids and not topic_kw_sets):
            return TopicHints()

        tv = embedder.encode([text])[0]  # vector del chunk
        tokens = _token_set(text, lang)

        scored: List[Tuple[int, float]] = []
        for tid in set(list(centroids.keys()) + list(topic_kw_sets.keys())):
            cos_part = _Embedder.cos_sim(tv, centroids.get(tid)) if tid in centroids else 0.0
            jac_part = _jaccard(tokens, topic_kw_sets.get(tid, set()))
            score = blend * float(cos_part) + (1.0 - blend) * float(jac_part)
            scored.append((tid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        k = min(max(1, cfg.topic_affinity_topk), len(scored))
        topk = scored[:k]

        topic_ids = [t for t, _ in topk]
        scores = {str(t): float(s) for t, s in topk}
        inherited_kws = (topics_doc.get("keywords_global") or [])[:10]
        return TopicHints(
            inherited_topic_ids=topic_ids,
            inherited_keywords=inherited_kws,
            topic_affinity=scores
        )

    return builder


# ----------------------- Pre-segmentación por headings -----------------------
def _presegment_blocks(
    blocks: List[Tuple[int, int, Dict[str, Any]]],
    cfg: ChunkingConfig
) -> List[Tuple[List[Tuple[int, int, Dict[str, Any]]], bool]]:
    """Agrupa bloques en secciones naturales: [(lista_bloques, comienza_con_heading_bool)]."""
    if not cfg.prefer_headings:
        return [(blocks, False)]
    sections: List[Tuple[List[Tuple[int, int, Dict[str, Any]]], bool]] = []
    current: List[Tuple[int, int, Dict[str, Any]]] = []
    current_is_heading = False
    for tup in blocks:
        _, _, blk = tup
        text = blk.get("text", "")
        if _is_heading(text, cfg.heading_patterns):
            if current:
                sections.append((current, current_is_heading))
            current = [tup]
            current_is_heading = True
        else:
            current.append(tup)
    if current:
        sections.append((current, current_is_heading))
    return sections or [(blocks, False)]


# ----------------------- Empaquetado de oraciones a chunks -----------------------
def _pack_sentences_to_chunks(
    doc_id: str,
    sentences_with_spans: List[Tuple[str, int, int]],
    cfg: ChunkingConfig,
    topic_hints_builder,
    embedder: _Embedder,
    lang: Optional[str]
) -> List[Chunk]:
    """
    Empaqueta oraciones en chunks respetando límites y preservando coherencia.
    sentences_with_spans: [(texto_oración, page_number, block_index), ...]
    """
    chunks: List[Chunk] = []
    buf_texts: List[str] = []
    buf_spans: List[ChunkSourceSpan] = []
    buf_chars = 0
    order = 0

    def flush():
        """Cierra el buffer y crea un chunk si hay contenido."""
        nonlocal chunks, buf_texts, buf_spans, buf_chars, order
        if not buf_texts:
            return
        text = " ".join(buf_texts).strip()
        if not text:
            buf_texts, buf_spans, buf_chars = [], [], 0
            return

        est = _estimate_tokens(text)
        chunk_id = f"{doc_id[:8]}_{order:04d}_{_hash_id(text)}"
        thints = topic_hints_builder(text)

        # Métricas léxicas internas (cohesión intra, densidad, ttr, longitud media oraciones)
        intra_lex = _lexical_intra_cohesion(buf_texts, lang)
        toks = _tokens(text, lang)
        uniq = set(toks)
        lexical_density = float(len([t for t in toks if t not in _stopwords(lang)])) / float(len(toks)) if toks else 0.0
        ttr = float(len(uniq)) / float(len(toks)) if toks else 0.0
        avg_sent_len_chars = float(np.mean([len(s) for s in buf_texts])) if buf_texts else 0.0

        meta_local = {}
        if cfg.save_embeddings:
            vec = embedder.encode([text])[0]
            meta_local["embedding"] = vec.tolist()
            if lang:
                meta_local["lang"] = lang

        scores = {
            "intra_cohesion_lex": round(intra_lex, 6),
            "lexical_density": round(lexical_density, 6),
            "type_token_ratio": round(ttr, 6),
            "avg_sentence_len_chars": round(avg_sent_len_chars, 2),
        }

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                order=order,
                text=text,
                char_len=len(text),
                est_tokens=est,
                source_spans=buf_spans or [],
                topic_hints=thints,
                scores=scores,
                meta_local=meta_local
            )
        )
        order += 1
        buf_texts, buf_spans, buf_chars = [], [], 0

    for sent, pno, bidx in sentences_with_spans:
        sent = sent.strip()
        if not sent:
            continue

        # Oración gigantesca → chunk forzado
        if len(sent) > cfg.max_chars:
            if buf_texts:
                flush()
            buf_texts = [sent]
            buf_spans = [ChunkSourceSpan(page_number=pno, block_indices=[bidx])]
            flush()
            continue

        # ¿Cabe en el buffer?
        next_chars = buf_chars + len(sent)
        next_tokens = _estimate_tokens(" ".join(buf_texts + [sent]))
        if (next_chars > cfg.max_chars) or (next_tokens > cfg.max_tokens):
            if buf_chars >= cfg.min_chars:
                flush()
            else:
                flush()

        # Añadir oración y span
        buf_texts.append(sent)
        if buf_spans and buf_spans[-1].page_number == pno:
            if bidx not in buf_spans[-1].block_indices:
                buf_spans[-1].block_indices.append(bidx)
        else:
            buf_spans.append(ChunkSourceSpan(page_number=pno, block_indices=[bidx]))
        buf_chars += len(sent)

    if buf_texts:
        flush()
    return chunks


# ----------------------- Métricas semánticas (coseno + Jaccard) -----------------------
def _compute_scores(chunks: List[Chunk], embedder: _Embedder, cfg: ChunkingConfig, lang: Optional[str]) -> None:
    """
    Anota en cada chunk:
    - cohesion_vs_doc        : coseno(chunk, centroide_doc)
    - redundancy_cos_max     : máx coseno vs otro chunk
    - redundancy_jaccard_max : máx Jaccard léxico vs otro chunk
    - redundancy_mix         : α*cos + β*jaccard
    - redundancy_norm        : redundancy_mix * clamp(len/avg_len, 0.5, 2.0)
    - novelty                : 1 - redundancy_mix
    - chunk_health           : w_cohesion * cohesion_vs_doc + w_novelty * novelty
    """
    if not chunks:
        return

    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts)
    doc_vec = vecs.mean(axis=0)

    # Cohesión con documento (coseno)
    sims_doc = [float(_Embedder.cos_sim(v, doc_vec)) for v in vecs]

    # Redundancia coseno entre chunks
    dots = np.inner(vecs, vecs)
    norms = np.outer(np.linalg.norm(vecs, axis=1), np.linalg.norm(vecs, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        sims_all = np.divide(dots, norms, where=norms != 0)
    np.fill_diagonal(sims_all, 0.0)
    red_cos_max = sims_all.max(axis=1) if len(chunks) > 1 else np.zeros(len(chunks))

    # Redundancia Jaccard léxica
    token_sets = [_token_set(t, lang) for t in texts]
    red_jac_max = []
    for i in range(len(chunks)):
        if len(chunks) == 1:
            red_jac_max.append(0.0)
            continue
        best = 0.0
        for j in range(len(chunks)):
            if i == j:
                continue
            jacc = _jaccard(token_sets[i], token_sets[j])
            if jacc > best:
                best = jacc
        red_jac_max.append(best)
    red_jac_max = np.array(red_jac_max, dtype=float)

    # Mezcla y normalización
    alpha = float(cfg.alpha_cos_redundancy)
    beta = float(cfg.beta_jaccard_redundancy)
    mix = alpha * red_cos_max + beta * red_jac_max

    def clamp(x: float, lo: float = 0.5, hi: float = 2.0) -> float:
        return max(lo, min(hi, x))

    avg_len = np.mean([len(c.text) for c in chunks]) if chunks else 1.0

    for i, c in enumerate(chunks):
        coh = sims_doc[i]
        red_cos = float(red_cos_max[i])
        red_jac = float(red_jac_max[i])
        red_mix = float(mix[i])
        scale = clamp(len(c.text) / max(1.0, avg_len))
        red_norm = red_mix * scale
        novelty = 1.0 - red_mix
        health = float(cfg.w_cohesion) * coh + float(cfg.w_novelty) * novelty

        c.scores.update({
            "cohesion_vs_doc": round(coh, 6),
            "redundancy_cos_max": round(red_cos, 6),
            "redundancy_jaccard_max": round(red_jac, 6),
            "redundancy_mix": round(red_mix, 6),
            "redundancy_norm": round(red_norm, 6),
            "novelty": round(novelty, 6),
            "chunk_health": round(health, 6),
        })


# ----------------------- Orquestador principal -----------------------
def run(
    ir_with_topics: Dict[str, Any],
    cfg: Optional[ChunkingConfig] = None
) -> DocumentChunks:
    """
    Punto de entrada del chunker:
    - recibe IR+Topics (dict),
    - aplica segmentación y empaquetado,
    - anota métricas,
    - devuelve DocumentChunks (objeto Pydantic).
    """
    cfg = cfg or ChunkingConfig()
    np.random.seed(cfg.seed)

    doc_id = ir_with_topics.get("doc_id") or _hash_id(json.dumps(ir_with_topics)[:256])
    source_path = ir_with_topics.get("source_path")
    mime = ir_with_topics.get("mime")
    lang = ir_with_topics.get("lang") or (ir_with_topics.get("meta") or {}).get("lang")

    # Detección de idioma opcional si no hay lang
    if cfg.detect_lang and not lang and _HAS_LANGDETECT:
        sample = []
        for _, _, blk in _iter_blocks(ir_with_topics)[:50]:
            t = blk.get("text", "")
            if isinstance(t, str) and t.strip():
                sample.append(t.strip())
            if sum(len(s) for s in sample) > 2000:
                break
        try:
            lang = detect(" ".join(sample)) if sample else cfg.lang_hint
        except Exception:
            lang = cfg.lang_hint
    lang = lang or cfg.lang_hint

    # spaCy y embedder
    spacy_name = _choose_spacy_model(cfg, lang)
    nlp = _load_spacy(spacy_name)
    embedder = _Embedder(cfg.embedding_model, cfg.embedding_batch_size)

    # Topics globales
    topics_doc = ((ir_with_topics.get("meta") or {}).get("topics_doc")) or (ir_with_topics.get("topics_doc"))
    topic_hints_builder = _build_topic_hints_fn(cfg, topics_doc, embedder, lang)

    # 1) Bloques y 2) Secciones
    blocks = _iter_blocks(ir_with_topics)
    sections = _presegment_blocks(blocks, cfg)

    # 3) Oraciones por sección y empaquetado a chunks
    all_chunks: List[Chunk] = []
    for sec_blocks, _starts_with_heading in sections:
        sentences_with_spans: List[Tuple[str, int, int]] = []
        for pno, bidx, blk in sec_blocks:
            sents = _sentence_split(blk.get("text", "") or "", nlp)
            for s in sents:
                if len(s) < 3:
                    continue
                sentences_with_spans.append((s, pno, bidx))
        sec_chunks = _pack_sentences_to_chunks(
            doc_id=doc_id,
            sentences_with_spans=sentences_with_spans,
            cfg=cfg,
            topic_hints_builder=topic_hints_builder,
            embedder=embedder,
            lang=lang
        )
        all_chunks.extend(sec_chunks)

    # 4) Métricas semánticas y de redundancia
    if cfg.use_embeddings and all_chunks:
        _compute_scores(all_chunks, embedder, cfg, lang)

    # 5) Metadatos y salida
    stats = {
        "n_sections": len(sections),
        "n_blocks": len(blocks),
        "n_chunks": len(all_chunks),
        "avg_chunk_chars": float(np.mean([c.char_len for c in all_chunks])) if all_chunks else 0.0,
        "avg_chunk_tokens": float(np.mean([c.est_tokens for c in all_chunks])) if all_chunks else 0.0,
    }
    # agregados si hay métricas
    if all_chunks and all_chunks[0].scores:
        keys = [
            "cohesion_vs_doc", "redundancy_cos_max", "redundancy_jaccard_max",
            "redundancy_mix", "redundancy_norm", "novelty", "chunk_health",
            "intra_cohesion_lex", "lexical_density", "type_token_ratio", "avg_sentence_len_chars"
        ]
        for k in keys:
            vals = [c.scores.get(k, 0.0) for c in all_chunks]
            stats[f"avg_{k}"] = float(np.mean(vals))

    meta = ChunkingMeta(
        config=cfg.model_dump(),
        topics_doc=topics_doc,
        stats=stats
    )

    out = DocumentChunks(
        doc_id=doc_id,
        source_path=source_path,
        mime=mime,
        lang=lang,
        chunks=all_chunks,
        meta=meta
    )
    return out


# ----------------------- IO helpers alineados al proyecto -----------------------
def load_ir_with_topics(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: DocumentChunks, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{chunks.doc_id}_chunks.json"
    p = outdir / fname
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            chunks.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2
        )
    logger.info(f"[chunker] Guardado: {p}")
    return p
