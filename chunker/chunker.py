# -*- coding: utf-8 -*-
"""
chunker.py — HybridChunker: regla + semántica + límites de longitud (versión mejorada, compatible)

Estrategia híbrida:
1) Pre-segmentación por estructura (headings/listas/bloques).
2) Refinamiento por oraciones (spaCy) para no cortar ideas (fallback robusto si spaCy no está).
3) Empaquetado greedy por longitud (≤ max_tokens aprox.) con control de min_chars y max_chars.
4) Cálculo opcional de embeddings para:
   - cohesión chunk ↔ documento (centroide global)
   - redundancia inter-chunk (máxima similitud)
   - afinidad con topics_doc (centroides de keywords + solapamiento léxico)

Robustez:
- Si embeddings no disponibles, fallback a HashingVectorizer.
- Si spaCy no disponible, fallback por regex de puntuación/saltos.
- Siempre transpone `topics_doc` del IR de entrada a meta de salida (sin mutarlo).
- Detección de idioma opcional (langdetect) → intenta elegir modelo spaCy adecuado; si falla, usa cfg.spacy_model.


"""

from __future__ import annotations
import json
import re
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .schemas import (
    Chunk, ChunkingConfig, DocumentChunks, ChunkSourceSpan, TopicHints, ChunkingMeta
)

# ====== Logging ======
logger = logging.getLogger("chunker")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ====== Optional deps ======
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_SKLEARN = False
_HAS_SPACY = False
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
    from langdetect import detect  # muy ligero; si no está, no rompemos
    _HAS_LANGDETECT = True
except Exception:
    pass


# ====== Parámetros semánticos (afinidad y scoring interno) ======
_AFF_ALPHA_SEM = 0.7  # peso señal semántica (cosine con centroides de topic)
_AFF_BETA_LEX = 0.3   # peso señal léxica (Jaccard chunk-tokens vs keywords topic)


# ====== Utiles básicos ======
def _estimate_tokens(text: str) -> int:
    """Estimación simple y estable (≈4 chars/token)."""
    return max(1, int(len(text) / 4))


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10].upper()


def _iter_blocks(ir: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Devuelve (page_number, block_idx, block_dict) solo para texto útil.
    """
    out = []
    for page in ir.get("pages", []):
        pno = page.get("page_number", 1)
        for i, blk in enumerate(page.get("blocks", [])):
            t = (blk or {}).get("text", "")
            if not isinstance(t, str) or not t.strip():
                continue
            out.append((pno, i, blk))
    return out


def _concat_blocks_text(blocks: List[Tuple[int, int, Dict[str, Any]]]) -> str:
    """Concatena el texto de todos los bloques para heurísticas globales (p.ej. detección de idioma)."""
    return "\n".join((b or {}).get("text", "") for _, _, b in blocks if (b or {}).get("text", ""))


def _detect_lang_heuristic(ir: Dict[str, Any], fallback: Optional[str]) -> Optional[str]:
    """
    Detecta idioma sobre el documento si no viene en IR.lang:
    - langdetect si está disponible.
    - Si falla, retorna fallback (cfg.spacy_model) o None.
    """
    if _HAS_LANGDETECT:
        try:
            # Intentamos detectar sobre meta texto suficiente (todos los bloques concatenados)
            blocks = _iter_blocks(ir)
            raw = _concat_blocks_text(blocks)[:20000]  # límite razonable
            if raw and len(raw) >= 40:
                ld = detect(raw)  # devuelve 'es', 'en', 'fr', ...
                return ld
        except Exception as e:
            logger.warning(f"[chunker] langdetect fallo: {e}")
    # fallback
    return None if (fallback is None or not fallback) else (fallback)


def _choose_spacy_model(lang_code: Optional[str], cfg_model: str) -> str:
    """
    Selecciona modelo spaCy según código de idioma aproximado.
    Si no lo reconoce o no está disponible, usa cfg_model.
    """
    if not lang_code:
        return cfg_model
    lang_code = lang_code.lower()
    mapping = {
        "es": "es_core_news_sm",
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm"
    }
    return mapping.get(lang_code, cfg_model)


def _load_spacy(model_name: str):
    if not _HAS_SPACY:
        return None
    try:
        import spacy
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer", "textcat"])
        return nlp
    except Exception as e:
        logger.warning(f"[chunker] spaCy model '{model_name}' no disponible: {e}")
        return None


def _sentence_split(text: str, nlp) -> List[str]:
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # Fallback robusto
    parts = re.split(r"(?<=[\.\?\!])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _is_heading(text: str, patterns: List[str]) -> bool:
    line = text.strip()
    if len(line) > 180:  # headings razonables
        return False
    for rx in patterns:
        if re.compile(rx, flags=re.IGNORECASE).match(line):
            return True
    # bullets/listas como pseudo-heading
    if re.match(r"^\s*[-•\*]\s+\S+", line):
        return True
    return False


def _normalize_tokens(text: str) -> List[str]:
    """Normaliza y tokeniza texto en palabras básicas (lowercase y limpieza leve)."""
    if not text:
        return []
    t = text.lower()
    t = re.sub(r"[^0-9a-záéíóúüñ\s]", " ", t)
    return [w for w in t.split() if w.strip()]


def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard simple entre dos listas de tokens."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / max(1, union))


# ====== Embeddings / Vectorizadores ======
class _Embedder:
    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.kind = "none"
        self.model = None
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.kind = "sbert"
                return
            except Exception as e:
                logger.warning(f"[chunker] No se pudo cargar SBERT '{model_name}': {e}")
        if _HAS_SKLEARN:
            self.model = HashingVectorizer(n_features=2**12, alternate_sign=False)
            self.kind = "hash"
            return

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.kind == "sbert":
            return np.array(self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False))
        if self.kind == "hash":
            mat = self.model.transform(texts)  # sparse
            return mat.toarray().astype(np.float32)
        # Sin embeddings: devolvemos longitud como vector dummy
        arr = np.array([[len(t)] for t in texts], dtype=np.float32)
        return arr

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        den = np.where(den == 0, 1e-6, den)
        return float(np.mean(num / den))


def _topic_keyword_centroids(topics_doc: Optional[Dict[str, Any]], embedder: _Embedder) -> Dict[int, np.ndarray]:
    """
    Calcula centroides por topic_id a partir de sus keywords (si existen).
    """
    if not topics_doc:
        return {}
    centroids: Dict[int, np.ndarray] = {}
    for t in topics_doc.get("topics", []):
        try:
            tid = int(t.get("topic_id"))
        except Exception:
            continue
        kws = t.get("keywords", []) or []
        if not kws:
            continue
        vecs = embedder.encode(kws)
        if vecs.size == 0:
            continue
        centroids[tid] = vecs.mean(axis=0)
    return centroids


# ====== Núcleo de chunking ======
def _presegment_blocks(
    blocks: List[Tuple[int, int, Dict[str, Any]]],
    cfg: ChunkingConfig
) -> List[Tuple[List[Tuple[int, int, Dict[str, Any]]], bool]]:
    """
    Agrupa bloques en secciones preliminares. Cada item = (lista_de_bloques, inicia_con_heading_bool)
    """
    if not cfg.prefer_headings:
        return [(blocks, False)]
    sections: List[Tuple[List[Tuple[int, int, Dict[str, Any]]], bool]] = []
    current: List[Tuple[int, int, Dict[str, Any]]] = []
    current_is_heading = False

    for tup in blocks:
        _, _, blk = tup
        text = blk.get("text", "")
        if _is_heading(text, cfg.heading_patterns):
            # cerramos sección anterior
            if current:
                sections.append((current, current_is_heading))
            current = [tup]
            current_is_heading = True
        else:
            current.append(tup)

    if current:
        sections.append((current, current_is_heading))

    if not sections:
        sections = [(blocks, False)]
    return sections


def _pack_sentences_to_chunks(
    doc_id: str,
    sentences_with_spans: List[Tuple[str, int, int]],
    cfg: ChunkingConfig,
    topic_hints_builder
) -> List[Chunk]:
    """
    Empaqueta oraciones en chunks respetando límites y manteniendo coherencia básica.
    sentences_with_spans: lista de (texto_oración, page_number, block_index)
    """
    chunks: List[Chunk] = []
    buf_texts: List[str] = []
    buf_spans: List[ChunkSourceSpan] = []
    buf_chars = 0
    order = 0 if not chunks else (chunks[-1].order + 1)  # (solo por claridad; realmente resume fuera)

    def flush():
        nonlocal chunks, buf_texts, buf_spans, buf_chars, order
        if not buf_texts:
            return
        text = " ".join(buf_texts).strip()
        if not text:
            buf_texts, buf_spans, buf_chars = [], [], 0
            return
        est = _estimate_tokens(text)
        chunk_id = f"{doc_id[:8]}_{len(chunks):04d}_{_hash_id(text)}"
        thints = topic_hints_builder(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                order=len(chunks),  # orden global creciente y estable
                text=text,
                char_len=len(text),
                est_tokens=est,
                source_spans=buf_spans or [],
                topic_hints=thints,
                scores={}
            )
        )
        buf_texts, buf_spans, buf_chars = [], [], 0

    for sent, pno, bidx in sentences_with_spans:
        sent = sent.strip()
        if not sent:
            continue
        # si una oración sola excede max_chars, flusheamos el buffer y la forzamos como chunk
        if len(sent) > cfg.max_chars:
            if buf_texts:
                flush()
            buf_texts = [sent]
            buf_spans = [ChunkSourceSpan(page_number=pno, block_indices=[bidx])]
            flush()
            continue

        # ¿Cabe en el buffer actual?
        if (buf_chars + len(sent) > cfg.max_chars) or (_estimate_tokens(" ".join(buf_texts + [sent])) > cfg.max_tokens):
            if buf_chars >= cfg.min_chars:
                flush()
            else:
                # flush suave: estaba corto pero no cabe la nueva; cerramos igual
                flush()

        # Añadimos
        buf_texts.append(sent)
        # agregamos span (acumular por página)
        if buf_spans and buf_spans[-1].page_number == pno:
            if bidx not in buf_spans[-1].block_indices:
                buf_spans[-1].block_indices.append(bidx)
        else:
            buf_spans.append(ChunkSourceSpan(page_number=pno, block_indices=[bidx]))
        buf_chars += len(sent)

    # flush final
    if buf_texts:
        flush()

    return chunks


def _build_topic_hints_fn(cfg: ChunkingConfig, topics_doc: Optional[Dict[str, Any]], embedder: _Embedder):
    """
    Construye una función que, dado el texto de un chunk, devuelva TopicHints:
    - inherited_topic_ids: topK por afinidad semántica con centroides de topics_doc
    - inherited_keywords: mezcla de keywords_global + keywords del mejor topic
    - topic_affinity: dict topic_id → score (combinación semántica+léxica normalizada)
    """
    centroids = _topic_keyword_centroids(topics_doc, embedder)
    keywords_global = (topics_doc or {}).get("keywords_global") or []
    topics_list = (topics_doc or {}).get("topics") or []

    # Preparamos tabla topic_id -> keywords (para señal léxica)
    topic_kw_map: Dict[int, List[str]] = {}
    for t in topics_list:
        try:
            tid = int(t.get("topic_id"))
        except Exception:
            continue
        kws = t.get("keywords", []) or []
        topic_kw_map[tid] = list({k for k in kws if k})

    def builder(text: str) -> TopicHints:
        if not topics_doc or (not centroids and not keywords_global):
            return TopicHints()

        tv = embedder.encode([text])[0]  # vector chunk
        chunk_tokens = _normalize_tokens(text)

        # Afinidad semántica con centroides
        affinities_sem: List[Tuple[int, float]] = []
        for tid, cv in centroids.items():
            sim = _Embedder.cos_sim(tv, cv)
            affinities_sem.append((tid, float(sim)))

        # Afinidad léxica (Jaccard tokens chunk vs keywords del topic)
        affinities_lex: Dict[int, float] = {}
        for tid, kws in topic_kw_map.items():
            affinities_lex[tid] = _jaccard(chunk_tokens, [k.lower() for k in kws])

        # Combinación ponderada (si falta una señal, la otra domina)
        combined: List[Tuple[int, float]] = []
        topic_ids_all = set([tid for tid, _ in affinities_sem]) | set(affinities_lex.keys())
        for tid in topic_ids_all:
            s_sem = next((v for t, v in affinities_sem if t == tid), 0.0)
            s_lex = affinities_lex.get(tid, 0.0)
            sc = _AFF_ALPHA_SEM * s_sem + _AFF_BETA_LEX * s_lex
            combined.append((tid, float(sc)))

        combined.sort(key=lambda x: x[1], reverse=True)
        topk = combined[: max(1, cfg.topic_affinity_topk)]
        topic_ids = [t for t, _ in topk]
        scores = {str(t): float(s) for t, s in topk}

        # Herencia de keywords: globales + del best topic (si existe)
        inherited_kws = list({k for k in keywords_global[:10] if k})
        if topk:
            best_tid = topk[0][0]
            for k in (topic_kw_map.get(best_tid, [])[:10]):
                if k and k not in inherited_kws:
                    inherited_kws.append(k)

        return TopicHints(
            inherited_topic_ids=topic_ids,
            inherited_keywords=inherited_kws[:15],  # límite prudente
            topic_affinity=scores
        )

    return builder


def _compute_doc_centroid(blocks: List[Tuple[int, int, Dict[str, Any]]], embedder: _Embedder) -> Optional[np.ndarray]:
    """
    Centroide del documento a partir de embeddings de bloques (robusto y barato).
    Si no hay bloques, devuelve None.
    """
    if not blocks:
        return None
    texts = [(b or {}).get("text", "") for _, _, b in blocks if (b or {}).get("text", "")]
    if not texts:
        return None
    vecs = embedder.encode(texts)
    if vecs.size == 0:
        return None
    return vecs.mean(axis=0)


def _compute_scores(chunks: List[Chunk], embedder: _Embedder, doc_centroid: Optional[np.ndarray]) -> None:
    """Anota métricas por chunk: cohesión vs doc-centroid y redundancia inter-chunk (+ normalizada)."""
    if not chunks:
        return
    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts)

    # Cohesión: similitud chunk ↔ centroide del documento (si existe)
    for i, c in enumerate(chunks):
        if doc_centroid is not None:
            coh = _Embedder.cos_sim(vecs[i], doc_centroid)
        else:
            # fallback: comparar contra media de chunks (mantiene comportamiento anterior)
            mean_vec = vecs.mean(axis=0)
            coh = _Embedder.cos_sim(vecs[i], mean_vec)
        c.scores["cohesion_vs_doc"] = float(coh)

    # Redundancia inter-chunk (vecino más similar) y versión normalizada
    avg_len = float(np.mean([max(1, ch.char_len) for ch in chunks]))
    for i in range(len(chunks)):
        sims = []
        for j in range(len(chunks)):
            if i == j:
                continue
            sims.append(_Embedder.cos_sim(vecs[i], vecs[j]))
        cmax = max(sims) if sims else 0.0
        chunks[i].scores["max_redundancy"] = float(cmax)
        # normalización por tamaño relativo del chunk (heurística contra duplicidad "barata")
        rel = chunks[i].char_len / avg_len if avg_len > 0 else 1.0
        chunks[i].scores["redundancy_norm"] = float(cmax * rel)


def run(
    ir_with_topics: Dict[str, Any],
    cfg: Optional[ChunkingConfig] = None
) -> DocumentChunks:
    """
    Punto de entrada principal del HybridChunker.
    """
    cfg = cfg or ChunkingConfig()
    np.random.seed(cfg.seed)

    doc_id = ir_with_topics.get("doc_id") or _hash_id(json.dumps(ir_with_topics)[:256])
    source_path = ir_with_topics.get("source_path")
    mime = ir_with_topics.get("mime")
    lang = ir_with_topics.get("lang")

    # Puede venir en ir.meta.topics_doc o al nivel raíz (compat)
    topics_doc = ((ir_with_topics.get("meta") or {}).get("topics_doc")) or (ir_with_topics.get("topics_doc"))

    # 1) Extraer bloques textuales
    blocks = _iter_blocks(ir_with_topics)
    if not blocks:
        logger.warning(f"[chunker] Documento sin bloques textuales: {doc_id}")

    # 1.1) Detección de idioma (solo si no viene o es 'und')
    detected_lang = None
    if (not lang) or (isinstance(lang, str) and lang.lower() in {"", "und", "xx"}):
        detected_lang = _detect_lang_heuristic(ir_with_topics, None)
        if detected_lang:
            lang = detected_lang

    # 2) Selección/carga spaCy según idioma (con fallback)
    chosen_spacy_model = _choose_spacy_model(lang, cfg.spacy_model)
    nlp = _load_spacy(chosen_spacy_model)
    if nlp is None and chosen_spacy_model != cfg.spacy_model:
        # último intento: el definido en config
        nlp = _load_spacy(cfg.spacy_model)

    # 3) Pre-segmentación por headings/listas
    sections = _presegment_blocks(blocks, cfg)

    # 4) Preparar herramientas semánticas
    embedder = _Embedder(cfg.embedding_model, cfg.embedding_batch_size)
    topic_hints_builder = _build_topic_hints_fn(cfg, topics_doc, embedder)

    # 4.1) Centroide del documento (para cohesión_vs_doc)
    doc_centroid = _compute_doc_centroid(blocks, embedder)

    # 5) Por sección: dividir en oraciones y empaquetar
    all_chunks: List[Chunk] = []
    for sec_blocks, _starts_with_heading in sections:
        sentences_with_spans: List[Tuple[str, int, int]] = []
        for pno, bidx, blk in sec_blocks:
            sents = _sentence_split(blk.get("text", ""), nlp)
            for s in sents:
                if len(s.strip()) < 3:
                    continue
                sentences_with_spans.append((s, pno, bidx))

        sec_chunks = _pack_sentences_to_chunks(
            doc_id=doc_id,
            sentences_with_spans=sentences_with_spans,
            cfg=cfg,
            topic_hints_builder=topic_hints_builder
        )
        # Ajuste de orden global: reasignamos para mantener monotonicidad absoluta
        for ch in sec_chunks:
            ch.order = len(all_chunks)
            ch.chunk_id = f"{doc_id[:8]}_{ch.order:04d}_{_hash_id(ch.text)}"
        all_chunks.extend(sec_chunks)

    # 6) Scoring por embeddings (cohesión y redundancia)
    if cfg.use_embeddings and len(all_chunks) >= 1:
        _compute_scores(all_chunks, embedder, doc_centroid)

    # 7) Construir salida
    meta_cfg = cfg.model_dump()
    # Añadimos huellas informativas, sin modificar el contrato (solo meta/config)
    meta_cfg["detected_lang"] = detected_lang or ""
    meta_cfg["used_spacy_model"] = chosen_spacy_model

    meta = ChunkingMeta(
        config=meta_cfg,
        topics_doc=topics_doc,
        stats={
            "n_sections": len(sections),
            "n_blocks": len(blocks),
            "n_chunks": len(all_chunks),
            "avg_chunk_chars": float(np.mean([c.char_len for c in all_chunks])) if all_chunks else 0.0,
            "avg_chunk_tokens": float(np.mean([c.est_tokens for c in all_chunks])) if all_chunks else 0.0,
        }
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


# ====== IO helpers (alineados al estilo de tus otros subsistemas) ======
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
