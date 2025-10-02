# -*- coding: utf-8 -*-
"""
chunker.py — HybridChunker: regla + semántica + límites de longitud

Estrategia híbrida:
1) Pre-segmentación por estructura (headings/listas/bloques).
2) Refinamiento por oraciones (spaCy) para no cortar ideas.
3) Empaquetado greedy por longitud (≤ max_tokens aprox.) con control de min_chars.
4) Cálculo opcional de embeddings para:
   - cohesión intra-chunk (similitud promedio)
   - redundancia inter-chunk
   - afinidad con topics_doc (por keywords → centroides)

Robustez:
- Si embeddings no disponibles, fallback a TF-IDF/Hashing.
- Si spaCy no disponible, fallback por puntos/guiones.
- Siempre transpone `topics_doc` del IR de entrada a meta de salida.
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
        if self.kind == "sbert":
            return np.array(self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False))
        if self.kind == "hash":
            mat = self.model.transform(texts)  # sparse
            return mat.toarray().astype(np.float32)
        # Sin embeddings: devolvemos promedio de longitud como vector dummy
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
    order = 0

    def flush():
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
                scores={}
            )
        )
        order += 1
        buf_texts, buf_spans, buf_chars = [], [], 0

    for sent, pno, bidx in sentences_with_spans:
        sent = sent.strip()
        if not sent:
            continue
        est = _estimate_tokens(sent)
        # Si la oración sola excede max_chars, la forzamos (caso extremo)
        if len(sent) > cfg.max_chars:
            if buf_texts:
                flush()
            buf_texts = [sent]
            buf_spans = [ChunkSourceSpan(page_number=pno, block_indices=[bidx])]
            flush()
            continue

        # ¿Cabe en el buffer actual?
        if (buf_chars + len(sent) > cfg.max_chars) or (_estimate_tokens(" ".join(buf_texts + [sent])) > cfg.max_tokens):
            # Si el buffer actual es muy corto, igual flush para no concatenar
            if buf_chars >= cfg.min_chars:
                flush()
            else:
                # Forzamos flush suave (demasiado corta pero no cabe la nueva)
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
    centroids = _topic_keyword_centroids(topics_doc, embedder)

    def builder(text: str) -> TopicHints:
        if not topics_doc or not centroids:
            return TopicHints()
        tv = embedder.encode([text])[0]
        affinities: List[Tuple[int, float]] = []
        for tid, cv in centroids.items():
            sim = _Embedder.cos_sim(tv, cv)
            affinities.append((tid, sim))
        affinities.sort(key=lambda x: x[1], reverse=True)
        topk = affinities[: max(1, cfg.topic_affinity_topk)]
        topic_ids = [t for t, _ in topk]
        scores = {str(t): float(s) for t, s in topk}
        inherited_kws = (topics_doc.get("keywords_global") or [])[:10]
        return TopicHints(
            inherited_topic_ids=topic_ids,
            inherited_keywords=inherited_kws,
            topic_affinity=scores
        )

    return builder


def _compute_scores(chunks: List[Chunk], embedder: _Embedder) -> None:
    """Anota métricas ligeras por chunk: cohesión y redundancia."""
    if not chunks:
        return
    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts)

    # Cohesión: similitud de cada oración interna vs promedio (aprox por troceo simple)
    # Para eficiencia, usamos el propio vector del chunk comparado con medias globales.
    mean_vec = vecs.mean(axis=0)
    for i, c in enumerate(chunks):
        # similitud del chunk vs media global (proxy de coherencia contextual del doc)
        coh = _Embedder.cos_sim(vecs[i], mean_vec)
        c.scores["cohesion_vs_doc"] = float(coh)

    # Redundancia inter-chunk (vecino más similar)
    for i in range(len(chunks)):
        sims = []
        for j in range(len(chunks)):
            if i == j:
                continue
            sims.append(_Embedder.cos_sim(vecs[i], vecs[j]))
        cmax = max(sims) if sims else 0.0
        chunks[i].scores["max_redundancy"] = float(cmax)


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

    topics_doc = ((ir_with_topics.get("meta") or {}).get("topics_doc")) or (ir_with_topics.get("topics_doc"))

    # 1) Extraer bloques textuales
    blocks = _iter_blocks(ir_with_topics)
    if not blocks:
        logger.warning(f"[chunker] Documento sin bloques textuales: {doc_id}")

    # 2) Pre-segmentación por headings/listas
    sections = _presegment_blocks(blocks, cfg)

    # 3) Preparar herramientas
    nlp = _load_spacy(cfg.spacy_model)
    embedder = _Embedder(cfg.embedding_model, cfg.embedding_batch_size)
    topic_hints_builder = _build_topic_hints_fn(cfg, topics_doc, embedder)

    # 4) Por sección: dividir en oraciones y empaquetar
    all_chunks: List[Chunk] = []
    for sec_blocks, starts_with_heading in sections:
        # concatenamos texto dentro de la sección, pero conservamos spans
        sentences_with_spans: List[Tuple[str, int, int]] = []
        for pno, bidx, blk in sec_blocks:
            sents = _sentence_split(blk.get("text", ""), nlp)
            for s in sents:
                # filtrado suave de basura
                if len(s.strip()) < 3:
                    continue
                sentences_with_spans.append((s, pno, bidx))

        sec_chunks = _pack_sentences_to_chunks(
            doc_id=doc_id,
            sentences_with_spans=sentences_with_spans,
            cfg=cfg,
            topic_hints_builder=topic_hints_builder
        )
        all_chunks.extend(sec_chunks)

    # 5) Scoring opcional por embeddings (cohesión y redundancia)
    if cfg.use_embeddings and len(all_chunks) >= 2:
        _compute_scores(all_chunks, embedder)

    # 6) Construir salida
    meta = ChunkingMeta(
        config=cfg.model_dump(),
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
            chunks.model_dump(mode="json"),  # 👈 este cambio
            f,
            ensure_ascii=False,
            indent=2
        )
    logger.info(f"[chunker] Guardado: {p}")
    return p
