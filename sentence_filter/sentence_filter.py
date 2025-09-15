# sentence_filter/sentence_filter.py
# -*- coding: utf-8 -*-
"""
Sentence/Filter — Divide chunks en oraciones y filtra ruido antes de IE.

Objetivos
---------
- Reducir costo de NER/RE evitando oraciones poco informativas.
- Entregar oraciones con offsets y trazabilidad a chunks (page_span, chunk_id).

Estrategia
----------
1) Normaliza texto de cada chunk (espacios, bullets, guiones partidos).
2) Divide en oraciones (spaCy si disponible; fallback regex robusto).
3) Aplica filtros configurables:
   - min_chars: descarta oraciones muy cortas
   - drop_stopword_only: descarta si son solo stopwords (spaCy o listas básicas)
   - drop_numeric_only: descarta si no hay letras (solo dígitos/puntuación)
   - dedupe: exact | fuzzy (SequenceMatcher) para remover near-duplicados
4) Emite DocumentSentences con métricas de keep/drop en meta.

Notas de implementación
-----------------------
- Si se usa spaCy y el pipeline cargado no trae 'parser' ni 'senter',
  se inyecta 'sentencizer' automáticamente para habilitar doc.sents.
- Si no hay spaCy (o falla), se cae a un regex robusto (_SENT_SPLIT_REGEX).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Contratos previos (chunks generados por el HybridChunker)
from parser.schemas import DocumentChunks
# Contratos de esta etapa (salida de oraciones)
from .schemas import DocumentSentences, SentenceIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuración de split/normalización
# ---------------------------------------------------------------------------

# Regex de split por oración (fallback): punto/exclamación/interrogación/… + espacio + mayúscula/dígito/signos
_SENT_SPLIT_REGEX = re.compile(r"(?<=[\.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡0-9])")

# Caracteres de bullets comunes al inicio de línea
_BULLETS = tuple("•·◦▪-–—*·")

# Detección de letras (incluye español)
_ALPHA_RX = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")

# Stopwords simples (fallback si no hay spaCy)
_STOP_ES = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no","una",
    "su","al","lo","como","más","pero","sus","le","ya","o","fue","ha","sí","porque","esta","entre",
    "cuando","muy","sin","sobre","también","me","hasta","hay","donde","quien","desde","todo","nos",
    "durante","todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto","mí",
    "antes","algunos","qué","unos","yo","otro","otras","otra","él","tanto","esa","estos","mucho",
    "quienes","nada","muchos","cual","poco","ella","estar","estas","algunas","algo","nosotros",
    "mi","mis","tú","te","ti","tu","tus","ellas","nosotras","vosotros","vosotras","os","mío","mía",
    "míos","mías","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos","suyas","nuestro","nuestra",
    "nuestros","nuestras","vuestro","vuestra","vuestros","vuestras","esos","esas","estoy","estás",
    "está","estamos","estáis","están","esté","estés","estemos","estéis","estén"
}
_STOP_EN = {
    "the","of","and","to","in","a","is","that","it","for","on","was","as","with","by","at","from",
    "or","an","be","this","which","are"
}


# ---------------------------------------------------------------------------
# Config y clase principal
# ---------------------------------------------------------------------------

@dataclass
class SentenceFilterConfig:
    # Splitter
    sentence_splitter: str = "auto"  # "spacy" | "regex" | "auto"
    # Normalización
    normalize_whitespace: bool = True
    dehyphenate: bool = True           # une "infor-\nmación" → "información"
    strip_bullets: bool = True         # quita bullets al inicio de la línea/oración
    # Filtros
    min_chars: int = 25                # descarta oraciones demasiado cortas
    drop_stopword_only: bool = True    # descarta si solo contiene stopwords
    drop_numeric_only: bool = True     # descarta si no tiene ninguna letra
    # Dedupe
    dedupe: str = "fuzzy"              # "none" | "exact" | "fuzzy"
    fuzzy_threshold: float = 0.92      # similitud mínima para considerar duplicado


class SentenceFilter:
    """
    Implementación determinística con spaCy opcional.
    - Usa spaCy para sentencias si hay modelo disponible.
    - Si no, cae a regex (_SENT_SPLIT_REGEX).
    """

    def __init__(self, config: Optional[SentenceFilterConfig] = None):
        self.cfg = config or SentenceFilterConfig()
        self._nlp = None

        # Si el usuario pide 'spacy' o 'auto', intentamos cargar un modelo
        if self.cfg.sentence_splitter in ("spacy", "auto"):
            try:
                import spacy
                for model in ("es_core_news_sm", "en_core_web_sm"):
                    try:
                        # Deshabilitamos componentes costosos para velocidad;
                        # NO deshabilitamos 'senter' si el modelo lo trae.
                        self._nlp = spacy.load(
                            model,
                            disable=[
                                "tagger", "ner", "lemmatizer", "attribute_ruler",
                                "tok2vec", "morphologizer", "parser"  # parser fuera por performance
                            ]
                        )
                        break
                    except Exception:
                        continue

                # Garantizar límites de oración: si no hay 'parser' ni 'senter', añadimos 'sentencizer'
                if self._nlp is not None:
                    pipe_names = set(self._nlp.pipe_names)
                    if ("parser" not in pipe_names) and ("senter" not in pipe_names):
                        try:
                            self._nlp.add_pipe("sentencizer")
                            logger.info("spaCy: 'sentencizer' agregado para habilitar doc.sents.")
                        except Exception as e:
                            logger.warning("No se pudo agregar 'sentencizer' (%s). Se usará regex.", e)
                            self._nlp = None
            except ImportError:
                # spaCy no está instalado → fallback regex
                self._nlp = None

        # Si el usuario eligió explícitamente 'spacy' pero no se pudo cargar, log de aviso
        if self.cfg.sentence_splitter == "spacy" and self._nlp is None:
            logger.warning("sentence_splitter='spacy' solicitado, pero no se pudo cargar spaCy. Usando regex.")

    # -----------------------------------------------------------------------
    # API principal
    # -----------------------------------------------------------------------

    def sentences_from_chunks(self, dc: DocumentChunks) -> DocumentSentences:
        """
        Genera oraciones a partir de DocumentChunks y aplica filtros.
        Devuelve DocumentSentences con métricas de keep/drop.

        - Mantiene trazabilidad al chunk origen (chunk_id, page_span, offsets).
        - Deduplicación exacta o fuzzy para abaratar etapas siguientes.
        """
        cfg = self.cfg
        sentences: List[SentenceIR] = []

        # Contadores de filtro (para métricas)
        total_split = 0
        dropped_short = 0
        dropped_stop = 0
        dropped_numeric = 0
        dropped_dupe = 0

        # Estructuras para dedupe
        seen_exact: set[str] = set()
        seen_buckets: List[str] = []  # para fuzzy: guardamos oraciones normalizadas

        for cidx, ch in enumerate(dc.chunks):
            text = ch.text or ""

            # 1) Normalización previa del texto del chunk
            text = self._normalize(text)

            # 2) Split a oraciones + offsets
            sents, offsets = self._split_with_offsets(text)

            # 3) Filtros y dedupe
            for sidx, sent in enumerate(sents):
                total_split += 1
                s = sent.strip()
                if not s:
                    continue

                # a) Filtros básicos
                if cfg.min_chars and len(s) < cfg.min_chars:
                    dropped_short += 1
                    continue
                if cfg.drop_numeric_only and not _ALPHA_RX.search(s):
                    dropped_numeric += 1
                    continue
                if cfg.drop_stopword_only and self._is_stopword_only(s):
                    dropped_stop += 1
                    continue

                # b) Dedupe (exact o fuzzy)
                norm = self._norm_for_dedupe(s)
                if cfg.dedupe == "exact":
                    if norm in seen_exact:
                        dropped_dupe += 1
                        continue
                    seen_exact.add(norm)
                elif cfg.dedupe == "fuzzy":
                    if self._is_near_duplicate(norm, seen_buckets, cfg.fuzzy_threshold):
                        dropped_dupe += 1
                        continue
                    seen_buckets.append(norm)

                # 4) Emitir oración con trazabilidad
                st, en = offsets[sidx]
                sid = SentenceIR.new_id(dc.doc_id, cidx, sidx)
                sentences.append(
                    SentenceIR(
                        id=sid,
                        text=s,
                        meta={
                            "chunk_id": ch.id,
                            "chunk_idx": cidx,
                            "page_span": ch.meta.get("page_span"),
                            "char_span_in_chunk": (int(st), int(en)),
                            "filters": {
                                "normalized": True,
                                "dedupe": cfg.dedupe,
                            },
                        },
                    )
                )

        kept = len(sentences)
        meta = {
            "source": "chunks",
            "params": {
                "sentence_splitter": (
                    ("spacy" if self._nlp else "regex")
                    if self.cfg.sentence_splitter == "auto"
                    else self.cfg.sentence_splitter
                ),
                "min_chars": cfg.min_chars,
                "drop_stopword_only": cfg.drop_stopword_only,
                "drop_numeric_only": cfg.drop_numeric_only,
                "dedupe": cfg.dedupe,
                "fuzzy_threshold": cfg.fuzzy_threshold,
                "strip_bullets": cfg.strip_bullets,
                "dehyphenate": cfg.dehyphenate,
                "normalize_whitespace": cfg.normalize_whitespace,
            },
            "counters": {
                "total_split": total_split,
                "kept": kept,
                "dropped_short": dropped_short,
                "dropped_stopword": dropped_stop,
                "dropped_numeric": dropped_numeric,
                "dropped_dupe": dropped_dupe,
                "keep_rate": (kept / total_split) if total_split else 0.0,
            },
        }
        return DocumentSentences(
            doc_id=dc.doc_id,
            strategy="sentences-v1",
            version="1.0",
            sentences=sentences,
            meta=meta,
        )

    # -----------------------------------------------------------------------
    # Utilidades internas
    # -----------------------------------------------------------------------

    def _normalize(self, s: str) -> str:
        """
        Normaliza texto antes del split:
        - Une palabras cortadas por guion al salto de línea ("infor-\\nmación" → "información").
        - Quita bullets/viñetas al inicio de cada línea.
        - Colapsa espacios y limpia bordes.
        """
        if not s:
            return s

        # 1) Dehyphenate
        if self.cfg.dehyphenate:
            s = s.replace("-\n", "")

        # 2) Quitar bullets al inicio de línea
        if self.cfg.strip_bullets:
            lines = []
            for line in s.splitlines():
                ls = line.lstrip()
                # p.ej. "• foo", "- bar", "* baz", "— item"
                if ls and ls[0] in _BULLETS:
                    ls = ls[1:].lstrip()
                lines.append(ls)
            s = "\n".join(lines)

        # 3) Normalizar espacios por línea
        if self.cfg.normalize_whitespace:
            s = "\n".join(" ".join(line.split()) for line in s.splitlines())

        return s.strip()

    def _split_with_offsets(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Divide 'text' en oraciones y retorna offsets (start,end) relativos al texto.

        - Con spaCy (preferido): usa doc.sents y offsets nativos (start_char, end_char).
        - Fallback regex: divide por signos de puntuación y reconstruye offsets manualmente.
        """
        if not text:
            return [], []

        # Camino spaCy si hay pipeline válido y el usuario lo pidió (o 'auto')
        if self._nlp is not None and self.cfg.sentence_splitter in ("spacy", "auto"):
            doc = self._nlp(text)
            # IMPORTANTE: aquí doc.sents está garantizado (por 'senter' del modelo o 'sentencizer' inyectado)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            offsets = [(s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
            return sents, offsets

        # Fallback regex (robusto para español/inglés con patrones comunes)
        parts = _SENT_SPLIT_REGEX.split(text)
        if len(parts) <= 1:
            # No hay separadores claros → una sola oración
            return [text.strip()], [(0, len(text))]

        sents: List[str] = []
        offsets: List[Tuple[int, int]] = []
        idx = 0
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Encontrar el substring desde idx para armar offsets consistentes
            start = text.find(p, idx)
            if start < 0:
                start = idx
            end = start + len(p)
            sents.append(p)
            offsets.append((start, end))
            idx = end

        return sents, offsets

    def _is_stopword_only(self, text: str) -> bool:
        """
        True si todos los tokens alfabéticos son stopwords o no hay tokens válidos.
        Usa stopwords de spaCy si está disponible; si no, listas ES/EN básicas.
        """
        toks = [t.lower() for t in re.findall(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ']+\b", text)]
        if not toks:
            # Si no hay tokens alfabéticos, tratamos como ruido (p. ej., '---', '1234')
            return True

        if self._nlp is not None:
            sw = self._nlp.Defaults.stop_words
            return all(t in sw for t in toks)

        # Fallback mixto ES/EN
        return all((t in _STOP_ES) or (t in _STOP_EN) for t in toks)

    def _norm_for_dedupe(self, s: str) -> str:
        """Normaliza una oración para dedupe (minúsculas + espacios colapsados)."""
        return " ".join(s.lower().split())

    def _is_near_duplicate(self, norm: str, seen: List[str], thr: float) -> bool:
        """
        Compara contra oraciones vistas; si similitud >= thr (SequenceMatcher), lo considera duplicado.
        Pequeñas cadenas (<8) se consideran duplicados solo si son exactamente iguales.
        """
        for prev in seen:
            if len(prev) < 8 and prev == norm:
                return True
            r = SequenceMatcher(None, prev, norm).ratio()
            if r >= thr:
                return True
        return False
