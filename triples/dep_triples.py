# triples/dep_triples.py
# -*- coding: utf-8 -*-
"""
Extractor de triples (S,R,O) ligeros:
  - Árbol de dependencias con spaCy (si está disponible) en ES/EN simultáneo.
  - Reglas heurísticas + regex como fallback.
  - Filtros de ruido (contacto, headings, números).
  - Relaciones canónicas (opcional) con superficie original en meta.

Diseño:
  - Determinístico, eficiente y extensible.
  - Batch por idioma con nlp.pipe.
  - Trazabilidad: copia meta de SentenceIR.meta (chunk_id, page_span, etc.).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Mapping, DefaultDict
from collections import defaultdict

from pydantic import BaseModel

# Contratos de entrada/salida
try:
    from sentence_filter.schemas import DocumentSentences  # type: ignore
except Exception:
    class DocumentSentences(BaseModel):  # type: ignore
        doc_id: str
        sentences: List[Dict[str, Any]]
        meta: Dict[str, Any] = {}

from .schemas import DocumentTriples, TripleIR

logger = logging.getLogger(__name__)


# ----------------------------
# Configuración del extractor
# ----------------------------

@dataclass
class DepTripleConfig:
    """Configuración de extracción de triples."""
    use_spacy: Optional[bool] = None  # None="auto", True="force", False="off"
    lang_pref: str = "auto"           # auto | es | en
    ruleset: str = "default-bilingual"  # placeholder
    max_triples_per_sentence: int = 4
    drop_pronoun_subjects: bool = True
    min_token_chars: int = 2
    # Normalización
    strip_quotes: bool = True
    collapse_spaces: bool = True
    # spaCy
    spacy_disable: Tuple[str, ...] = ("ner", "lemmatizer", "textcat")
    enable_ner: bool = False  # si True y el modelo lo soporta, no lo deshabilita
    # Spans
    max_span_chars: int = 300
    # Canonicalización de relaciones (es→en)
    canonicalize_relations: bool = True
    min_conf_keep: Optional[float] = None  # descarta lo más débil (ej. regex_prep 0.60)

# ----------------------------
# Utilidades y filtros
# ----------------------------

_WS_RE = re.compile(r"\s+", re.UNICODE)
_URL_RE = re.compile(r"(https?://|www\.)", re.I)
_EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{7,}")  # >=7 dígitos con separadores
_CP_RE = re.compile(r"\bC\.?P\.?\s*\d{4,6}\b", re.I)
_HEADING_PREFIX = re.compile(r"^\s*(title|author|membership|interests|skills)\b[:\-–]?\s*", re.I)


TITLE_LEXICON = {
    # ES (lemmas/strings comunes)
    "director", "directora", "gerente", "jefe", "jefa", "oficial", "analista", "ingeniero",
    "ingeniera", "científico", "cientifica", "científica", "consultor", "consultora",
    "vicepresidente", "vicepresidenta", "presidente", "presidenta", "coordinador",
    "coordinadora", "lider", "líder", "responsable", "arquitecto", "arquitecta",
    # EN
    "chief", "officer", "cto", "ceo", "coo", "cdo", "cfo", "vp", "director", "manager",
    "lead", "principal", "engineer", "scientist", "analyst", "consultant", "architect",
    "head", "owner",
}

ACQ_LEMMAS_EN = {"acquire", "buy", "purchase", "merge"}
ACQ_LEMMAS_ES = {"adquirir", "comprar", "fusionar"}
WORK_LEMMAS_EN = {"work", "serve", "lead", "head"}  # "is ... at" se maneja aparte
WORK_LEMMAS_ES = {"trabajar", "liderar", "encabezar", "dirigir"}

def _norm(s: str, cfg: DepTripleConfig) -> str:
    if s is None:
        return ""
    t = s.strip()
    if cfg.strip_quotes:
        t = t.strip("\"'“”‘’`´")
    if cfg.collapse_spaces:
        t = _WS_RE.sub(" ", t)
    return t

def _strip_punct_edges(s: str) -> str:
    return s.lstrip(",;: ").rstrip(",;: ")

def _short_sha1(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def _fold(s: str) -> str:
    """Lower + remove accents."""
    s = s.casefold()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _guess_lang(text: str) -> str:
    t = f" {text.lower()} "
    es_hits = sum(k in t for k in (" el ", " la ", " los ", " las ", " de ", " para ", " es ", " fue ", " en "))
    en_hits = sum(k in t for k in (" the ", " of ", " for ", " is ", " was ", " in ", " at "))
    if es_hits > en_hits and es_hits >= 1:
        return "es"
    if en_hits > es_hits and en_hits >= 1:
        return "en"
    return "und"

def _has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)

def _digits_ratio(s: str) -> float:
    d = sum(1 for ch in s if ch.isdigit())
    return d / max(1, len(s))

def _mostly_non_letters(s: str, thresh: float = 0.40) -> bool:
    non = sum(1 for ch in s if not (ch.isalpha() or ch.isspace()))
    return (non / max(1, len(s))) >= thresh

def _is_contact_like(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _URL_RE.search(t) or _EMAIL_RE.search(t) or _PHONE_RE.search(t) or _CP_RE.search(t):
        return True
    # Encabezados tipo markdown/section
    if t.lstrip().startswith("#"):
        return True
    if _HEADING_PREFIX.match(t):
        return True
    # Si no hay verbos/copulares y hay mucha puntuación/números, también saltar
    lower = f" {t.lower()} "
    has_verbish = any(k in lower for k in (" es ", " son ", " fue ", " eran ", " is ", " are ", " was ", " were "))
    if not has_verbish and _mostly_non_letters(t, 0.35):
        return True
    # Demasiados dígitos
    if _digits_ratio(t) > 0.35:
        return True
    return False

def _canonical_relation(surface: str, lang: str, cfg: DepTripleConfig) -> Tuple[str, str]:
    """
    Devuelve (canonical, surface). Si canonicalize_relations=False, retorna (surface, surface).
    """
    s = surface.strip().lower().replace(" ", "_")
    if not cfg.canonicalize_relations:
        return (s, surface)

    # Normalizaciones básicas bilingües
    mapping = {
        # copulares
        "es": "is", "son": "is", "fue": "is", "eran": "is", "será": "is", "ha_sido": "is",
        "is": "is", "are": "is", "was": "is", "were": "is", "has_been": "is",
        # trabajo
        "trabajar_en": "works_at", "trabaja_en": "works_at", "work_at": "works_at", "works_at": "works_at",
        "serve_at": "works_at", "serves_at": "works_at", "head_at": "works_at",
        # adquisición/compra/fusión
        "adquirir": "acquired", "adquirió": "acquired", "compra": "acquired", "compró": "acquired",
        "acquire": "acquired", "acquired": "acquired", "buy": "acquired", "bought": "acquired",
        "purchase": "acquired", "purchased": "acquired", "merge": "acquired", "merged": "acquired",
        # ubicación
        "en": "in", "in": "in", "ubicado_en": "located_in", "located_in": "located_in",
        # genéricas
        "de": "of", "of": "of", "para": "for", "for": "for", "con": "with", "with": "with",
        # alias
        "alias": "alias",
        # relaciones comunes
        "miembro_de": "member_of", "member_of": "member_of",
        "colaboración_con": "collaboration_with", "collaboration_with": "collaboration_with",
        "autor_de": "author_of", "author_of": "author_of",
    }
    canonical = mapping.get(s, s)
    return (canonical, surface)


# ----------------------------
# spaCy (opcional, bilingüe)
# ----------------------------

class _SpacyHandle:
    """Carga perezosa de spaCy con bilingüe opcional."""
    def __init__(self, cfg: DepTripleConfig) -> None:
        self.cfg = cfg
        self._nlp_map: Dict[str, Any] = {}  # {"es": nlp, "en": nlp}
        self.available = False

    def ensure(self) -> None:
        if self._nlp_map:
            return
        if self.cfg.use_spacy is False:
            self.available = False
            return
        try:
            import spacy  # type: ignore
        except Exception:
            self._nlp_map = {}
            self.available = False
            return

        disable = tuple(d for d in self.cfg.spacy_disable if not self.cfg.enable_ner)
        loaded_any = False

        # Orden de carga según preferencia
        order: List[str] = []
        if self.cfg.lang_pref in ("es", "auto"):
            order.append("es")
        if self.cfg.lang_pref in ("en", "auto"):
            order.append("en")
        if not order:
            order = ["es", "en"]

        for lang in order:
            pkg = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
            try:
                self._nlp_map[lang] = spacy.load(pkg, disable=disable)
                loaded_any = True
            except Exception:
                # si no está instalado, lo omitimos
                continue

        self.available = loaded_any

    def nlp_for(self, lang: str):
        """Retorna el pipeline para 'es' o 'en' si existe; si no, alguno disponible."""
        if lang in self._nlp_map:
            return self._nlp_map[lang]
        # fallback a cualquiera cargado
        return next(iter(self._nlp_map.values()), None)

    def has_lang(self, lang: str) -> bool:
        return lang in self._nlp_map


# ----------------------------
# Extractor principal
# ----------------------------

class DepTripleExtractor:
    """
    Extrae triples (S,R,O) por oración usando:
      A) Reglas sobre dependencias (spaCy) — bilingüe.
      B) Heurísticas regex (fallback).
    """
    def __init__(self, cfg: Optional[DepTripleConfig] = None) -> None:
        self.cfg = cfg or DepTripleConfig()
        self._spacy = _SpacyHandle(self.cfg)
        self._regex_patterns_es = self._build_regex_es()
        self._regex_patterns_en = self._build_regex_en()

    # -------- API pública --------

    def extract_document(self, ds: DocumentSentences) -> DocumentTriples:
        """
        Procesa un DocumentSentences y devuelve DocumentTriples.
        Soporta ds.sentences como lista de SentenceIR (Pydantic) o dicts.
        """
        doc_id = getattr(ds, "doc_id", None) or (
            ds.meta.get("doc_id") if isinstance(getattr(ds, "meta", {}), dict) else None
        ) or "DOC-UNKNOWN"

        sentences = getattr(ds, "sentences", [])
        total_sents = len(sentences)

        self._spacy.ensure()

        # 1) Pre-extraemos campos y agrupamos por idioma
        entries: List[Dict[str, Any]] = []
        groups: DefaultDict[str, List[int]] = defaultdict(list)  # lang -> indices
        for idx, sent in enumerate(sentences):
            text = _field(sent, ("text", "sentence", "content", "raw"))
            if not text or not str(text).strip():
                continue
            text = str(text)

            # Salta oraciones “contact-like”
            if _is_contact_like(text):
                continue

            sentence_id = _field(sent, ("id", "sentence_id", "sid")) or f"SENT-{idx}"

            # Idioma
            if self.cfg.lang_pref in ("es", "en"):
                lang = self.cfg.lang_pref
            else:
                lang_override = _field(sent, ("lang",))
                lang = lang_override if isinstance(lang_override, str) and lang_override in ("es", "en") else _guess_lang(text)

            # Meta trazable
            sent_meta = getattr(sent, "meta", None)
            if not isinstance(sent_meta, dict) and isinstance(sent, Mapping):
                sent_meta = sent.get("meta", {})
            sent_meta = sent_meta or {}
            extra_meta: Dict[str, Any] = {}
            for k in ("chunk_id", "chunk_idx", "page_span", "char_span_in_chunk", "filters"):
                if k in sent_meta:
                    extra_meta[k] = sent_meta[k]

            entries.append({
                "idx": idx,
                "sentence_id": str(sentence_id),
                "text": text,
                "lang": lang,
                "extra_meta": extra_meta,
            })
            groups[lang].append(len(entries) - 1)

        # 2) spaCy batch por idioma (si disponible)
        spacy_docs_by_entry: Dict[int, Any] = {}
        if self._spacy.available:
            for lang, entry_indices in groups.items():
                if not entry_indices:
                    continue
                nlp = self._spacy.nlp_for(lang)
                if nlp is None:
                    continue
                texts = [entries[i]["text"] for i in entry_indices]
                for eidx, doc in zip(entry_indices, nlp.pipe(texts, batch_size=64, n_process=1)):
                    spacy_docs_by_entry[eidx] = doc

        # 3) Extraemos
        triples: List[TripleIR] = []
        seen: set = set()
        used_sents = 0

        for eidx, ent in enumerate(entries):
            text = ent["text"]
            lang = ent["lang"]
            sentence_id = ent["sentence_id"]
            idx = ent["idx"]
            extra_meta = ent["extra_meta"]

            s_triples = self._extract_from_sentence(
                text=text,
                sentence_idx=idx,
                sentence_id=sentence_id,
                doc_id=doc_id,
                lang=lang,
                extra_meta=extra_meta,
                spacy_doc=spacy_docs_by_entry.get(eidx),
            )

            kept = 0
            for tr in s_triples:
                key = (_fold(tr.subject), _fold(tr.relation), _fold(tr.object))
                if key in seen:
                    continue
                seen.add(key)
                triples.append(tr)
                kept += 1

            if kept > 0:
                used_sents += 1

        # --- Filtrado por confianza (suave) ---
        if self.cfg.min_conf_keep is not None:
            triples = [t for t in triples if t.meta.get("conf", 0.0) >= self.cfg.min_conf_keep]

        # --- Recalcular used_sents tras el filtro ---
        kept_sent_idxs = {t.meta.get("sentence_idx") for t in triples}
        used_sents = len([i for i in kept_sent_idxs if i is not None])

            
            

        dt = DocumentTriples(
            doc_id=doc_id,
            triples=triples,
            meta={
                "params": {
                    "use_spacy": self._spacy.available,
                    "lang_pref": self.cfg.lang_pref,
                    "ruleset": self.cfg.ruleset,
                    "max_triples_per_sentence": self.cfg.max_triples_per_sentence,
                    "drop_pronoun_subjects": self.cfg.drop_pronoun_subjects,
                    "canonicalize_relations": self.cfg.canonicalize_relations,
                },
                "counters": {
                    "total_sents": total_sents,
                    "used_sents": used_sents,
                    "total_triples": len(triples),
                    "avg_triples_per_used_sent": round(len(triples) / max(1, used_sents), 3),
                },
            },
        )
        return dt

    # -------- Extracción por oración --------

    def _extract_from_sentence(
        self,
        text: str,
        sentence_idx: int,
        sentence_id: str,
        doc_id: str,
        lang: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        spacy_doc: Optional[Any] = None,
    ) -> List[TripleIR]:
        out: List[TripleIR] = []

        if self._spacy.available:
            out.extend(
                self._extract_with_spacy(text, sentence_idx, sentence_id, doc_id, lang, extra_meta, spacy_doc)
            )

        if (not self._spacy.available) or (len(out) == 0):
            out.extend(
                self._extract_with_regex(text, sentence_idx, sentence_id, doc_id, lang, extra_meta)
            )

        if len(out) > self.cfg.max_triples_per_sentence:
            out = out[: self.cfg.max_triples_per_sentence]
        return out

    # -------- Reglas con spaCy --------

    def _extract_with_spacy(
        self,
        text: str,
        sentence_idx: int,
        sentence_id: str,
        doc_id: str,
        lang: str,
        extra_meta: Optional[Dict[str, Any]],
        spacy_doc: Optional[Any],
    ) -> List[TripleIR]:
        nlp = self._spacy.nlp_for(lang)
        if nlp is None:
            return []
        doc = spacy_doc if spacy_doc is not None else nlp(text)

        triples: List[TripleIR] = []

        # A) SVO (nsubj -> VERB -> obj/dobj) y prep pobj
        for token in doc:
            if token.pos_ != "VERB":
                continue

            subjects = [c for c in token.children if c.dep_.startswith("nsubj")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "obj")]
            preps = [c for c in token.children if c.dep_ == "prep"]
            pobj_candidates = []
            for p in preps:
                pobj_candidates.extend([gc for gc in p.children if gc.dep_ == "pobj"])

            if not subjects:
                continue

            for sbj in subjects:
                if self.cfg.drop_pronoun_subjects and sbj.pos_ == "PRON":
                    continue
                sbj_text = _strip_punct_edges(_norm(doc[sbj.left_edge.i : sbj.right_edge.i + 1].text, self.cfg))

                # SVO directo
                for obj in objects:
                    obj_text = _strip_punct_edges(_norm(doc[obj.left_edge.i : obj.right_edge.i + 1].text, self.cfg))
                    if not self._valid_sro(sbj_text, token.lemma_, obj_text):
                        continue
                    rel_surface = _strip_punct_edges(_norm(token.lemma_, self.cfg))
                    rel_canon, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                    start = min(sbj.idx, token.idx, obj.idx)
                    end = max(sbj.idx + len(sbj.text), token.idx + len(token.text), obj.idx + len(obj.text))
                    triples.append(
                        self._make_triple(
                            doc_id, sentence_idx, sentence_id,
                            sbj_text, rel_canon, obj_text,
                            dep_rule="VERB_dobj_nsubj", conf=0.86, lang=lang,
                            span=(start, min(end, self.cfg.max_span_chars)),
                            extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                        )
                    )

                # VERB + prep + pobj → relación compuesta (work_at / located_in, etc.)
                for obj in pobj_candidates:
                    obj_text = _strip_punct_edges(_norm(doc[obj.left_edge.i : obj.right_edge.i + 1].text, self.cfg))
                    if not obj_text:
                        continue
                    prep = obj.head  # preposición
                    rel_surface = _strip_punct_edges(_norm(f"{token.lemma_}_{prep.lemma_}", self.cfg))
                    rel_canon, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                    if not self._valid_sro(sbj_text, rel_canon, obj_text):
                        continue
                    start = min(sbj.idx, token.idx, obj.idx)
                    end = max(sbj.idx + len(sbj.text), token.idx + len(token.text), obj.idx + len(obj.text))
                    triples.append(
                        self._make_triple(
                            doc_id, sentence_idx, sentence_id,
                            sbj_text, rel_canon, obj_text,
                            dep_rule="VERB_prep_pobj_nsubj", conf=0.8, lang=lang,
                            span=(start, min(end, self.cfg.max_span_chars)),
                            extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                        )
                    )

        # B) Copulares (nsubj -> cop -> attr)
        for token in doc:
            if token.dep_ in ("attr", "acomp"):
                copulas = [c for c in token.children if c.dep_ == "cop"]
                if not copulas:
                    continue
                head = token.head
                subjects = [c for c in head.children if c.dep_.startswith("nsubj")]
                for sbj in subjects:
                    if self.cfg.drop_pronoun_subjects and sbj.pos_ == "PRON":
                        continue
                    sbj_text = _strip_punct_edges(_norm(doc[sbj.left_edge.i : sbj.right_edge.i + 1].text, self.cfg))
                    obj_text = _strip_punct_edges(_norm(doc[token.left_edge.i : token.right_edge.i + 1].text, self.cfg))
                    if not self._valid_sro(sbj_text, "is", obj_text):
                        continue
                    cop = copulas[0]
                    rel_surface = "is" if lang == "en" else "es"
                    rel_canon, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                    start = min(sbj.idx, cop.idx, token.idx)
                    end = max(sbj.idx + len(sbj.text), cop.idx + len(cop.text), token.idx + len(token.text))
                    triples.append(
                        self._make_triple(
                            doc_id, sentence_idx, sentence_id,
                            sbj_text, rel_canon, obj_text,
                            dep_rule="nsubj_cop_attr", conf=0.82, lang=lang,
                            span=(start, min(end, self.cfg.max_span_chars)),
                            extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                        )
                    )

       # C) Nominal con preposición (head NOUN/PROPN -> prep -> pobj)
        valid_preps = {"de", "en", "para", "con"} if lang == "es" else {"of", "in", "for", "with"}
        for token in doc:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            preps = [c for c in token.children if c.dep_ == "prep" and c.lemma_ in valid_preps]
            for p in preps:
                objs = [c for c in p.children if c.dep_ == "pobj"]
                for obj in objs:
                    sbj_text = _strip_punct_edges(_norm(doc[token.left_edge.i : token.right_edge.i + 1].text, self.cfg))
                    obj_text = _strip_punct_edges(_norm(doc[obj.left_edge.i : obj.right_edge.i + 1].text, self.cfg))

                    head_lemma = token.lemma_.lower()
                    prep_lemma = p.lemma_.lower()

                    # Detecta nominales semánticos: miembro de, colaboración con, autor de
                    if (head_lemma in {"miembro","member"}) and (prep_lemma in {"de","of"}):
                        rel_surface = "member_of"
                    elif (head_lemma in {"colaboración","colaboracion","collaboration"}) and (prep_lemma in {"con","with"}):
                        rel_surface = "collaboration_with"
                    elif (head_lemma in {"autor","author"}) and (prep_lemma in {"de","of"}):
                        rel_surface = "author_of"
                    else:
                        # fallback a la preposición si no es uno de los semánticos
                        rel_surface = _strip_punct_edges(_norm(p.lemma_, self.cfg))

                    rel_canon, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                    if not self._valid_sro(sbj_text, rel_canon, obj_text):
                        continue

                    start = min(token.idx, p.idx, obj.idx)
                    end = max(token.idx + len(token.text), p.idx + len(p.text), obj.idx + len(obj.text))
                    triples.append(
                        self._make_triple(
                            doc_id, sentence_idx, sentence_id,
                            sbj_text, rel_canon, obj_text,
                            dep_rule="NOUN_prep_pobj", conf=0.70,  # puedes dejar 0.68 si prefieres
                            lang=lang,
                            span=(start, min(end, self.cfg.max_span_chars)),
                            extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                        )
                    )
        # D) Aposición endurecida (NOUN appos NOUN) → alias
        for token in doc:
            if token.dep_ == "appos" and token.head is not None:
                head = token.head
                if head.pos_ not in {"PROPN", "NOUN"} or token.pos_ not in {"PROPN", "NOUN"}:
                    continue
                sbj_text = _strip_punct_edges(_norm(doc[head.left_edge.i : head.right_edge.i + 1].text, self.cfg))
                obj_text = _strip_punct_edges(_norm(doc[token.left_edge.i : token.right_edge.i + 1].text, self.cfg))
                if not (_has_letter(sbj_text) and _has_letter(obj_text)):
                    continue
                if len(obj_text.split()) > 6 or len(sbj_text.split()) > 12:
                    continue
                between = doc[min(head.i, token.i) : max(head.i, token.i)]
                if "," not in between.text:
                    continue
                start = min(head.idx, token.idx)
                end = max(head.idx + len(head.text), token.idx + len(token.text))
                rel_canon, rel_surf = _canonical_relation("alias", lang, self.cfg)
                triples.append(
                    self._make_triple(
                        doc_id, sentence_idx, sentence_id,
                        sbj_text, rel_canon, obj_text,
                        dep_rule="NOUN_appos_NOUN", conf=0.62, lang=lang,
                        span=(start, min(end, self.cfg.max_span_chars)),
                        extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                    )
                )

        # E) Voz pasiva (adquisiciones): "Y fue adquirido por X" / "Y was acquired by X"
        for token in doc:
            if token.pos_ != "VERB":
                continue
            lemma = token.lemma_.lower()
            is_acq = (lemma in (ACQ_LEMMAS_ES if lang == "es" else ACQ_LEMMAS_EN))
            if not is_acq:
                continue

            # Sujeto pasivo
            subs = [c for c in token.children if c.dep_.startswith("nsubj")]
            if not subs:
                # spaCy EN tiene nsubjpass; ES puede marcar "nsubj:pass"
                subs = [c for c in token.children if "nsubj" in c.dep_]
            if not subs:
                continue

            # Agente "por/by" + pobj
            agents = []
            for c in token.children:
                if c.dep_ in ("agent", "prep"):
                    # EN: agent->pobj; ES: prep 'por' -> pobj
                    if c.dep_ == "agent" or c.lemma_ in ("por", "by"):
                        agents.extend([gc for gc in c.children if gc.dep_ == "pobj"])
            if not agents:
                continue

            for subj in subs:
                obj_text = _strip_punct_edges(_norm(doc[subj.left_edge.i : subj.right_edge.i + 1].text, self.cfg))
                for ag in agents:
                    sbj_text = _strip_punct_edges(_norm(doc[ag.left_edge.i : ag.right_edge.i + 1].text, self.cfg))
                    rel_surface = "acquired" if lang == "en" else "adquirió"
                    rel_canon, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                    if not self._valid_sro(sbj_text, rel_canon, obj_text):
                        continue
                    start = min(subj.idx, token.idx, ag.idx)
                    end = max(subj.idx + len(subj.text), token.idx + len(token.text), ag.idx + len(ag.text))
                    triples.append(
                        self._make_triple(
                            doc_id, sentence_idx, sentence_id,
                            sbj_text, rel_canon, obj_text,
                            dep_rule="PASSIVE_agent_pobj", conf=0.85, lang=lang,
                            span=(start, min(end, self.cfg.max_span_chars)),
                            extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                        )
                    )

        # F) Empleo/Cargo: "X es {cargo} en Y" / "X is {title} at Y" / "X works at Y"
        #   Heurística: attr/cop + prep; o VERB work/serve/head + prep at/en
        preps_work = {"en"} if lang == "es" else {"at", "in"}
        for token in doc:
            # Caso 1: copular con atributo "título" y prep a org
            if token.dep_ in ("attr", "acomp"):
                head = token.head
                subjects = [c for c in head.children if c.dep_.startswith("nsubj")]
                preps = [c for c in token.children if c.dep_ == "prep" and c.lemma_ in preps_work] + \
                        [c for c in head.children if c.dep_ == "prep" and c.lemma_ in preps_work]
                orgs = [gc for p in preps for gc in p.children if gc.dep_ == "pobj"]
                if subjects and orgs:
                    for sbj in subjects:
                        if self.cfg.drop_pronoun_subjects and sbj.pos_ == "PRON":
                            continue
                        title_text = _strip_punct_edges(_norm(doc[token.left_edge.i : token.right_edge.i + 1].text, self.cfg))
                        if not title_text:
                            continue
                        # ¿parece título?
                        if not any(w in TITLE_LEXICON for w in title_text.lower().split()):
                            continue
                        subj_text = _strip_punct_edges(_norm(doc[sbj.left_edge.i : sbj.right_edge.i + 1].text, self.cfg))
                        for org in orgs:
                            org_text = _strip_punct_edges(_norm(doc[org.left_edge.i : org.right_edge.i + 1].text, self.cfg))
                            # (X, title, cargo) + (X, works_at, Y)
                            rel_c1, surf_c1 = _canonical_relation("title", lang, self.cfg)
                            rel_c2, surf_c2 = _canonical_relation("works_at" if lang == "en" else "trabaja_en", lang, self.cfg)
                            if self._valid_sro(subj_text, rel_c1, title_text):
                                triples.append(
                                    self._make_triple(
                                        doc_id, sentence_idx, sentence_id,
                                        subj_text, rel_c1, title_text,
                                        dep_rule="COP_title_prep_org", conf=0.84, lang=lang,
                                        span=(sbj.idx, min(org.idx + len(org.text), self.cfg.max_span_chars)),
                                        extra_meta={**(extra_meta or {}), "rel_surface": surf_c1},
                                    )
                                )
                            if self._valid_sro(subj_text, rel_c2, org_text):
                                triples.append(
                                    self._make_triple(
                                        doc_id, sentence_idx, sentence_id,
                                        subj_text, rel_c2, org_text,
                                        dep_rule="COP_title_prep_org", conf=0.84, lang=lang,
                                        span=(sbj.idx, min(org.idx + len(org.text), self.cfg.max_span_chars)),
                                        extra_meta={**(extra_meta or {}), "rel_surface": surf_c2},
                                    )
                                )

            # Caso 2: verbo de trabajo (work/serve/head/lead) + prep at/en
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower()
                if (lang == "es" and lemma in WORK_LEMMAS_ES) or (lang == "en" and lemma in WORK_LEMMAS_EN):
                    subjects = [c for c in token.children if c.dep_.startswith("nsubj")]
                    preps = [c for c in token.children if c.dep_ == "prep" and c.lemma_ in preps_work]
                    orgs = [gc for p in preps for gc in p.children if gc.dep_ == "pobj"]
                    for sbj in subjects:
                        if self.cfg.drop_pronoun_subjects and sbj.pos_ == "PRON":
                            continue
                        subj_text = _strip_punct_edges(_norm(doc[sbj.left_edge.i : sbj.right_edge.i + 1].text, self.cfg))
                        for org in orgs:
                            org_text = _strip_punct_edges(_norm(doc[org.left_edge.i : org.right_edge.i + 1].text, self.cfg))
                            rel_surface = "works_at" if lang == "en" else "trabaja_en"
                            rel_c, rel_surf = _canonical_relation(rel_surface, lang, self.cfg)
                            if not self._valid_sro(subj_text, rel_c, org_text):
                                continue
                            start = min(sbj.idx, token.idx, org.idx)
                            end = max(sbj.idx + len(sbj.text), token.idx + len(token.text), org.idx + len(org.text))
                            triples.append(
                                self._make_triple(
                                    doc_id, sentence_idx, sentence_id,
                                    subj_text, rel_c, org_text,
                                    dep_rule="VERB_work_prep_org", conf=0.82, lang=lang,
                                    span=(start, min(end, self.cfg.max_span_chars)),
                                    extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                                )
                            )

        return triples

    # -------- Fallback con regex --------

    def _build_regex_es(self) -> List[Tuple[re.Pattern, str, float]]:
        flags = re.IGNORECASE | re.UNICODE
        pats: List[Tuple[re.Pattern, str, float]] = []
        # Copulares
        pats.append((re.compile(r"^\s*(.+?)\s+(es|son|fue|eran|será|ha sido)\s+(.+?)\s*$", flags), "cop", 0.72))
        # Verbales típicos de adquisición
        pats.append((re.compile(r"^\s*(.+?)\s+(adquir(i[oó]|e|irá)|compra|compr[oó])\s+(.+?)\s*$", flags), "verb_do", 0.7))
        # Empleo / cargo
        pats.append((re.compile(r"^\s*(.+?)\s+es\s+(.+?)\s+en\s+(.+?)\s*$", flags), "is_title_at", 0.78))
        pats.append((re.compile(r"^\s*(.+?)\s+trabaja\s+en\s+(.+?)\s*$", flags), "works_at", 0.76))
        # Preposicionales
        pats.append((re.compile(r"^\s*(.{3,}?)\s+(de|en|para|con)\s+(.{2,}?)\s*$", flags), "prep", 0.6))
        # Pasiva de adquisición (Y fue adquirido por X)
        pats.append((re.compile(r"^\s*(.+?)\s+fue\s+adquirid[oa]s?\s+por\s+(.+?)\s*$", flags), "passive_acq", 0.82))
        return pats

    def _build_regex_en(self) -> List[Tuple[re.Pattern, str, float]]:
        flags = re.IGNORECASE | re.UNICODE
        pats: List[Tuple[re.Pattern, str, float]] = []
        pats.append((re.compile(r"^\s*(.+?)\s+(is|are|was|were|has been)\s+(.+?)\s*$", flags), "cop", 0.72))
        pats.append((re.compile(r"^\s*(.+?)\s+(acquired|buys|bought|purchased)\s+(.+?)\s*$", flags), "verb_do", 0.7))
        pats.append((re.compile(r"^\s*(.+?)\s+is\s+(.+?)\s+(at|in)\s+(.+?)\s*$", flags), "is_title_at", 0.78))
        pats.append((re.compile(r"^\s*(.+?)\s+(works|work)\s+(at|in)\s+(.+?)\s*$", flags), "works_at", 0.76))
        pats.append((re.compile(r"^\s*(.{3,}?)\s+(of|in|for|with)\s+(.{2,}?)\s*$", flags), "prep", 0.6))
        pats.append((re.compile(r"^\s*(.+?)\s+was\s+acquired\s+by\s+(.+?)\s*$", flags), "passive_acq", 0.82))
        return pats

    def _extract_with_regex(
        self,
        text: str,
        sentence_idx: int,
        sentence_id: str,
        doc_id: str,
        lang: str,
        extra_meta: Optional[Dict[str, Any]],
    ) -> List[TripleIR]:
        triples: List[TripleIR] = []
        t = (text or "").strip().rstrip(";")
        if not t or _is_contact_like(t):
            return triples

        pats = self._regex_patterns_es if lang == "es" else self._regex_patterns_en

        for rx, kind, base_conf in pats:
            m = rx.match(t)
            if not m:
                continue

            def _emit(s: str, r: str, o: str, rule: str, conf: float, s1: int, e3: int):
                s = _strip_punct_edges(_norm(s, self.cfg))
                o = _strip_punct_edges(_norm(o, self.cfg))
                if not self._valid_sro(s, r, o):
                    return
                rel_canon, rel_surf = _canonical_relation(r, lang, self.cfg)
                triples.append(
                    self._make_triple(
                        doc_id, sentence_idx, sentence_id,
                        s, rel_canon, o,
                        dep_rule=rule, conf=conf, lang=lang,
                        span=(s1, min(e3, self.cfg.max_span_chars)),
                        extra_meta={**(extra_meta or {}), "rel_surface": rel_surf},
                    )
                )

            if kind == "cop":
                s, cop, o = m.group(1), m.group(2), m.group(3)
                _emit(s, "is" if lang == "en" else "es", o, "regex_cop", base_conf, m.start(1), m.end(3))

            elif kind == "verb_do":
                s, verb, o = m.group(1), m.group(2), m.group(3)
                _emit(s, verb.lower(), o, "regex_verb_do", base_conf, m.start(1), m.end(3))

            elif kind == "is_title_at":
                if lang == "es":
                    s, title, o = m.group(1), m.group(2), m.group(3)
                    # Emite (X, title, cargo) y (X, works_at, org)
                    _emit(s, "title", title, "regex_is_title_at", base_conf, m.start(1), m.end(2))
                    _emit(s, "trabaja_en", o, "regex_is_title_at", base_conf, m.start(1), m.end(3))
                else:
                    s, title, _, o = m.group(1), m.group(2), m.group(3), m.group(4)
                    _emit(s, "title", title, "regex_is_title_at", base_conf, m.start(1), m.end(2))
                    _emit(s, "works_at", o, "regex_is_title_at", base_conf, m.start(1), m.end(4))

            elif kind == "works_at":
                if lang == "es":
                    s, o = m.group(1), m.group(2)
                    _emit(s, "trabaja_en", o, "regex_works_at", base_conf, m.start(1), m.end(2))
                else:
                    s, _, o = m.group(1), m.group(2), m.group(3)
                    _emit(s, "works_at", o, "regex_works_at", base_conf, m.start(1), m.end(3))

            elif kind == "prep":
                s, prep, o = m.group(1), m.group(2), m.group(3)
                _emit(s, prep.lower(), o, "regex_prep", base_conf, m.start(1), m.end(3))

            elif kind == "passive_acq":
                if lang == "es":
                    o, s = m.group(1), m.group(2)  # "Y fue adquirido por X" -> (X, adquirió, Y)
                    _emit(s, "adquirió", o, "regex_passive_acq", base_conf, m.start(1), m.end(2))
                else:
                    o, s = m.group(1), m.group(2)  # "Y was acquired by X" -> (X, acquired, Y)
                    _emit(s, "acquired", o, "regex_passive_acq", base_conf, m.start(1), m.end(2))

            if len(triples) >= self.cfg.max_triples_per_sentence:
                break

        return triples

    # -------- Helpers --------

    def _valid_sro(self, s: str, r: str, o: str) -> bool:
        if not s or not r or not o:
            return False
        if len(s) < self.cfg.min_token_chars or len(o) < self.cfg.min_token_chars:
            return False
        if s == o:
            return False
        if len(r) > 40:
            return False
        # Evita objetos/sujetos dominados por números
        if _digits_ratio(s) > 0.5 or _digits_ratio(o) > 0.5:
            return False
        # Evita que relación sea puro número/puntuación
        if not _has_letter(r):
            return False
        return True

    def _make_triple(
        self,
        doc_id: str,
        sentence_idx: int,
        sentence_id: str,
        s: str,
        r: str,
        o: str,
        dep_rule: str,
        conf: float,
        lang: str,
        span: Tuple[int, int],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> TripleIR:
        key = "|".join([doc_id, str(sentence_idx), f"{span[0]}-{span[1]}", dep_rule, s, r, o])
        tid = _short_sha1(key)
        meta = {
            "sentence_id": sentence_id,
            "sentence_idx": sentence_idx,
            "char_span_in_sentence": [int(span[0]), int(span[1])],
            "dep_rule": dep_rule,
            "conf": round(float(conf), 3),
            "lang": lang,
        }
        if extra_meta:
            meta.update(extra_meta)
        return TripleIR(
            id=tid,
            subject=s,
            relation=r,
            object=o,
            meta=meta,
        )


# ----------------------------
# Helpers de acceso a campos de SentenceIR / dict
# ----------------------------

def _field(sent: Any, keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if hasattr(sent, k):
            try:
                return getattr(sent, k)
            except Exception:
                pass
    if isinstance(sent, Mapping):
        for k in keys:
            if k in sent:
                return sent[k]
    meta = getattr(sent, "meta", None)
    if isinstance(meta, Mapping):
        for k in keys:
            if k in meta:
                return meta[k]
    return None


# ----------------------------
# Runner utilitario (standalone)
# ----------------------------

def run_on_file(
    in_path: str,
    out_path: str,
    cfg: Optional[DepTripleConfig] = None,
) -> DocumentTriples:
    """Carga un JSON de DocumentSentences y escribe DocumentTriples."""
    cfg = cfg or DepTripleConfig()
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validación suave
    try:
        ds = DocumentSentences(**data)  # type: ignore
        doc_id = ds.doc_id
    except Exception:
        ds = DocumentSentences.model_validate(  # type: ignore
            {"doc_id": data.get("doc_id", "DOC-UNKNOWN"),
             "sentences": data.get("sentences", []),
             "meta": data.get("meta", {})}
        )
        doc_id = ds.doc_id

    extractor = DepTripleExtractor(cfg)
    dt = extractor.extract_document(ds)

    # Escritura atómica
    tmp = f"{out_path}.tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dt.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

    logger.info("Triples escritos | doc_id=%s path=%s triples=%d",
                doc_id, out_path, len(dt.triples))
    return dt
