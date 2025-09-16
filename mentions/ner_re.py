# -*- coding: utf-8 -*-
"""
mentions/ner_re.py — Extractor de Menciones (NER/RE) para T2G

Objetivos de diseño
-------------------
- Convertir oraciones (DocumentSentences) en:
    * EntityMention: menciones de entidades con spans, confianza, idioma y trazabilidad.
    * RelationMention: menciones de relaciones entre entidades de la misma oración.
- Alta robustez sin red: spaCy (si está), regex estructurado y léxicos genéricos ES/EN.
- Velocidad: batching con nlp.pipe por idioma; opcional Aho-Corasick como gazetteer rápido.
- Determinismo: IDs SHA1 y reglas estáticas; dedupe por span/label y doc-level unify.
- Integración con 'triples': consenso/boost opcional para mejorar precisión y auditoría.
- Extensibilidad: hook opcional a un re-ranker de relaciones (transformers locales).

Compatibilidad
--------------
- Entrada: sentence_filter.schemas.DocumentSentences o dict equivalente.
- Salida: mentions.schemas.DocumentMentions (EntityMention/RelationMention con trazabilidad).
- Sin dependencias de red; CPU-only. Transformers (opcional) deben estar instalados localmente.

Nota:
- Para máxima compatibilidad con tus schemas actuales, los campos extra
  (p. ej., canonical_label, token_span_in_sentence, lang_source, norm_hints, evidence)
  se guardan dentro de `meta`.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Any, Mapping, Set
import re
import json
import hashlib
import os
import glob
import logging

# Entrada laxa (no rompemos si el import falla en tiempo de import)
try:
    from sentence_filter.schemas import DocumentSentences  # type: ignore
except Exception:
    DocumentSentences = Any  # type: ignore

# Contratos de salida
from .schemas import DocumentMentions, EntityMention, RelationMention

# Gazetteer opcional (pyahocorasick)
try:
    import ahocorasick
    _HAS_AHO = True
except Exception:
    _HAS_AHO = False

logger = logging.getLogger(__name__)

# HF (opcional) — wrapper local; si no existe, seguimos sin re-rank
try:
    from mentions.hf_plugins import HFRelRanker  # type: ignore
except Exception:
    HFRelRanker = None  # type: ignore


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _sha1(text: str, n: int = 40) -> str:
    """SHA1 hex (n primeras)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Acceso resiliente para Pydantic o dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _canonical_entity(label: str) -> str:
    """Normaliza etiquetas de entidad ES/EN a un conjunto canónico."""
    m = (label or "").upper()
    if m in {"GPE", "LOC"}:
        return "LOC"
    if m in {"ORG", "ORGANIZATION"}:
        return "ORG"
    if m in {"PERSON", "PER"}:
        return "PERSON"
    if m in {"DATE"}:
        return "DATE"
    if m in {"MONEY"}:
        return "MONEY"
    if m in {"URL"}:
        return "URL"
    if m in {"EMAIL", "E-MAIL"}:
        return "EMAIL"
    if m in {"TITLE", "POSITION", "CARGO"}:
        return "TITLE"
    if m in {"DEGREE", "DEG"}:
        return "DEGREE"
    if m in {"PRODUCT"}:
        return "PRODUCT"
    return m or "MISC"

def _canonical_relation(label: str) -> str:
    """Normaliza relaciones ES/EN a forma canónica."""
    l = (label or "").strip().lower()
    mapping = {
        "trabaja_en": "works_at",
        "works at": "works_at",
        "works_at": "works_at",
        "is": "title", "es": "title",
        "member of": "member_of", "miembro de": "member_of",
        "located in": "located_in", "ubicado en": "located_in",
        "sede en": "located_in", "con sede en": "located_in",
        "acquired": "acquired", "adquirió": "acquired",
        "fue adquirido por": "acquired", "was acquired by": "acquired",
        "author of": "author_of", "autor de": "author_of",
        "collaboration with": "collaboration_with", "colaboración con": "collaboration_with",
        "title": "title",
    }
    return mapping.get(l, l.replace(" ", "_") or "related_to")

def _guess_lang(text: str) -> str:
    """Heurística ligera ES/EN; evita dependencias externas."""
    t = f" {text.lower()} "
    es = sum(k in t for k in (" el ", " la ", " los ", " las ", " de ", " para ", " es ", " fue ", " en "))
    en = sum(k in t for k in (" the ", " of ", " for ", " is ", " was ", " in ", " at "))
    if es > en:
        return "es"
    if en > es:
        return "en"
    return "und"

def _get_can_ent_label(e: EntityMention) -> str:
    """Etiqueta canónica de entidad (meta['canonical_label'] fallback a label)."""
    meta = getattr(e, "meta", {}) or {}
    if isinstance(meta, dict):
        can = meta.get("canonical_label")
        if isinstance(can, str) and can:
            return can
    return e.label

def _get_can_rel_label(r: RelationMention) -> str:
    """Etiqueta canónica de relación (meta['canonical_label'] fallback a label)."""
    meta = getattr(r, "meta", {}) or {}
    if isinstance(meta, dict):
        can = meta.get("canonical_label")
        if isinstance(can, str) and can:
            return can
    return r.label


# ---------------------------------------------------------------------
# Configuración + Léxicos ligeros
# ---------------------------------------------------------------------

@dataclass
class MentionConfig:
    """
    Config principal del extractor de Mentions (NER/RE).

    Campos clave:
      - use_spacy: 'auto' | 'force' | 'off'
      - lang_pref: 'auto' | 'es' | 'en'
      - allowed_*: subconjuntos permitidos de etiquetas (None = defaults)
      - min_conf_keep: umbral global para filtrar menciones por confianza
      - boost_from_triples_glob: glob de JSON de triples para consenso/boost (opcional)
      - use_transformers + hf_*: re-ranker opcional de relaciones (sin red)
    """
    # Núcleo
    use_spacy: str = "auto"                 # 'auto' | 'force' | 'off'
    lang_pref: str = "auto"                 # 'auto' | 'es' | 'en'
    min_conf_keep: float = 0.66

    # Etiquetas permitidas (si None, se setean en __post_init__)
    allowed_entity_labels: Optional[Set[str]] = None
    allowed_relation_labels: Optional[Set[str]] = None

    # Post-procesado
    drop_overlapping_entities: bool = True
    dedupe_entities: bool = True
    dedupe_relations: bool = True
    max_relations_per_sentence: int = 6
    canonicalize_labels: bool = True
    spacy_disable: Tuple[str, ...] = ("lemmatizer", "textcat")

    # Consenso con Triples
    boost_from_triples_glob: Optional[str] = None
    boost_overlap_chars: int = 6
    boost_conf: float = 0.05

    # Re-ranker HF (opcional)
    use_transformers: bool = False
    hf_rel_model_path: Optional[str] = None     # p.ej. "hf_plugins/re_models/relation_mini"
    hf_device: str = "cpu"                      # "cpu" | "cuda" | "mps"
    hf_batch_size: int = 16
    hf_mode: str = "re_rank"                    # "re_rank" (recalifica candidatos) | "multi" (predice etiqueta)
    hf_min_prob: float = 0.55                   # umbral mínimo del modelo para considerar que "hay relación"
    transformer_weight: float = 0.30            # peso del score HF en la fusión: conf_fused = 0.7*base + 0.3*hf
    triples_boost_weight: float = 1.0           # multiplicador del boost_conf cuando hay consenso con triples

    def __post_init__(self) -> None:
        # Defaults de etiquetas
        if self.allowed_entity_labels is None:
            self.allowed_entity_labels = {
                "PERSON", "ORG", "LOC", "DATE", "MONEY", "EMAIL", "URL",
                "TITLE", "DEGREE", "PRODUCT"
            }
        if self.allowed_relation_labels is None:
            self.allowed_relation_labels = {
                "works_at", "title", "located_in", "acquired", "member_of",
                "author_of", "degree", "collaboration_with"
            }

        # Sanitizar numéricos
        self.min_conf_keep = float(max(0.0, min(0.99, self.min_conf_keep)))
        self.boost_conf = float(max(0.0, min(0.5, self.boost_conf)))
        self.hf_min_prob = float(max(0.0, min(0.99, self.hf_min_prob)))
        self.transformer_weight = float(max(0.0, min(1.0, self.transformer_weight)))
        self.triples_boost_weight = float(max(0.1, min(2.0, self.triples_boost_weight)))

        # Normalizar banderas de modo/idioma/spaCy
        self.use_spacy = (self.use_spacy or "auto").lower()
        if self.use_spacy not in {"auto", "force", "off"}:
            self.use_spacy = "auto"
        self.lang_pref = (self.lang_pref or "auto").lower()
        if self.lang_pref not in {"auto", "es", "en"}:
            self.lang_pref = "auto"
        self.hf_mode = (self.hf_mode or "re_rank").lower()
        if self.hf_mode not in {"re_rank", "multi"}:
            self.hf_mode = "re_rank"
        self.hf_device = (self.hf_device or "cpu").lower()


# Léxicos genéricos para Phrase/Gazetteer (ampliables en resources/ en el futuro)
TITLES_ES_EN = [
    "Chief Executive Officer", "Chief Data Officer", "Chief Operating Officer",
    "Vice President", "Director", "Manager", "Head", "Lead",
    "Gerente", "Directora", "Director", "Jefa", "Jefe", "Líder", "Arquitecto", "Arquitecta",
    "Profesor", "Professor", "Engineer", "Scientist", "Analyst", "Consultant"
]
DEGREES_ES_EN = [
    "Licenciatura", "Maestría", "Doctorado", "MBA", "PhD", "MSc", "BSc",
    "Bachelor of Science", "Master of Science", "Ingeniería"
]

# Entidades por regex (tipos “estructurados”)
RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
RE_URL   = re.compile(r"\bhttps?://[^\s)]+|\bwww\.[^\s)]+\b")
RE_MONEY = re.compile(r"\b(?:MXN|USD|EUR|US\$|\$)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b")
RE_DATE  = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)\w*\s+\d{1,2},?\s+\d{2,4})\b",
    flags=re.IGNORECASE
)

# Relaciones por regex (fallback)
RE_WORKS_AT   = re.compile(r"\b(works|work|trabaja|labora|is at|is in|works at|trabaja en)\b", re.IGNORECASE)
RE_TITLE_AT   = re.compile(r"\b(is|es|serves as|actúa como|se desempeña como)\b", re.IGNORECASE)
RE_MEMBER_OF  = re.compile(r"\b(member of|miembro de)\b", re.IGNORECASE)
RE_LOCATED_IN = re.compile(r"\b(located in|con sede en|sede en|ubicado en)\b", re.IGNORECASE)
RE_ACQUIRED   = re.compile(r"\b(acquired|adquirió|fue adquirido por|was acquired by|bought|compró)\b", re.IGNORECASE)
RE_AUTHOR_OF  = re.compile(r"\b(author of|autor de)\b", re.IGNORECASE)
RE_COLLAB_WITH= re.compile(r"\b(collaboration with|colaboración con)\b", re.IGNORECASE)


# ---------------------------------------------------------------------
# Contexto por documento (unificación de entidades)
# ---------------------------------------------------------------------

@dataclass
class DocContext:
    """
    Índice (label_canónica, superficie) -> entity_id para unificar
    la misma entidad textual a lo largo del documento (coherencia doc-level).
    """
    doc_id: str
    ent_index: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def unify_entity(self, e: EntityMention) -> EntityMention:
        can = _get_can_ent_label(e).upper()
        key = (can, e.text.strip().lower())
        if key in self.ent_index:
            e.id = self.ent_index[key]
            return e
        self.ent_index[key] = e.id
        return e


# ---------------------------------------------------------------------
# Extractor principal
# ---------------------------------------------------------------------

class NERREExtractor:
    """
    Extrae EntityMention y RelationMention por oración, combinando:
      - spaCy NER + PhraseMatcher (o Aho-Corasick si disponible).
      - DependencyMatcher (patrones estructurales) para relaciones clave.
      - Regex (fallback) para entidades/relaciones estructuradas.
      - Consenso/boost con TripleIR (opcional).
      - (Opcional) Re-ranker de relaciones con transformers locales.
    """

    def __init__(self, config: Optional[MentionConfig] = None):
        self.cfg = config or MentionConfig()
        self.spacy_nlp: Optional[Dict[str, Any]] = None
        self._dep_matchers: Dict[str, Tuple[Any, Any]] = {}
        self._maybe_init_spacy()
        self._build_dependency_matchers()

        # Índice opcional de triples para consenso/boost
        self.triple_index = self._maybe_load_triples_index(self.cfg.boost_from_triples_glob)

        # HF re-ranker (opcional)
        self._hf_rel_ranker = None
        if self.cfg.use_transformers and self.cfg.hf_rel_model_path:
            if HFRelRanker is None:
                logger.warning("mentions/transformers: 'transformers' no disponible; sigo sin HF.")
            else:
                try:
                    self._hf_rel_ranker = HFRelRanker(
                        model_path=self.cfg.hf_rel_model_path,
                        device=self.cfg.hf_device or "cpu",
                        batch_size=self.cfg.hf_batch_size,
                    )
                    logger.info(
                        "mentions/transformers: enabled | model=%s | device=%s",
                        self.cfg.hf_rel_model_path, self.cfg.hf_device
                    )
                except Exception as e:
                    logger.warning("mentions/transformers: no se pudo cargar (%s). Sigo sin HF.", e)
                    self._hf_rel_ranker = None

    # ------------------------ Inicialización spaCy ------------------------

    def _maybe_init_spacy(self) -> None:
        """Carga spaCy ES/EN si está permitido; usa blank+sentencizer si faltan modelos."""
        if self.cfg.use_spacy == "off":
            return
        try:
            import spacy  # type: ignore
        except Exception:
            if self.cfg.use_spacy == "force":
                raise RuntimeError("spaCy requerido pero no disponible.")
            return

        self.spacy_nlp = {}
        for lang, model in [("es", "es_core_news_sm"), ("en", "en_core_web_sm")]:
            try:
                nlp = spacy.load(model, disable=list(self.cfg.spacy_disable))
            except Exception:
                nlp = spacy.blank(lang)
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
            self.spacy_nlp[lang] = nlp

    def _get_spacy(self, lang: str):
        if self.spacy_nlp is None:
            return None
        if lang in self.spacy_nlp:
            return self.spacy_nlp[lang]
        return self.spacy_nlp.get("en")

    # ------------------------ DependencyMatcher ------------------------

    def _build_dependency_matchers(self) -> None:
        """Crea patrones DependencyMatcher para relaciones clave ES/EN."""
        if self.spacy_nlp is None:
            return
        try:
            from spacy.matcher import DependencyMatcher  # type: ignore
        except Exception:
            return

        def mk(lang: str):
            nlp = self._get_spacy(lang)
            if nlp is None:
                return None
            return nlp, DependencyMatcher(nlp.vocab)

        # EN
        pack = mk("en")
        if pack:
            nlp_en, dm_en = pack
            pattern_works_en = [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["work", "serve", "lead", "head"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}, "ENT_TYPE": "PERSON"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["at", "in"]}}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": {"IN": ["ORG", "GPE", "LOC"]}}},
            ]
            dm_en.add("works_at", [pattern_works_en])

            pattern_title_en = [
                {"RIGHT_ID": "attr", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "acomp"]}, "POS": {"IN": ["NOUN", "PROPN"]}}},
                {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "cop",  "RIGHT_ATTRS": {"DEP": "cop", "LEMMA": {"IN": ["be"]}}},
                {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["at", "in"]}}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "org",  "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "ORG"}},
                {"LEFT_ID": "attr", "REL_OP": "<", "RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}, "ENT_TYPE": "PERSON"}},
            ]
            dm_en.add("title_at", [pattern_title_en])

            pattern_acq_en = [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["acquire", "buy", "merge", "purchase"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "nsubj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "agent", "RIGHT_ATTRS": {"DEP": {"IN": ["agent", "prep"]}, "LEMMA": {"IN": ["by"]}}},
                {"LEFT_ID": "agent", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "ORG"}},
            ]
            dm_en.add("acquired_passive", [pattern_acq_en])

            self._dep_matchers["en"] = (nlp_en, dm_en)

        # ES
        pack = mk("es")
        if pack:
            nlp_es, dm_es = pack
            pattern_works_es = [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["trabajar", "liderar", "dirigir", "encabezar"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}, "ENT_TYPE": {"IN": ["PER", "PERSON"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["en"]}}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": {"IN": ["ORG", "LOC"]}}},
            ]
            dm_es.add("works_at", [pattern_works_es])

            pattern_title_es = [
                {"RIGHT_ID": "attr", "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "acomp"]}, "POS": {"IN": ["NOUN", "PROPN"]}}},
                {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "cop",  "RIGHT_ATTRS": {"DEP": "cop", "LEMMA": {"IN": ["ser"]}}},
                {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["en"]}}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "org",  "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "ORG"}},
                {"LEFT_ID": "attr", "REL_OP": "<", "RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}, "ENT_TYPE": {"IN": ["PER", "PERSON"]}}},
            ]
            dm_es.add("title_at", [pattern_title_es])

            pattern_acq_es = [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["adquirir", "comprar", "fusionar"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "nsubj", "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj"}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep",  "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["por"]}}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj",  "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "ORG"}},
            ]
            dm_es.add("acquired_passive", [pattern_acq_es])

            self._dep_matchers["es"] = (nlp_es, dm_es)

    def _extract_relations_dep_matcher(
        self, text: str, lang: str, ents: List[EntityMention],
        doc_id: str, s_idx: int, s_id: str
    ) -> List[RelationMention]:
        """Aplica DependencyMatcher para relaciones; usa entidades como anclas suaves."""
        out: List[RelationMention] = []
        pack = self._dep_matchers.get(lang)
        if not pack:
            return out
        nlp, dm = pack
        doc = nlp(text)
        matches = dm(doc)

        persons = [e for e in ents if _get_can_ent_label(e) == "PERSON"]
        orgs    = [e for e in ents if _get_can_ent_label(e) in {"ORG", "LOC"}]

        def add(subj: EntityMention, obj: EntityMention, label: str, start: int, end: int, surface: str, conf: float):
            can = _canonical_relation(label) if self.cfg.canonicalize_labels else label
            if can not in self.cfg.allowed_relation_labels:
                return
            rid = _sha1(f"{subj.id}|{can}|{obj.id}|dep")
            out.append(RelationMention(
                id=rid,
                subj_entity_id=subj.id,
                obj_entity_id=obj.id,
                label=label,
                sentence_idx=s_idx,
                sentence_id=s_id,
                char_span_in_sentence=(start, end),
                conf=conf,
                source="matcher",
                lang=lang,
                surface=surface,
                meta={"canonical_label": can, "evidence": surface},
            ))

        for match_id, token_ids in matches:
            name = doc.vocab.strings[match_id]
            toks = [doc[i] for i in token_ids]
            start = min(t.idx for t in toks)
            end = max(t.idx + len(t.text) for t in toks)
            window = text[start:end]

            if name in ("works_at", "title_at"):
                for pe in persons:
                    for oe in orgs:
                        if pe.char_span_in_sentence[0] >= start and pe.char_span_in_sentence[1] <= end \
                           and oe.char_span_in_sentence[0] >= start and oe.char_span_in_sentence[1] <= end:
                            add(pe, oe, "title" if name == "title_at" else "works_at", start, end, window, 0.84)

            elif name == "acquired_passive":
                org_list = [e for e in ents if _get_can_ent_label(e) == "ORG"]
                for a in org_list:
                    for b in org_list:
                        if a.id == b.id:
                            continue
                        if a.char_span_in_sentence[0] >= start and a.char_span_in_sentence[1] <= end \
                           and b.char_span_in_sentence[0] >= start and b.char_span_in_sentence[1] <= end:
                            # Voz pasiva: b acquired a
                            add(b, a, "acquired", start, end, window, 0.86)
        return out

    # ------------------------ Triples (índice y consenso) ------------------------

    def _maybe_load_triples_index(self, glob_path: Optional[str]) -> Dict[Tuple[str, int, str], List[Dict]]:
        """Índice {(doc_id, sentence_idx, rel_canónica) -> [triples...]} para inyección/boost."""
        if not glob_path:
            return {}
        idx: Dict[Tuple[str, int, str], List[Dict]] = {}
        for path in glob.glob(glob_path):
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                continue
            doc_id = data.get("doc_id") or os.path.splitext(os.path.basename(path))[0]
            for t in data.get("triples", []):
                rel = t.get("relation")
                sid = t.get("meta", {}).get("sentence_idx")
                if rel is None or sid is None:
                    continue
                key = (str(doc_id), int(sid), _canonical_relation(str(rel)))
                idx.setdefault(key, []).append(t)
        return idx

    def _inject_from_triples(
        self, doc_id: str, s_idx: int, s_id: str, text: str,
        entities: List[EntityMention], relations: List[RelationMention]
    ) -> None:
        """Si hay triples para (doc_id, s_idx), crea menciones mínimas y relación canónica."""
        if not self.triple_index:
            return
        for (tdoc, tsent, rel_can), triples in list(self.triple_index.items()):
            if tdoc != doc_id or tsent != s_idx:
                continue
            for t in triples:
                s_surf = str(t.get("subject", "")).strip()
                o_surf = str(t.get("object", "")).strip()
                rel = _canonical_relation(t.get("relation", "related_to"))

                # Heurística de tipo: nombres capitalizados -> PERSON; otro -> ORG
                subj_guess = "PERSON" if s_surf.istitle() else "ORG"
                obj_guess  = "ORG"

                def ensure_entity(surface: str, guess: str) -> EntityMention:
                    pos = text.find(surface)
                    if pos < 0:
                        pos, end = 0, min(len(text), max(1, len(surface)))
                    else:
                        end = pos + len(surface)
                    can = _canonical_entity(guess)
                    for e in entities:
                        if e.char_span_in_sentence == (pos, end) and _get_can_ent_label(e) == can:
                            return e
                    eid = _sha1(f"{doc_id}|{s_idx}|{pos}|{end}|{can}|triple")
                    em = EntityMention(
                        id=eid, text=text[pos:end], label=guess,
                        sentence_idx=s_idx, sentence_id=s_id,
                        char_span_in_sentence=(pos, end),
                        conf=0.70, source="triple", lang=_guess_lang(text),
                        meta={"canonical_label": can, "norm_hints": {"surface": surface}}
                    )
                    entities.append(em)
                    return em

                se = ensure_entity(s_surf, subj_guess)
                oe = ensure_entity(o_surf, obj_guess)

                rid = _sha1(f"{se.id}|{rel}|{oe.id}|triple")
                relations.append(RelationMention(
                    id=rid, subj_entity_id=se.id, obj_entity_id=oe.id,
                    label=rel,
                    sentence_idx=s_idx, sentence_id=s_id,
                    char_span_in_sentence=(0, min(len(text), max(se.char_span_in_sentence[1], oe.char_span_in_sentence[1]))),
                    conf=min(0.9, 0.75 + self.cfg.boost_conf), source="triple",
                    lang=_guess_lang(text), surface=text,
                    meta={"canonical_label": rel, "from_triple": True, "evidence": text}
                ))

    def _boost_from_triples(self, doc_id: str, s_idx: int, rels: List[RelationMention], sentence_text: str) -> None:
        """Sube conf +boost_conf si existe triple con misma relación canónica en la misma oración."""
        if not self.triple_index:
            return
        for r in rels:
            can = _get_can_rel_label(r)
            key = (doc_id, s_idx, can)
            if key in self.triple_index:
                r.conf = min(0.95, r.conf + self.cfg.boost_conf * self.cfg.triples_boost_weight)

    # ------------------------ API pública ------------------------

    def extract_document(self, ds: Any) -> DocumentMentions:
        """
        Recibe un DocumentSentences (o dict equivalente) y devuelve DocumentMentions.
        - Agrupa por idioma para hacer batching en spaCy (velocidad).
        - Unifica entidades iguales a nivel doc (DocContext).
        - Mezcla fuentes (spaCy/phrase/regex) y dedupe/filtra por confianza.
        """
        doc_id = _safe_getattr(ds, "doc_id", _safe_getattr(ds, "id", "UNKNOWN_DOC"))
        sents = _safe_getattr(ds, "sentences", [])
        doc_lang = self.cfg.lang_pref if self.cfg.lang_pref != "auto" else self._derive_doc_lang(sents)
        doc_ctx = DocContext(doc_id=str(doc_id))

        entities_all: List[EntityMention] = []
        relations_all: List[RelationMention] = []

        # --- Agrupa oraciones por idioma para pipelining ---
        groups: Dict[str, List[Tuple[int, Any]]] = {}
        for s_idx, s in enumerate(sents):
            text = _safe_getattr(s, "text", "") or ""
            if not text.strip():
                continue
            lang = doc_lang if self.cfg.lang_pref != "auto" else _guess_lang(text)
            groups.setdefault(lang, []).append((s_idx, s))

        for lang, items in groups.items():
            # spaCy opcional (doc list) para reusar análisis
            nlp = self._get_spacy(lang)
            docs = []
            if nlp is not None:
                texts = [_safe_getattr(s, "text", "") or "" for _, s in items]
                docs = list(nlp.pipe(texts, batch_size=64))
            # Procesa oración por oración (reutilizando spacy_doc si existe)
            for i, (s_idx, s) in enumerate(items):
                text  = _safe_getattr(s, "text", "") or ""
                if not text:
                    continue
                s_id  = _safe_getattr(s, "id", f"{doc_id}-sent-{s_idx}")
                s_meta= _safe_getattr(s, "meta", {}) or {}
                spacy_doc = docs[i] if (nlp is not None and i < len(docs)) else None

                # -------- ENTIDADES --------
                ents: List[EntityMention] = []
                ents.extend(self._extract_entities_spacy(text, lang, doc_id, s_idx, s_id, s_meta, spacy_doc=spacy_doc))
                ents.extend(self._extract_entities_phrase(text, lang, doc_id, s_idx, s_id, s_meta, spacy_doc=spacy_doc))
                ents.extend(self._extract_entities_regex(text, lang, doc_id, s_idx, s_id, s_meta))
                ents = self._merge_entities(ents)
                ents = [doc_ctx.unify_entity(e) for e in self._postprocess_entities(ents)]
                entities_all.extend(ents)

                # -------- RELACIONES --------
                rels: List[RelationMention] = []
                rels.extend(self._extract_relations_dep_matcher(text, lang, ents, doc_id, s_idx, s_id))
                rels.extend(self._extract_relations_regex(text, lang, ents, doc_id, s_idx, s_id))
                rels = self._merge_relations(rels)
                rels = self._postprocess_relations(rels, self.cfg.max_relations_per_sentence)

                # Consenso con triples (inyecta/boosta)
                if self.triple_index:
                    self._inject_from_triples(doc_id, s_idx, s_id, text, ents, rels)
                    self._boost_from_triples(doc_id, s_idx, rels, text)

                # Re-rank opcional con HF
                if self._hf_rel_ranker is not None and rels and ents:
                    self._hf_rescore_relations(text, ents, rels)

                relations_all.extend(rels)

        # --- Filtrado global por confianza ---
        entities_all  = [e for e in entities_all  if e.conf >= self.cfg.min_conf_keep]
        relations_all = [r for r in relations_all if r.conf >= self.cfg.min_conf_keep]

        # --- Meta serializable (sets -> lists) ---
        def _jsonable_cfg(cfg: MentionConfig) -> Dict[str, Any]:
            d = asdict(cfg)
            for k, v in list(d.items()):
                if isinstance(v, set):
                    d[k] = sorted(list(v))
            return d

        meta = {
            "params": _jsonable_cfg(self.cfg),
            "counters": {"entities": len(entities_all), "relations": len(relations_all)},
            "version": "mentions_v1"
        }
        return DocumentMentions(doc_id=str(doc_id), entities=entities_all, relations=relations_all, meta=meta)

    # ------------------------ ENTIDADES ------------------------

    def _extract_entities_spacy(
        self, text: str, lang: str, doc_id: str, s_idx: int, s_id: str, s_meta: Dict, spacy_doc=None
    ) -> List[EntityMention]:
        """spaCy NER (si está disponible). Conf ~0.72 (score fijo; modelos 'sm' no exponen probas)."""
        out: List[EntityMention] = []
        nlp = self._get_spacy(lang)
        if nlp is None:
            return out
        doc = spacy_doc if spacy_doc is not None else nlp(text)
        for ent in getattr(doc, "ents", []):
            can = _canonical_entity(ent.label_)
            if can not in self.cfg.allowed_entity_labels:
                continue
            start, end = int(ent.start_char), int(ent.end_char)
            eid = _sha1(f"{doc_id}|{s_idx}|{start}|{end}|{can}|spacy")
            out.append(EntityMention(
                id=eid, text=ent.text, label=ent.label_,
                sentence_idx=s_idx, sentence_id=s_id,
                char_span_in_sentence=(start, end),
                conf=0.72, source="spacy", lang=lang,
                meta={
                    "sentence_meta": s_meta,
                    "canonical_label": can,
                    "token_span_in_sentence": (int(ent.start), int(ent.end)),
                    "lang_source": "spacy"
                }
            ))
        return out

    def _extract_entities_phrase(
        self, text: str, lang: str, doc_id: str, s_idx: int, s_id: str, s_meta: Dict, spacy_doc=None
    ) -> List[EntityMention]:
        """
        TITLE/DEGREE a partir de expresiones multi-palabra.
        - Usa Aho-Corasick si está disponible (rápido).
        - Si no, usa spaCy PhraseMatcher (requiere vocab/tokenizer).
        """
        out: List[EntityMention] = []
        # Aho-Corasick (opcional)
        if _HAS_AHO:
            auto = ahocorasick.Automaton()
            for p in TITLES_ES_EN:
                auto.add_word(p.lower(), ("TITLE", p))
            for p in DEGREES_ES_EN:
                auto.add_word(p.lower(), ("DEGREE", p))
            auto.make_automaton()
            low = text.lower()
            for end_idx, (lbl, surface) in auto.iter(low):
                start_idx = end_idx - len(surface) + 1
                can = _canonical_entity(lbl)
                if can not in self.cfg.allowed_entity_labels:
                    continue
                eid = _sha1(f"{doc_id}|{s_idx}|{start_idx}|{start_idx+len(surface)}|{can}|aho")
                out.append(EntityMention(
                    id=eid, text=text[start_idx:start_idx+len(surface)], label=lbl,
                    sentence_idx=s_idx, sentence_id=s_id,
                    char_span_in_sentence=(start_idx, start_idx+len(surface)),
                    conf=0.67, source="gazetteer", lang=lang,
                    meta={"sentence_meta": s_meta, "canonical_label": can, "norm_hints": {"surface": surface}, "lang_source": "heuristic"}
                ))
            return out

        # PhraseMatcher (fallback)
        nlp = self._get_spacy(lang)
        if nlp is None:
            return out
        from spacy.matcher import PhraseMatcher  # type: ignore
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns_t = [nlp.make_doc(p) for p in TITLES_ES_EN]
        patterns_d = [nlp.make_doc(p) for p in DEGREES_ES_EN]
        if patterns_t:
            matcher.add("TITLE", patterns_t)
        if patterns_d:
            matcher.add("DEGREE", patterns_d)
        doc = spacy_doc if spacy_doc is not None else nlp(text)
        for match_id, start, end in matcher(doc):
            label = nlp.vocab.strings[match_id]
            start_char = doc[start].idx
            end_char   = doc[end - 1].idx + len(doc[end - 1])
            can = _canonical_entity(label)
            if can not in self.cfg.allowed_entity_labels:
                continue
            eid = _sha1(f"{doc_id}|{s_idx}|{start_char}|{end_char}|{can}|phrase")
            out.append(EntityMention(
                id=eid, text=text[start_char:end_char], label=label,
                sentence_idx=s_idx, sentence_id=s_id,
                char_span_in_sentence=(start_char, end_char),
                conf=0.68, source="phrase", lang=lang,
                meta={"sentence_meta": s_meta, "canonical_label": can, "norm_hints": {"surface": text[start_char:end_char]}, "lang_source": "spacy"}
            ))
        return out

    def _extract_entities_regex(
        self, text: str, lang: str, doc_id: str, s_idx: int, s_id: str, s_meta: Dict
    ) -> List[EntityMention]:
        """EMAIL/URL/MONEY/DATE por regex (estructurado), conf 0.60–0.64."""
        out: List[EntityMention] = []

        def add(m: re.Match, label: str, conf: float):
            start, end = m.span()
            can = _canonical_entity(label)
            if can not in self.cfg.allowed_entity_labels:
                return
            eid = _sha1(f"{doc_id}|{s_idx}|{start}|{end}|{can}|regex")
            out.append(EntityMention(
                id=eid, text=text[start:end], label=label,
                sentence_idx=s_idx, sentence_id=s_id,
                char_span_in_sentence=(start, end),
                conf=conf, source="regex", lang=lang,
                meta={"sentence_meta": s_meta, "canonical_label": can, "norm_hints": {"surface": text[start:end]}, "lang_source": ("heuristic" if self.cfg.lang_pref == "auto" else "config")}
            ))

        for m in RE_EMAIL.finditer(text): add(m, "EMAIL", 0.64)
        for m in RE_URL.finditer(text):   add(m, "URL",   0.62)
        for m in RE_MONEY.finditer(text): add(m, "MONEY", 0.62)
        for m in RE_DATE.finditer(text):  add(m, "DATE",  0.62)
        return out

    def _merge_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """
        Fusión por (span, label canónica) con consenso:
        - Si dos fuentes coinciden en el mismo span/label -> boost +0.05 (cap 0.95)
        - Prefiere no-regex ante empate (spacy/phrase > regex)
        """
        merged: Dict[Tuple[int, int, str], EntityMention] = {}
        for e in entities:
            can = _get_can_ent_label(e)
            key = (e.char_span_in_sentence[0], e.char_span_in_sentence[1], can)
            if key not in merged:
                merged[key] = e
            else:
                merged[key].conf = min(0.95, max(merged[key].conf, e.conf) + 0.05)
                if e.source != "regex" and merged[key].source == "regex":
                    merged[key] = e
        return list(merged.values())

    def _postprocess_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Drop overlaps (prefiere conf alta / span largo), dedupe exacto y filtro de etiquetas."""
        if not entities:
            return entities
        entities = sorted(
            entities,
            key=lambda e: (e.conf, e.char_span_in_sentence[1] - e.char_span_in_sentence[0]),
            reverse=True,
        )
        out: List[EntityMention] = []
        occupied: List[Tuple[int, int]] = []
        for e in entities:
            can = _get_can_ent_label(e)
            if can not in self.cfg.allowed_entity_labels:
                continue
            if self.cfg.drop_overlapping_entities:
                s, t = e.char_span_in_sentence
                if any(not (t <= os_ or s >= oe_) for (os_, oe_) in occupied):
                    continue
                occupied.append((s, t))
            out.append(e)
        if self.cfg.dedupe_entities:
            seen = set(); uniq: List[EntityMention] = []
            for e in out:
                key = (e.char_span_in_sentence, _get_can_ent_label(e), e.text.lower())
                if key in seen: continue
                seen.add(key); uniq.append(e)
            out = uniq
        return out

    # ------------------------ RELACIONES ------------------------

    def _extract_relations_regex(
        self, text: str, lang: str, ents: List[EntityMention],
        doc_id: str, s_idx: int, s_id: str
    ) -> List[RelationMention]:
        """Relaciones por regex bilingüe; conf 0.66–0.70 (fallback)."""
        out: List[RelationMention] = []

        def add(subj: EntityMention, obj: EntityMention, label: str, start: int, end: int, surface: str, conf: float):
            can = _canonical_relation(label) if self.cfg.canonicalize_labels else label
            if can not in self.cfg.allowed_relation_labels:
                return
            rid = _sha1(f"{subj.id}|{can}|{obj.id}|regex")
            out.append(RelationMention(
                id=rid, subj_entity_id=subj.id, obj_entity_id=obj.id,
                label=label,
                sentence_idx=s_idx, sentence_id=s_id,
                char_span_in_sentence=(start, end),
                conf=conf, source="regex", lang=lang,
                surface=surface, meta={"canonical_label": can, "evidence": surface}
            ))

        persons = [e for e in ents if _get_can_ent_label(e) == "PERSON"]
        orgs    = [e for e in ents if _get_can_ent_label(e) == "ORG"]
        locs    = [e for e in ents if _get_can_ent_label(e) == "LOC"]

        for pe in persons:
            for oe in orgs:
                start = min(pe.char_span_in_sentence[0], oe.char_span_in_sentence[0])
                end   = max(pe.char_span_in_sentence[1], oe.char_span_in_sentence[1])
                window = text[start:end]
                if RE_WORKS_AT.search(window):
                    add(pe, oe, "works_at", start, end, window, 0.68)
                if RE_TITLE_AT.search(window):
                    add(pe, oe, "title", start, end, window, 0.68)
                if RE_MEMBER_OF.search(window):
                    add(pe, oe, "member_of", start, end, window, 0.66)

        for oe in orgs:
            for le in locs:
                start = min(oe.char_span_in_sentence[0], le.char_span_in_sentence[0])
                end   = max(oe.char_span_in_sentence[1], le.char_span_in_sentence[1])
                window = text[start:end]
                if RE_LOCATED_IN.search(window):
                    add(oe, le, "located_in", start, end, window, 0.66)

        # Adquisiciones: dos ORG con verbo de adquisición en ventana
        for i, a in enumerate(orgs):
            for b in orgs[i+1:]:
                start = min(a.char_span_in_sentence[0], b.char_span_in_sentence[0])
                end   = max(a.char_span_in_sentence[1], b.char_span_in_sentence[1])
                window = text[start:end]
                if RE_ACQUIRED.search(window):
                    add(a, b, "acquired", start, end, window, 0.66)

        for pe in persons:
            for oe in orgs:
                start = min(pe.char_span_in_sentence[0], oe.char_span_in_sentence[0])
                end   = max(pe.char_span_in_sentence[1], oe.char_span_in_sentence[1])
                window = text[start:end]
                if RE_AUTHOR_OF.search(window):
                    add(pe, oe, "author_of", start, end, window, 0.66)
                if RE_COLLAB_WITH.search(window):
                    add(pe, oe, "collaboration_with", start, end, window, 0.66)

        return out

    def _merge_relations(self, rels: List[RelationMention]) -> List[RelationMention]:
        """Fusión por (subj_id, rel_canónica, obj_id) con boost de consenso."""
        merged: Dict[Tuple[str, str, str], RelationMention] = {}
        for r in rels:
            can = _get_can_rel_label(r)
            key = (r.subj_entity_id, can, r.obj_entity_id)
            if key not in merged:
                merged[key] = r
            else:
                merged[key].conf = min(0.95, max(merged[key].conf, r.conf) + 0.05)
                if r.source == "matcher" and merged[key].source != "matcher":
                    merged[key] = r
        return list(merged.values())

    def _postprocess_relations(self, rels: List[RelationMention], max_per_sent: int) -> List[RelationMention]:
        """Filtra etiquetas, dedupe exacto, ordena por conf y recorta a top-K por oración."""
        if not rels:
            return rels
        rels = [r for r in rels if _get_can_rel_label(r) in self.cfg.allowed_relation_labels]
        if self.cfg.dedupe_relations:
            seen = set(); uniq: List[RelationMention] = []
            for r in rels:
                key = (r.subj_entity_id, _get_can_rel_label(r), r.obj_entity_id)
                if key in seen: continue
                seen.add(key); uniq.append(r)
            rels = uniq
        rels = sorted(rels, key=lambda r: r.conf, reverse=True)[:max_per_sent]
        return rels

    # ------------------------ Re-score HF (opcional) ------------------------

    def _hf_rescore_relations(self, sentence_text: str, ents: List[EntityMention], rels: List[RelationMention]) -> None:
        """
        Re-rank ligero con HFRelRanker (clasificador local):
        - Construye inputs con oración + SUBJ/OBJ + tipos.
        - Ajusta confianza hacia 0.75/0.85 si el modelo está seguro.
        """
        if not rels or not ents or self._hf_rel_ranker is None:
            return

        inputs = []
        for r in rels:
            s = next((e for e in ents if e.id == r.subj_entity_id), None)
            o = next((e for e in ents if e.id == r.obj_entity_id), None)
            if not s or not o:
                continue
            stype = _get_can_ent_label(s)
            otype = _get_can_ent_label(o)
            marked = sentence_text
            # Marcado suave (1° aparición de cada superficie)
            if s.text:
                marked = marked.replace(s.text, f"[SUBJ]{s.text}[/SUBJ]", 1)
            if o.text and o.text != s.text:
                marked = marked.replace(o.text, f"[OBJ]{o.text}[/OBJ]", 1)
            # Contexto mínimo adicional (tipos)
            text = f"{marked}\nSUBJ_TYPE={stype}\nOBJ_TYPE={otype}"
            inputs.append((r, text))

        if not inputs:
            return

        try:
            scores = self._hf_rel_ranker.predict([t for _, t in inputs])  # list[float] in [0,1]
        except Exception as e:
            logger.warning("HFRelRanker.predict falló (%s). Continúo sin re-rank.", e)
            return

        w = float(self.cfg.transformer_weight)
        thr = float(self.cfg.hf_min_prob)
        for (r, _), p in zip(inputs, scores):
            p = float(p)
            if p < thr:
                continue
            # Fusión lineal con cap
            fused = min(0.99, (1.0 - w) * float(r.conf) + w * p)
            r.conf = max(r.conf, fused)
            r.source = (r.source + "+transformer") if "transformer" not in (r.source or "") else r.source
            # Traza en meta
            meta = r.meta or {}
            meta["hf_score"] = round(p, 4)
            meta["hf_model"] = self.cfg.hf_rel_model_path
            r.meta = meta

    # ------------------------ Idioma del documento ------------------------

    def _derive_doc_lang(self, sentences: List[Any]) -> str:
        """Deriva idioma dominante del doc con un muestreo de oraciones."""
        sample = " ".join((_safe_getattr(s, "text", "") or "") for s in sentences[:6])
        return _guess_lang(sample)
