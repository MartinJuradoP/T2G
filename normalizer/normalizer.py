# -*- coding: utf-8 -*-
"""
normalizer/normalizer.py
========================

Etapa de **Normalización**:
DocumentMentions (dict)  ──►  DocumentEntities (Pydantic)

Responsabilidades:
- Mapear etiquetas de Mentions (incluyendo variantes spaCy) a tipos T2G.
- Re-etiquetar por superficie (heurísticas: direcciones, ORG por sufijo/acrónimo,
  títulos comunes, países/ciudades, campos/disciplinas).
- (NUEVO) Co-tipado usando relaciones de Mentions (p. ej., `works_at`, `title`)
  para corregir el tipo de sujeto/objeto antes de normalizar.
- Normalizar valores específicos por tipo (DATE ISO-8601, MONEY tipado, EMAIL/URL).
- Generar *claves canónicas* (org_key/person_key/url_norm/email_norm/value_iso/...).
- Resolver conflictos PERSON↔ORG por la misma clave canónica (preferir ORG).
- Hacer *merge/dedupe* por clave con reglas por tipo.
- Conservar trazabilidad (lista de `mentions`).
- Producir `meta.counters` y `meta.config`.

Notas:
- Esta etapa se beneficia de un buen IE (Mentions + boost con Triples), pero añade
  defensas y canónicos por si Mentions trae ruido de superficie.
- No requiere leer Triples ni Sentences; solo usa `mentions.entities` y opcionalmente
  `mentions.relations` para co-tipado.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .config import NormalizeConfig
from .schemas import DocumentEntities, NormalizedEntity
from .merge import merge_by_key
from .rules import TITLES_MAP, DEGREES_MAP, ORG_SUFFIXES_MX, ORG_SUFFIXES_EN
from .utils import (
    normalize_date, normalize_money, normalize_email, normalize_url,
    normalize_org, normalize_person, stable_id, clean_attrs,
    is_address_like, is_org_like, is_title_like, is_field_of_study, is_country_or_city_lite
)

# -----------------------------------------------------------------------------
# Label → Tipo (canónico del normalizador)
# -----------------------------------------------------------------------------

LABEL2TYPE: Dict[str, str] = {
    # Canónicas del proyecto
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "EMAIL": "EMAIL",
    "URL": "URL",
    "TITLE": "TITLE",
    "DEGREE": "DEGREE",
    "PRODUCT": "PRODUCT",
    "ID": "ID",
    # Variantes comunes (spaCy, etc.)
    "PER": "PERSON",
    "GPE": "LOC",
    "FAC": "LOC",
    "NORP": "ORG",          # nacionalidades/organizaciones → suele ayudar más como ORG para índices
    "JOB_TITLE": "TITLE",
}

# -----------------------------------------------------------------------------
# Helpers locales (sin depender de utils extra)
# -----------------------------------------------------------------------------

def _looks_like_acronym(s: str) -> bool:
    """
    Heurística simple: un token (o dos con &/.) mayormente en mayúsculas y corto (<=5)
    p.ej. "MIT", "UNAM", "I.B.M.".
    """
    txt = (s or "").strip()
    if not txt:
        return False
    # Un solo token (permitimos puntos y &), o token con puntos internos
    t = txt.replace(".", "").replace("&", "")
    if " " in t:
        return False
    # Longitud corta y mayoría mayúsculas
    return len(t) <= 5 and t.upper() == t and any(ch.isalpha() for ch in t)


def _prefer_person_org_from_relation(label: str) -> Tuple[bool, bool, bool]:
    """
    Dada una relación canónica (`label`), devuelve hints de tipado (preferencias):
    (prefer_person_on_subject, prefer_org_on_object, prefer_title_on_object)
    """
    l = (label or "").lower()
    # Relaciones de afiliación → (sujeto PERSON, objeto ORG)
    if l in ("works_at", "member_of", "collaboration_with", "acquired"):
        return (True, True, False)
    # Título/puesto → (sujeto PERSON, objeto TITLE)
    if l in ("title",):
        return (True, False, True)
    return (False, False, False)


def _slug_for_conflict(entity: NormalizedEntity) -> Optional[str]:
    """
    Devuelve una 'clave' neutral para detectar conflictos PERSON↔ORG en el mismo doc.
    - Para ORG: usa org_key/core.
    - Para PERSON: usa person_key.
    - Si no hay, usa name lower (si existe).
    """
    if not entity:
        return None
    if entity.type == "ORG":
        return (str(entity.attrs.get("org_key") or entity.attrs.get("org_core") or (entity.name or "")).lower() or None)
    if entity.type == "PERSON":
        return (str(entity.attrs.get("person_key") or (entity.name or "")).lower() or None)
    return None


# -----------------------------------------------------------------------------
# Label mapping + overrides de superficie
# -----------------------------------------------------------------------------

def _map_label_to_type(raw_label: Optional[str], canonical_label: Optional[str], surface: str) -> str:
    """
    a) Usa canonical_label si viene; si no, usa raw_label.
    b) Aplica overrides por superficie (conservadores):
       - dirección → LOC
       - ORG por sufijo/acrónimo → ORG
       - país/ciudad evidente → LOC
       - título común → TITLE
       - disciplina (matemáticas/analytics) si cayó como LOC → OTHER (o deja PERSON si fue PERSON y no es field-of-study)
    c) fallback a OTHER
    """
    base = (canonical_label or raw_label or "").strip().upper()
    t = LABEL2TYPE.get(base, "OTHER")

    # Overrides de superficie
    if is_address_like(surface):
        return "LOC"

    if is_org_like(surface) or _looks_like_acronym(surface):
        # "Acme", "MIT", "UNAM" → ORG (muy común en CVs/docs)
        return "ORG"

    if t in {"OTHER", "PERSON"} and is_country_or_city_lite(surface):
        return "LOC"

    if is_title_like(surface):
        return "TITLE"

    # Si cayó como LOC pero parece disciplina/campo → NO queremos LOC
    if t == "LOC" and is_field_of_study(surface):
        return "OTHER"

    return t


# -----------------------------------------------------------------------------
# Normalizer
# -----------------------------------------------------------------------------

class Normalizer:
    """Ejecutor de la etapa de Normalización."""

    def __init__(self, cfg: NormalizeConfig) -> None:
        self.cfg = cfg
        # Sufijos de organización: MX + EN para robustez cross-lingual
        self._org_suffixes: List[str] = list(dict.fromkeys(ORG_SUFFIXES_MX + ORG_SUFFIXES_EN))

    # -------------------------------------------------------------------------
    # API principal
    # -------------------------------------------------------------------------
    def run(self, mentions_doc: Dict[str, Any]) -> DocumentEntities:
        """
        Convierte un `DocumentMentions` (dict) a `DocumentEntities` (Pydantic).
        Estructura mínima esperada:
          {
            "doc_id": "...",
            "entities": [
               {"id": "E0", "text": "...", "label": "PERSON", "canonical_label": "PERSON",
                "conf": 0.72, "lang": "es" },
               ...
            ],
            "relations": [ ... ]   # opcional (si viene, la usamos para co-tipado defensivo)
          }
        """
        doc_id = str(mentions_doc.get("doc_id") or "DOC-UNKNOWN")
        # Copiamos listas para evitar mutar el dict de entrada fuera de aquí
        ents_in: List[Dict[str, Any]] = list(mentions_doc.get("entities") or [])
        rels_in: List[Dict[str, Any]] = list(mentions_doc.get("relations") or [])

        # 0) Co-tipado por relaciones (si hay), corrige labels de entities in-place (suavemente)
        if rels_in:
            self._apply_relation_typing_hints(ents_in, rels_in)

        counters = {
            "input_mentions": len(ents_in),
            "kept_protos": 0,     # prototipos válidos previos al merge
            "entities": 0,        # entidades finales post-merge
        }

        protos: List[NormalizedEntity] = []

        # 1) Construir protos desde las menciones (con defensas y overrides)
        for ent in ents_in:
            mention_id: str = str(ent.get("id") or "")
            text: str = str(ent.get("text") or "").strip()
            if not text:
                continue  # mención vacía → descartar

            conf: float = float(ent.get("conf") or 0.0)
            if conf < float(self.cfg.min_conf_keep):
                continue  # mención débil → descartar

            raw_label: Optional[str] = ent.get("label")
            canonical_label: Optional[str] = ent.get("canonical_label")
            lang: str = str(ent.get("lang") or "auto").lower()

            # Mapeo de label → tipo con overrides por superficie
            etype = _map_label_to_type(raw_label, canonical_label, text)

            # (extra) PERSON de un token sospechoso → ORG/OTHER
            if etype == "PERSON":
                toks = text.split()
                if len(toks) == 1:
                    if _looks_like_acronym(text) or is_org_like(text):
                        etype = "ORG"
                    elif is_field_of_study(text):
                        etype = "OTHER"

            # Construcción del *prototipo* (NormalizedEntity)
            proto = self._build_proto(doc_id, mention_id, text, etype, conf, lang)
            if proto is None:
                continue

            counters["kept_protos"] += 1
            protos.append(proto)

        # 2) Resolver conflictos PERSON↔ORG por clave canónica (preferir ORG)
        protos = self._prefer_org_over_person_when_conflict(doc_id, protos)

        # 3) MERGE/DEDUPE por tipo/clave
        merged = self._merge_all(protos)
        counters["entities"] = len(merged)

        meta = {
            "counters": counters,
            "config": {
                "date_locale": self.cfg.date_locale,
                "min_conf_keep": self.cfg.min_conf_keep,
                "merge_threshold": self.cfg.merge_threshold,
                "canonicalize": self.cfg.canonicalize,
                "default_currency": self.cfg.default_currency,
            },
        }
        return DocumentEntities(doc_id=doc_id, entities=merged, meta=meta)

    # -------------------------------------------------------------------------
    # Paso 0: co-tipado con relaciones (suave, defensivo)
    # -------------------------------------------------------------------------
    def _apply_relation_typing_hints(self, ents_in: List[Dict[str, Any]], rels_in: List[Dict[str, Any]]) -> None:
        """
        Si Mentions trae relaciones, úsalo para **sugerir** tipado más consistente:
        - works_at/member_of/...  ⇒ subject→PERSON, object→ORG
        - title                   ⇒ subject→PERSON, object→TITLE

        No borra info; solo ajusta `label`/`canonical_label` cuando difieren de la preferencia.
        """
        prefer_person: set[str] = set()
        prefer_org: set[str] = set()
        prefer_title: set[str] = set()

        for r in rels_in:
            lab = (r.get("canonical_label") or r.get("label") or "").lower()
            subj = r.get("subj_entity_id")
            obj  = r.get("obj_entity_id")
            if not subj or not obj:
                continue
            p_subj, p_obj_org, p_obj_title = _prefer_person_org_from_relation(lab)
            if p_subj:
                prefer_person.add(subj)
            if p_obj_org:
                prefer_org.add(obj)
            if p_obj_title:
                prefer_title.add(obj)

        # Reetiquetar suavemente entities de entrada
        for e in ents_in:
            eid = e.get("id")
            if not eid:
                continue
            lab = (e.get("canonical_label") or e.get("label") or "").upper()
            # si hay preferencia y el label actual difiere, ajustamos
            if eid in prefer_person and lab != "PERSON":
                e["label"] = e["canonical_label"] = "PERSON"
            if eid in prefer_org and lab != "ORG":
                e["label"] = e["canonical_label"] = "ORG"
            if eid in prefer_title and lab != "TITLE":
                e["label"] = e["canonical_label"] = "TITLE"

    # -------------------------------------------------------------------------
    # Construcción de prototipos (por tipo)
    # -------------------------------------------------------------------------
    def _build_proto(
        self, doc_id: str, mention_id: str, text: str, etype: str, conf: float, lang: str
    ) -> Optional[NormalizedEntity]:
        """Crea un NormalizedEntity *prototipo* a partir de una mención y su etype."""
        if etype == "DATE":
            # locale efectivo (auto→ heurística simple por lang)
            locale = self.cfg.date_locale if self.cfg.date_locale != "auto" else ("en" if lang.startswith("en") else "es")
            value_iso, precision = normalize_date(text, locale=locale)
            attrs = clean_attrs({"value_iso": value_iso, "precision": precision, "raw": text})
            nid = stable_id(doc_id, "DATE", (value_iso or text))
            return NormalizedEntity(
                id=nid, type="DATE", name=None, value=(value_iso or None),
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "MONEY":
            val, cur, src = normalize_money(text, default_currency=self.cfg.default_currency)
            cur = cur or "UNK"
            attrs = clean_attrs({"normalized_value": val, "currency": cur, "currency_source": src, "raw": text})
            key = f"{val if val is not None else 'NA'}::{cur}"
            nid = stable_id(doc_id, "MONEY", key)
            return NormalizedEntity(
                id=nid, type="MONEY", name=None,
                value=(str(val) if val is not None else None),
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "EMAIL":
            em = normalize_email(text)
            if not em:
                return None
            email_norm, user, domain = em
            attrs = clean_attrs({"email_norm": email_norm, "user": user, "domain": domain})
            nid = stable_id(doc_id, "EMAIL", email_norm)
            return NormalizedEntity(
                id=nid, type="EMAIL", name=email_norm, value=email_norm,
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "URL":
            norm = normalize_url(text)
            if not norm:
                return None
            url_norm, parts = norm
            attrs = clean_attrs({"url_norm": url_norm, **parts})
            nid = stable_id(doc_id, "URL", url_norm)
            return NormalizedEntity(
                id=nid, type="URL", name=url_norm, value=url_norm,
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "ORG":
            core, found, org_key = normalize_org(text, suffixes=self._org_suffixes)
            name = (core or text.strip().lower())
            attrs = clean_attrs({"org_core": core, "org_suffix": found, "org_key": org_key, "raw": text})
            nid = stable_id(doc_id, "ORG", org_key or core or text.lower())
            return NormalizedEntity(
                id=nid, type="ORG", name=name,
                value=None, attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "PERSON":
            # Defensa extra: si parece dirección, tratamos como LOC
            if is_address_like(text):
                return self._build_proto(doc_id, mention_id, text, "LOC", conf, lang)
            given, family, key = normalize_person(text)
            name = (f"{given} {family}".strip() or text.strip())
            attrs = clean_attrs({"given_name": given, "family_name": family, "person_key": key, "raw": text})
            nid = stable_id(doc_id, "PERSON", key or text.lower())
            return NormalizedEntity(
                id=nid, type="PERSON", name=name,
                value=None, attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "TITLE":
            raw = text.strip()
            canonical = TITLES_MAP.get(raw.lower(), TITLES_MAP.get(raw.lower().replace(".", ""), raw))
            attrs = clean_attrs({"title_raw": text, "title_canonical": canonical})
            nid = stable_id(doc_id, "TITLE", canonical.lower())
            return NormalizedEntity(
                id=nid, type="TITLE", name=canonical, value=canonical,
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "DEGREE":
            raw = text.strip()
            canonical = DEGREES_MAP.get(raw.lower(), DEGREES_MAP.get(raw.lower().replace(".", ""), raw))
            attrs = clean_attrs({"degree_raw": text, "degree_canonical": canonical})
            nid = stable_id(doc_id, "DEGREE", canonical.lower())
            return NormalizedEntity(
                id=nid, type="DEGREE", name=canonical, value=canonical,
                attrs=attrs, mentions=[mention_id], conf=conf
            )

        if etype == "LOC":
            # Nota: no hacemos geocoding; clave simple por lowercase
            key = text.strip().lower()
            # Disciplinas no deben caer a LOC → degradamos a OTHER
            if is_field_of_study(text):
                return self._build_proto(doc_id, mention_id, text, "OTHER", conf, lang)
            attrs = clean_attrs({"loc_key": key, "raw": text})
            nid = stable_id(doc_id, "LOC", key)
            return NormalizedEntity(
                id=nid, type="LOC", name=text.strip(),
                value=None, attrs=attrs, mentions=[mention_id], conf=conf
            )

        # Fallback OTHER: conserva superficie para auditoría
        attrs = clean_attrs({"raw": text})
        nid = stable_id(doc_id, "OTHER", text.strip().lower())
        return NormalizedEntity(
            id=nid, type="OTHER", name=text.strip(), value=None,
            attrs=attrs, mentions=[mention_id], conf=conf
        )

    # -------------------------------------------------------------------------
    # Paso 2: Conflictos PERSON↔ORG por clave canónica (preferir ORG)
    # -------------------------------------------------------------------------
    def _prefer_org_over_person_when_conflict(self, doc_id: str, protos: List[NormalizedEntity]) -> List[NormalizedEntity]:
        """
        Si en un mismo documento hay protos PERSON y ORG que comparten *slug canónico*
        (ej., 'acme' ↔ 'acme corp'), preferimos tipificar como ORG.
        - Detecta conflictos por slug (org_key/person_key/name).
        - Si solo existe PERSON dentro del grupo conflictivo, intentamos re-construir
          ese proto como ORG a partir de su 'raw' o 'name'.
        - Si ya existe un ORG en el grupo, descartamos los PERSON duplicados por la misma clave.

        Esto reduce falsos PERSON como “Acme”, “Analytics” en CVs.
        """
        # 2.1 Agrupar por slug neutral
        by_slug: Dict[str, List[NormalizedEntity]] = {}
        for e in protos:
            s = _slug_for_conflict(e)
            if not s:
                continue
            by_slug.setdefault(s, []).append(e)

        # 2.2 Resolver grupo a grupo
        adjusted: List[NormalizedEntity] = []
        seen_ids: set[str] = set()

        for slug, group in by_slug.items():
            types = {e.type for e in group}
            # Si el grupo trae ORG y PERSON: preferimos quedarnos con los ORG
            if "ORG" in types and "PERSON" in types:
                for e in group:
                    if e.type == "ORG":
                        adjusted.append(e)
                        seen_ids.add(e.id)
                # PERSON duplicados se descartan (mismo slug)
                continue

            # Si solo PERSON (pero slug parece de ORG) intentamos reconstruir como ORG
            if types == {"PERSON"}:
                # criterio: superficie org-like o acrónimo
                surf = group[0].attrs.get("raw") or group[0].name or ""
                if is_org_like(str(surf)) or _looks_like_acronym(str(surf)):
                    rebuilt = []
                    for e in group:
                        raw = str(e.attrs.get("raw") or e.name or "")
                        core, found, org_key = normalize_org(raw, suffixes=self._org_suffixes)
                        name = (core or raw.strip().lower())
                        attrs = clean_attrs({"org_core": core, "org_suffix": found, "org_key": org_key, "raw": raw})
                        nid = stable_id(doc_id, "ORG", org_key or core or raw.lower())
                        ne = NormalizedEntity(
                            id=nid, type="ORG", name=name, value=None,
                            attrs=attrs, mentions=list(e.mentions), conf=e.conf
                        )
                        rebuilt.append(ne)
                        seen_ids.add(e.id)
                    adjusted.extend(rebuilt)
                    continue  # grupo resuelto como ORG

            # No hubo conflicto o no aplica reconstrucción → conservar tal cual
            for e in group:
                adjusted.append(e)
                seen_ids.add(e.id)

        # Añadir protos que no tenían slug (no entraron a by_slug)
        for e in protos:
            if e.id not in seen_ids:
                adjusted.append(e)

        return adjusted

    # -------------------------------------------------------------------------
    # Paso 3: Merge por tipo/clave
    # -------------------------------------------------------------------------
    def _merge_all(self, protos: List[NormalizedEntity]) -> List[NormalizedEntity]:
        """
        Aplica *merge* por clave canónica, por tipo:
        - EMAIL/URL/DATE/MONEY/ID: claves “fuertes” → 1:1
        - ORG/PERSON/LOC/TITLE/DEGREE/PRODUCT/OTHER: claves por nombre/canonical
        """
        if not protos:
            return []

        out: List[NormalizedEntity] = []

        # --- Claves “fuertes” (valor inequívoco)
        email = [e for e in protos if e.type == "EMAIL"]
        url   = [e for e in protos if e.type == "URL"]
        date  = [e for e in protos if e.type == "DATE"]
        money = [e for e in protos if e.type == "MONEY"]
        idx   = [e for e in protos if e.type == "ID"]

        out.extend(merge_by_key(email, lambda e: str(e.attrs.get("email_norm") or "")))
        out.extend(merge_by_key(url,   lambda e: str(e.attrs.get("url_norm") or "")))
        out.extend(merge_by_key(date,  lambda e: str(e.attrs.get("value_iso") or "")))
        out.extend(merge_by_key(money, lambda e: f"{e.attrs.get('normalized_value','NA')}::{e.attrs.get('currency','UNK')}"))
        out.extend(merge_by_key(idx,   lambda e: str(e.attrs.get("normalized_id") or "")))

        # --- Nominales con clave derivada
        orgs    = [e for e in protos if e.type == "ORG"]
        people  = [e for e in protos if e.type == "PERSON"]
        locs    = [e for e in protos if e.type == "LOC"]
        titles  = [e for e in protos if e.type == "TITLE"]
        degrees = [e for e in protos if e.type == "DEGREE"]
        prods   = [e for e in protos if e.type == "PRODUCT"]
        others  = [e for e in protos if e.type == "OTHER"]

        out.extend(merge_by_key(orgs,    lambda e: str(e.attrs.get("org_key") or e.attrs.get("org_core") or e.name or "").lower()))
        out.extend(merge_by_key(people,  lambda e: str(e.attrs.get("person_key") or (e.name or "").lower())))
        out.extend(merge_by_key(locs,    lambda e: str(e.attrs.get("loc_key") or (e.name or "").lower())))
        out.extend(merge_by_key(titles,  lambda e: str(e.value or e.name or "").lower()))
        out.extend(merge_by_key(degrees, lambda e: str(e.value or e.name or "").lower()))
        out.extend(merge_by_key(prods,   lambda e: str(e.attrs.get("product_key") or (e.name or "").lower())))
        out.extend(merge_by_key(others,  lambda e: (e.name or "").lower()))

        return out
