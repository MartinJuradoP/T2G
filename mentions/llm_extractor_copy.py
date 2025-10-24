# -*- coding: utf-8 -*-
"""
llm_extractor.py — Mentions (enriquecido con schema + top_domains + registry)
"""

from __future__ import annotations
import os
import json
import datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from mentions.prompt_builder import build_prompt
from schema_selector.registry import REGISTRY, RegistryHelper
from mentions.schemas import MentionsConfig
from mentions.utils import preserve_order
from .metrics import attach_metrics_to_output


# ------------------ Setup OpenAI ------------------
# ------------------ Setup LLM Client ------------------
from mentions.llm_client import get_client

client, meta = get_client()
print(f"[MENTIONS] Using provider={meta['provider']} | model={meta['model']}")

# Debug
MENTIONS_DEBUG = os.getenv("MENTIONS_DEBUG", "0") == "1"
PROMPT_SAVE_DIR = os.getenv("PROMPT_SAVE_DIR", "outputs_prompts")

# ------------------ Helpers ------------------


def _domain_hint_map(helper: RegistryHelper, top_domains: List[str], alias_limit: int = 40) -> Dict[str, List[str]]:
    """Mapa dominio → pistas léxicas para reetiquetar genéricas."""
    return helper.hint_map(top_domains, alias_limit=alias_limit)


def _retag_generic_mentions(
    mentions: List[Dict[str, Any]],
    doc_text: str,
    helper: RegistryHelper,
    top_domains: List[str],
    min_hits: int = 2
) -> None:
    """
    Re-etiqueta menciones con domain='generic' si el contexto del span
    contiene suficientes alias de un dominio top (señal ontológica).
    """
    if not mentions or not top_domains:
        return

    dhints = _domain_hint_map(helper, top_domains)
    text_low = doc_text.lower() if isinstance(doc_text, str) else ""

    for m in mentions:
        if (m.get("domain") or "").lower() != "generic":
            continue

        # contexto local
        sctx = (m.get("text") or "").lower()
        if m.get("start_char") is not None and m.get("end_char") is not None and text_low:
            s = max(0, int(m["start_char"]) - 50)
            e = min(len(text_low), int(m["end_char"]) + 50)
            sctx = text_low[s:e]

        best_dom, best_hits = None, 0
        for dom, aliases in dhints.items():
            hits = sum(1 for a in aliases if a and a in sctx)
            if hits > best_hits:
                best_dom, best_hits = dom, hits

        if best_dom and best_hits >= min_hits:
            m["domain"] = best_dom
            m["confidence"] = float(min(0.95, m.get("confidence", 0.7)))


def _merge_overlapping_spans(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Limpia duplicados exactos por (text,type,domain)."""
    seen = set()
    out = []
    for m in mentions:
        key = (m.get("text", "").strip(), m.get("type", ""), m.get("domain", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _coerce_json_array(raw: str) -> List[dict]:
    """
    Intenta convertir el output del modelo a un JSON array:
    - Recorta backticks.
    - Extrae el primer bloque con apariencia de JSON array.
    - Acepta {"mentions":[...]} como fallback.
    """
    s = (raw or "").strip()
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("mentions"), list):
            return data["mentions"]
    except Exception:
        pass
    return []


def _join_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Concatena chunks con su id para aportar contexto al LLM."""
    lines = []
    for c in chunks or []:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        cid = c.get("chunk_id", "UNK")
        lines.append(f"[CHUNK {cid}] {t}")
    return "\n\n".join(lines)


# ------------------ Main ------------------

def extract_mentions(chunks_glob: str, schema_dir: str, cfg: MentionsConfig) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(Path().glob(chunks_glob))
    if not chunk_files:
        print(f"[MENTIONS] ⚠️ No se encontraron archivos con patrón: {chunks_glob}")
        return

    helper = RegistryHelper(REGISTRY)

    for ch_file in chunk_files:
        try:
            chunk_data = json.loads(Path(ch_file).read_text(encoding="utf-8"))
            doc_id = chunk_data.get("doc_id", ch_file.stem)

            # Carga selector output real si existe
            schema_path = Path(schema_dir) / f"{doc_id}_schema.json"
            if schema_path.exists():
                schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
            else:
                schema_data = {"doc": {"selected_schema": "generic", "top_domains": ["generic"]}}

            # Texto fuente (todos los chunks)
            doc_text = _join_chunks_for_prompt(chunk_data.get("chunks", []))

            # Guardrail: si lead != generic y selected_schema es genérico, forzar el del registry
            doc_meta = schema_data.get("doc", {}) or {}
            selected_schema = doc_meta.get("selected_schema") or "generic"
            top_domains = (doc_meta.get("top_domains") or ["generic"])
            lead = (top_domains[0] if top_domains else "generic").lower()
            if lead != "generic" and selected_schema.startswith("generic"):
                from schema_selector.selector import get_schema_for_domain
                forced_schema = get_schema_for_domain(lead, REGISTRY)
                schema_data.setdefault("doc", {})["selected_schema"] = forced_schema
                print(f"[MENTIONS] Forzado selected_schema → {forced_schema} (lead={lead})")

            # Prompt enriquecido con registry + schema + top_domains + texto
            prompt = build_prompt(
                schema_data=schema_data,
                registry=REGISTRY,
                helper=helper,
                doc_text=doc_text,
                alias_limit=15
            )
            if not isinstance(prompt, str):
                prompt = "".join(prompt) if isinstance(prompt, (list, tuple)) else str(prompt)

            # Dominios permitidos post-proceso
            raw_allowed = ["generic"] + (top_domains or [])
            allowed = preserve_order(raw_allowed)
            allowed = [helper.normalize_domain(d) or d for d in allowed]

            #allowed = set(_allowed_domains(helper, top_domains))
            


            # --- DEBUG: dump prompt a disco para inspección ---
            if MENTIONS_DEBUG:
                os.makedirs(PROMPT_SAVE_DIR, exist_ok=True)
                save_path = os.path.join(
                    PROMPT_SAVE_DIR,
                    f"{doc_id}_prompt_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
                )
                header = [
                    f"[doc_id] {doc_id}",
                    f"[selected_schema] {schema_data.get('doc',{}).get('selected_schema')}",
                    f"[top_domains] {top_domains}",
                    f"[allowed_domains] {allowed}",
                    f"[text_len] {len(doc_text)}",
                    "-" * 80,
                ]
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(header))
                    f.write("\n")
                    f.write(prompt)
                print(f"[MENTIONS DEBUG] Prompt → {save_path}")
                print(f"[MENTIONS DEBUG] Prompt(head): {prompt[:280].replace(chr(10),' ')} ...")

            print(f"[MENTIONS] Modelo {cfg.llm_model} | allowed_domains={allowed}")

            # Llamada principal al modelo
            response = client.responses.create(
                model=cfg.llm_model,
                input=[
                    {"role": "system", "content": "Eres un analista experto en extracción semántica estructurada."},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_tokens
            )
            raw = response.output_text.strip() if hasattr(response, "output_text") else ""
            mentions = _coerce_json_array(raw)

            # ⚡ Fallback inferencial: si no devuelve nada, reintenta con instrucción reforzada
            if not mentions and len(doc_text) > 50:
                print(f"[MENTIONS] ⚠️ Sin menciones explícitas; reintentando en modo inferencial...")
                prompt += (
                    "\n\n⚡ No detectaste entidades. "
                    "Ahora identifica las más probables (explícitas o implícitas) y devuélvelas "
                    "en un JSON array de al menos 5 menciones tentativas con tipos, dominios y confianza estimada."
                )
                response = client.responses.create(
                    model=cfg.llm_model,
                    input=[
                        {"role": "system", "content": "Eres un analista experto en extracción de entidades legales y relaciones contractuales."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=cfg.temperature + 0.3,
                    max_output_tokens=cfg.max_tokens
                )
                raw = response.output_text.strip() if hasattr(response, "output_text") else ""
                mentions = _coerce_json_array(raw)

            # Filtro por confianza mínima
            try:
                thr = float(cfg.confidence_threshold)
            except Exception:
                thr = 0.25
            mentions = [m for m in mentions if float(m.get("confidence", 1.0)) >= thr]

            # Normaliza dominios a la lista permitida
            for m in mentions:
                d = (m.get("domain") or "generic").lower().strip()
                m["domain"] = d if d in allowed else "generic"

            # Re-etiqueta 'generic' si hay evidencia local (ontológica)
            _retag_generic_mentions(mentions, doc_text, helper, list(allowed))

            # Limpieza de duplicados
            mentions = _merge_overlapping_spans(mentions)

            # Empaqueta resultado
            result = {
                "doc_id": doc_id,
                "created_at": datetime.datetime.now().isoformat(),
                "mentions": mentions,
                "meta": {
                    "provider": "openai",
                    "model": cfg.llm_model,
                    "schema_used": schema_data.get("doc", {}).get("selected_schema", "generic"),
                    "domain_detected": (schema_data.get("doc", {}).get("top_domains") or ["generic"])[0],
                    "confidence_threshold": thr,
                    "registry_version": "1.0.0",
                }
            }

            # Adjunta métricas (⚠️ debe devolver un dict)
            result = attach_metrics_to_output(result)

            out_file = outdir / f"{doc_id}_mentions.json"
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"[MENTIONS OK] {doc_id} → {out_file}  (n={len(mentions)})")

        except Exception as e:
            print(f"[MENTIONS ERROR] {ch_file.name}: {e}")
