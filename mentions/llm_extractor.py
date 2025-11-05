# -*- coding: utf-8 -*-
"""
llm_extractor.py — Extracción de menciones con procesamiento por lotes de chunks (BatchChunk Mode)

──────────────────────────────────────────────────────────────
📘 Descripción
──────────────────────────────────────────────────────────────
Versión extendida del extractor de menciones para el pipeline T2G.

Permite:
  • Procesar los chunks de un documento en **lotes (batches)** de tamaño configurable.
  • Mantener coherencia semántica inter-chunk sin saturar el contexto del modelo.
  • Conservar trazabilidad completa por lote, por chunk y por documento.
  • Unificar menciones y limpiar duplicados al final.

──────────────────────────────────────────────────────────────
⚙️ Parámetros clave
──────────────────────────────────────────────────────────────
  batch_size  → cantidad de chunks procesados juntos (por defecto: 3)
  MENTIONS_DEBUG → guarda los prompts generados en outputs_prompts/
"""

from __future__ import annotations
import os
import json
import datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from mentions.prompt_builder import build_prompt
from schema_selector.registry import REGISTRY, RegistryHelper
from mentions.schemas import MentionsConfig
from mentions.utils import preserve_order
from .metrics import attach_metrics_to_output
from mentions.llm_client import get_client

# ============================================================
# ⚙️ Inicialización
# ============================================================
client, meta = get_client()
print(f"[MENTIONS] Using provider={meta['provider']} | model={meta['model']}")

MENTIONS_DEBUG = os.getenv("MENTIONS_DEBUG", "0") == "1"
PROMPT_SAVE_DIR = os.getenv("PROMPT_SAVE_DIR", "outputs_prompts")
BATCH_SIZE = int(os.getenv("MENTIONS_BATCH_SIZE", "3"))  # 👈 configurable por entorno

# ============================================================
# 🔧 Utilidades internas
# ============================================================
def _coerce_json_array(raw: str) -> List[dict]:
    """Normaliza salida LLM a JSON list válida."""
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


def _merge_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Elimina duplicados exactos (text, type, domain) y promedia confianza."""
    merged = {}
    for m in mentions:
        key = (m.get("text", "").strip().lower(), m.get("type", ""), m.get("domain", ""))
        if key not in merged:
            merged[key] = {**m, "source_chunk": [m.get("source_chunk", "UNK")], "_count": 1}
        else:
            merged[key]["_count"] += 1
            merged[key]["confidence"] = round(
                (merged[key]["confidence"] + m.get("confidence", 0.85)) / 2, 3
            )
            sc = m.get("source_chunk", "UNK")
            if sc not in merged[key]["source_chunk"]:
                merged[key]["source_chunk"].append(sc)

    for m in merged.values():
        m["source_chunk"] = ", ".join(m["source_chunk"])
        m.pop("_count", None)
    return list(merged.values())


def _join_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Concatena texto de varios chunks para formar el batch."""
    lines = []
    for c in chunks or []:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        cid = c.get("chunk_id", "UNK")
        lines.append(f"[CHUNK {cid}] {t}")
    return "\n\n".join(lines)


def _retag_generic_mentions(mentions: List[Dict[str, Any]], text: str, helper: RegistryHelper, top_domains: List[str]) -> None:
    """Reetiqueta dominios genéricos según contexto local."""
    dhints = helper.hint_map(top_domains, alias_limit=40)
    text_low = text.lower()
    for m in mentions:
        if (m.get("domain") or "").lower() != "generic":
            continue
        sctx = m.get("text", "").lower()
        s = max(0, m.get("start_char", 0) - 50)
        e = min(len(text_low), m.get("end_char", 0) + 50)
        sctx += " " + text_low[s:e]
        best_dom, hits = None, 0
        for d, aliases in dhints.items():
            count = sum(1 for a in aliases if a in sctx)
            if count > hits:
                best_dom, hits = d, count
        if best_dom and hits >= 2:
            m["domain"] = best_dom
            m["confidence"] = min(0.95, m.get("confidence", 0.8))


# ============================================================
# 🚀 FUNCIÓN PRINCIPAL — Extracción BatchChunk
# ============================================================
def extract_mentions(chunks_glob: str, schema_dir: str, cfg: MentionsConfig) -> None:
    """Procesa cada documento por lotes de chunks y combina resultados."""
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
            chunks = chunk_data.get("chunks", [])
            if not chunks:
                print(f"[MENTIONS] ⚠️ Documento sin chunks: {doc_id}")
                continue

            # --- Cargar esquema del selector ---
            schema_path = Path(schema_dir) / f"{doc_id}_schema.json"
            if schema_path.exists():
                schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
            else:
                schema_data = {"doc": {"selected_schema": "generic", "top_domains": ["generic"]}}

            doc_meta = schema_data.get("doc", {})
            selected_schema = doc_meta.get("selected_schema", "generic")
            top_domains = doc_meta.get("top_domains", ["generic"])
            lead = (top_domains[0] if top_domains else "generic").lower()

            allowed = preserve_order(["generic"] + top_domains)
            allowed = [helper.normalize_domain(d) or d for d in allowed]
            print(f"[MENTIONS] Modelo {cfg.llm_model} | allowed_domains={allowed}")

            # ======================================================
            # 🔁 Procesamiento por lotes de chunks
            # ======================================================
            all_mentions: List[Dict[str, Any]] = []
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                batch_text = _join_chunks_for_prompt(batch_chunks)
                batch_id = i // BATCH_SIZE + 1

                print(f"[MENTIONS] ▶ Procesando batch {batch_id}/{total_batches} "
                      f"({len(batch_chunks)} chunks, {len(batch_text)} chars)")

                prompt = build_prompt(
                    schema_data=schema_data,
                    registry=REGISTRY,
                    helper=helper,
                    doc_text=batch_text,
                    alias_limit=12,
                )

                # Guardar prompt si debug activo
                if MENTIONS_DEBUG:
                    os.makedirs(PROMPT_SAVE_DIR, exist_ok=True)
                    fname = f"{doc_id}_batch-{batch_id}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
                    Path(PROMPT_SAVE_DIR, fname).write_text(prompt, encoding="utf-8")
                    print(f"[MENTIONS DEBUG] Prompt guardado: {fname}")

                # --- Llamada al modelo ---
                response = client.responses.create(
                    model=cfg.llm_model,
                    input=[
                        {"role": "system", "content": "Eres un analista experto en extracción semántica estructurada."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=cfg.temperature,
                    max_output_tokens=cfg.max_tokens,
                )
                raw = getattr(response, "output_text", "").strip()
                mentions_local = _coerce_json_array(raw)

                # --- Fallback ---
                if not mentions_local:
                    print(f"[MENTIONS] ⚠️ Sin menciones en batch {batch_id}, reintentando...")
                    prompt += (
                        "\n\n⚡ No detectaste entidades. "
                        "Identifica las más probables (explícitas o implícitas) "
                        "y devuélvelas en formato JSON (mínimo 5 menciones)."
                    )
                    response = client.responses.create(
                        model=cfg.llm_model,
                        input=[
                            {"role": "system", "content": "Eres un analista experto en extracción semántica estructurada."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=cfg.temperature + 0.3,
                        max_output_tokens=cfg.max_tokens,
                    )
                    raw = getattr(response, "output_text", "").strip()
                    mentions_local = _coerce_json_array(raw)

                # Normalizar y anotar procedencia
                for m in mentions_local:
                    m["source_chunk"] = ",".join([c.get("chunk_id", "UNK") for c in batch_chunks])
                    m["domain"] = (m.get("domain") or "generic").lower()
                    if m["domain"] not in allowed:
                        m["domain"] = "generic"
                    m["confidence"] = float(m.get("confidence", 0.85))
                all_mentions.extend(mentions_local)

                print(f"[MENTIONS]  Batch {batch_id} procesado ({len(mentions_local)} menciones)")

            # ======================================================
            # 🧹 Consolidación global
            # ======================================================
            mentions = _merge_mentions(all_mentions)
            joined_text = "\n".join(c.get("text", "") for c in chunks)
            _retag_generic_mentions(mentions, joined_text, helper, allowed)
            thr = float(cfg.confidence_threshold or 0.25)
            mentions = [m for m in mentions if m.get("confidence", 1.0) >= thr]

            result = {
                "doc_id": doc_id,
                "created_at": datetime.datetime.now().isoformat(),
                "mentions": mentions,
                "meta": {
                    "provider": meta["provider"],
                    "model": cfg.llm_model,
                    "schema_used": selected_schema,
                    "domain_detected": lead,
                    "batch_size": BATCH_SIZE,
                    "batches_total": total_batches,
                    "confidence_threshold": thr,
                    "registry_version": "1.0.0",
                },
            }

            result = attach_metrics_to_output(result)
            out_path = Path(cfg.outdir) / f"{doc_id}_mentions.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"[MENTIONS OK] {doc_id} → {out_path}  "
                  f"(batches={total_batches} | menciones={len(mentions)})")

        except Exception as e:
            print(f"[MENTIONS ERROR] {ch_file.name}: {e}")
