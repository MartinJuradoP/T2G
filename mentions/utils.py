# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_schema_for_doc(schema_dir: Path, doc_id: str) -> Path | None:
    for f in schema_dir.glob("*_schema.json"):
        if doc_id in f.name:
            return f
    return None


def build_prompt(text: str, schema_data: Dict[str, Any], registry_domains: List[str]) -> str:
    top_domains = schema_data.get("doc", {}).get("top_domains", [])
    selected_schema = schema_data.get("doc", {}).get("selected_schema", "generic_text_v1")
    explanation = schema_data.get("doc", {}).get("explanation", "")
    guidance = (
        f"Dominios detectados (doc-level): {', '.join(top_domains) or 'N/A'}.\n"
        f"Esquema sugerido: '{selected_schema}'.\n"
        f"Pistas: {explanation or '—'}\n"
    )
    # Instrucciones estrictas para salida JSON array
    format_rules = (
        "Devuelve EXCLUSIVAMENTE un JSON array válido de objetos con la forma:\n"
        "[{\"text\":\"...\",\"type\":\"...\",\"domain\":\"...\",\"confidence\":0.95,"
        "\"start_char\":null,\"end_char\":null,\"source_chunk\":\"...\"}, ...]\n"
        "No incluyas explicación, prosa ni backticks, SOLO el JSON.\n"
        "Si no hay entidades, devuelve []."
    )
    return (
        f"=== TEXTO ===\n{text}\n\n"
        f"=== CONTEXTO SCHEMA SELECTOR ===\n{guidance}\n"
        "Tarea: extrae TODAS las menciones de entidades relevantes conforme al esquema/dominio, "
        "incluyendo Contract/Party/Obligation/Penalty si aplica; de lo contrario usa genéricas "
        "(Person, Organization, Date, Amount, Location, ReferenceCode, URL, EmailAddress, PhoneNumber, SocialHandle, Hashtag, Emoji).\n\n"
        f"{format_rules}"
    )


def coerce_json_array(raw: str) -> List[dict]:
    """
    Intenta convertir el output del modelo a un JSON array:
    - Extrae el primer bloque con apariencia de JSON array.
    - Si no hay, intenta parsear directamente.
    """
    s = raw.strip()
    # si viene con backticks, quítalos
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    # busca el primer array JSON
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        s = m.group(0)

    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        # Si vino como obj con clave "mentions"
        if isinstance(data, dict) and "mentions" in data and isinstance(data["mentions"], list):
            return data["mentions"]
    except Exception:
        pass
    return []

def preserve_order(seq):
    """Deduplica preservando el orden (case-insensitive y sin None)."""
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        k = str(x).strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

