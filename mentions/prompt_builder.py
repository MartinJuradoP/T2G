# -*- coding: utf-8 -*-
"""
prompt_builder.py — Generación adaptativa de prompts para extracción de menciones

Integra de forma aumentativa:
  1) Dominios detectados por el Adaptive Schema Selector (top_domains)
  2) Dominio implícito del selected_schema (p. ej. 'legal_contract_v1' → 'legal')
  3) Fallback genérico (siempre incluido)
  4) Ontología del Registry (entidades, relaciones, aliases)
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional

from schema_selector.registry import RegistryHelper, OntologyRegistry
from mentions.utils import preserve_order


# ===========================================================================
#  Utilidades internas
# ===========================================================================
def _schemas_from_selector(sd: Dict[str, Any]) -> List[str]:
    """Recupera todos los esquemas activos (doc + chunks)."""
    doc = sd.get("doc", {}) or {}
    chunks = sd.get("chunks", []) or []
    schemas = [doc.get("selected_schema", "generic")]
    schemas += [c.get("selected_schema") for c in chunks if c.get("selected_schema")]
    return sorted({s for s in schemas if s})


def _selector_entities(sd: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extrae entidades detectadas por el selector con score > 0."""
    doc = sd.get("doc", {}) or {}
    out: Dict[str, List[str]] = {}
    for d in doc.get("domain_scores", []) or []:
        dom = d.get("domain")
        ents = [et["type_name"] for et in d.get("entity_type_scores", []) if et.get("score", 0) > 0]
        if dom and ents:
            out[dom] = ents
    return out


def _infer_schema_domain(schema_name: str) -> Optional[str]:
    """
    Deriva el dominio implícito desde el nombre del esquema.
    Ej: 'legal_contract_v1' -> 'legal'
    """
    if not schema_name:
        return None
    name = schema_name.lower()
    m = re.match(r"([a-z_]+?)_text", name)
    if m:
        return m.group(1)
    if "generic" in name:
        return "generic"
    # fallback simple para plantillas no estándar
    for known in [
        "legal", "medical", "financial", "tech_review", "tech", "ecommerce",
        "veterinary", "geopolitical", "reviews_and_opinions", "reviews"
    ]:
        if known in name:
            return "tech_review" if known == "tech" else (
                "reviews_and_opinions" if known == "reviews" else known
            )
    return None


# ===========================================================================
#  Prompt principal
# ===========================================================================
def build_prompt(
    schema_data: Dict[str, Any],
    registry: OntologyRegistry,
    helper: Optional[RegistryHelper] = None,
    doc_text: Optional[str] = None,
    alias_limit: int = 15
) -> str:
    """
    Construye un prompt contextual y aumentativo combinando:
      - top_domains detectados por el selector
      - dominio implícito del selected_schema
      - fallback 'generic'
      - bloques ontológicos del Registry (entidades/relaciones/aliases)
      - texto de los chunks (si se proporciona)
    """
    helper = helper or RegistryHelper(registry)
    doc = schema_data.get("doc", {}) or {}
    meta = schema_data.get("meta", {}) or {}

    # 1) Determinar dominios activos (top_domains + schema_domain + generic)
    #Funcion para mantener el orden y eliminar duplicados
    """def preserve_order(seq):
      seen = set()
      return [x for x in seq if not (x in seen or seen.add(x))]"""
    
    top_domains_raw = doc.get("top_domains", []) or ["generic"]
    selected_schema = doc.get("selected_schema", "generic")
    schema_domain = _infer_schema_domain(selected_schema)
    #domains_combined = list(set(top_domains_raw + ([schema_domain] if schema_domain else []) + ["generic"]))
    #domains_combined = preserve_order(["generic"] + top_domains_raw + ([schema_domain] if schema_domain else []))
    domains_combined = preserve_order(
      ["generic"] + top_domains_raw + ([schema_domain] if schema_domain else [])
    )
  
    top_domains = helper.match_domains(domains_combined)
    lead_domain = top_domains[0] if top_domains else "generic"

    # 2) Esquemas activos y entidades sugeridas por el selector
    all_schemas = _schemas_from_selector(schema_data)
    selector_ents = _selector_entities(schema_data)

    # 3) Bloques de ontología (aumentativos) sólo para dominios activos
    ontology_block = helper.prompt_blocks_for(top_domains, alias_limit=alias_limit)

    # 4) Señales/explicación del selector (trazabilidad)
    signals = doc.get("signals_used", meta.get("signals", [])) or []
    weights = doc.get("weights_used", meta.get("weights", {})) or {}
    explanation = doc.get("explanation", "") or "(sin explicación)"

    # 5) Texto del documento (opcional, recomendado)
    safe_text = (doc_text or "").strip()

    # 6) Formateo
    domain_list_for_json = ", ".join(top_domains)
    schemas_list = "\n".join(f"  • {s}" for s in all_schemas) or "  • generic"
    selector_entities_str = (
        "\n".join(f"  • {d}: {', '.join(ents)}" for d, ents in selector_ents.items())
        or "  • (sin entidades detectadas por el selector)"
    )

    # 7) Ejemplo JSON (como string literal, SIN f-string, para evitar llaves interpretadas)
    example_json = """[
  {
    "text": "Juan Pérez",
    "type": "Person",
    "domain": "generic",
    "confidence": 0.92,
    "start_char": 15,
    "end_char": 25,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "Contrato de Servicios",
    "type": "Contract",
    "domain": "legal",
    "confidence": 0.90,
    "start_char": 60,
    "end_char": 82,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "Paracetamol",
    "type": "Drug",
    "domain": "medical",
    "confidence": 0.87,
    "start_char": 120,
    "end_char": 131,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "México",
    "type": "Location",
    "domain": "geopolitical",
    "confidence": 0.85,
    "start_char": 240,
    "end_char": 246,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "$2,500.00",
    "type": "Amount",
    "domain": "financial",
    "confidence": 0.83,
    "start_char": 300,
    "end_char": 310,
    "source_chunk": "CHUNK-001"
  }
]"""


    # 8) Prompt final (se concatena el ejemplo JSON como literal; NO hay llaves sin escapar)
    header = f"""
Eres un analista experto en extracción de entidades y relaciones.Con mucho conocimiento en la creación de Grafos de Conocimiento (Knowledge Graphs) y ontologías.

El selector clasificó el documento con dominio líder **{lead_domain}**, 
y también identificó o sugirió los dominios: {', '.join(top_domains)}.

Esquemas activos:
{schemas_list}

Entidades detectadas por el selector (indicativas, no limitativas):
{selector_entities_str}

Ontología combinada (Registry):
{ontology_block}

Señales y pesos del selector:
- Señales: {', '.join(signals) or 'no registradas'}
- Pesos: {json.dumps(weights, ensure_ascii=False)}
- Explicación: {explanation}

Instrucciones (modo enriquecido y jerárquico):

1) Analiza cuidadosamente el texto y **extrae todas las menciones de entidades, valores, conceptos o relaciones relevantes**, no omitas ninguna.
   presentes o inferibles según el contexto. No te limites a un número fijo: devuelve tantas menciones como sean necesarias
   para representar de forma completa la información semántica del fragmento.
2) Usa los **tipos de entidad y relación coherentes con los dominios y esquemas listados arriba**.
   El orden de los dominios refleja **prioridad contextual y jerarquía semántica**:
   los primeros dominios tienen más peso para clasificar menciones ambiguas o de contexto compartido.
3) Cuando un mismo tipo o entidad pueda pertenecer a varios dominios (por ejemplo, `Organization` en *legal* y *geopolitical*),
   selecciona el dominio principal según el **orden jerárquico de los dominios activos**, pero conserva la riqueza contextual.
   No excluyas dominios secundarios si aportan matices o subtipos complementarios.
4) Si el texto no contiene entidades explícitas, **sugiere las más probables o implícitas** según los dominios y el esquema activo,
   priorizando aquellas incluidas o relacionadas en la ontología del Registry.
5) Si existen relaciones semánticas evidentes (por ejemplo, “A contrata a B”, “firma de convenio”, “pago de monto”),
   inclúyelas como entidades de tipo `"Relation"` o `"Action"`, según su naturaleza y contexto de dominio.
6) Si el fragmento no encaja claramente en ningún dominio o sugiere uno nuevo, usa `"domain": "generic"`,
   pero evita degradar menciones de dominios conocidos si su contexto lo justifica.
7) Devuelve **únicamente un JSON array válido** con todas las menciones detectadas o inferidas.
   No incluyas comentarios, backticks, ni texto adicional fuera del JSON.

""".strip()

    text_section = f"\n\n Texto a analizar:\n{safe_text}" if safe_text else ""

    prompt = header + "\n\n" + example_json + text_section
    return prompt
