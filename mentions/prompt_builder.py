# -*- coding: utf-8 -*-
"""
prompt_builder.py — Generación adaptativa de prompts para extracción de menciones

Integra de forma aumentativa:
  1) Dominios detectados por el Adaptive Schema Selector (top_domains)
  2) Dominio implícito del selected_schema (p. ej. 'legal_contract_v1' → 'legal')
  3) Fallback genérico (siempre incluido)
  4) Ontología del Registry (entidades, relaciones, aliases)
Esto se puede mejorar para un entrenamiento o inferencia más precisa y contextualizada.
Ontología combinada (Registry):
{ontology_block}

Señales y pesos del selector:
- Señales: {', '.join(signals) or 'no registradas'}
- Pesos: {json.dumps(weights, ensure_ascii=False)}
- Explicación: {explanation}
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
    alias_limit: int = 0
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
   
    
    top_domains_raw = doc.get("top_domains", []) or ["generic"]
    selected_schema = doc.get("selected_schema", "generic")
    schema_domain = _infer_schema_domain(selected_schema)
    domains_combined = preserve_order(
      ["generic"] + top_domains_raw + ([schema_domain] if schema_domain else [])
    )
  
    top_domains = helper.match_domains(domains_combined)
    lead_domain = top_domains[0] if top_domains else "generic"

    # 2) Esquemas activos y entidades sugeridas por el selector
    all_schemas = _schemas_from_selector(schema_data)
    selector_ents = _selector_entities(schema_data)

    # 3) Bloques de ontología (aumentativos) sólo para dominios activos
    #ontology_block = helper.prompt_blocks_for(top_domains, alias_limit=alias_limit)
    ontology_block = helper.prompt_blocks_for(
    top_domains,
    alias_limit=alias_limit,
    include_entities=True,
    include_relations=False,
    include_aliases=True
)


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
    "text": "Entity A",
    "type": "Organization",
    "domain": "generic",
    "confidence": 0.95,
    "start_char": 10,
    "end_char": 19,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "2024-05-10",
    "type": "Date",
    "domain": "generic",
    "confidence": 0.93,
    "start_char": 55,
    "end_char": 65,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "$1,000,000",
    "type": "Amount",
    "domain": "financial",
    "confidence": 0.92,
    "start_char": 120,
    "end_char": 130,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "ABC Index",
    "type": "Index",
    "domain": "financial",
    "confidence": 0.90,
    "start_char": 200,
    "end_char": 209,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "XZY-100",
    "type": "Ticker",
    "domain": "financial",
    "confidence": 0.88,
    "start_char": 220,
    "end_char": 227,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "Service Contract",
    "type": "Contract",
    "domain": "legal",
    "confidence": 0.87,
    "start_char": 310,
    "end_char": 326,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "Product Model 5",
    "type": "Product",
    "domain": "tech_review",
    "confidence": 0.86,
    "start_char": 400,
    "end_char": 414,
    "source_chunk": "CHUNK-001"
  },
  {
    "text": "Entity A reported earnings of $1,000,000",
    "type": "reports",
    "domain": "financial",
    "confidence": 0.90,
    "start_char": 450,
    "end_char": 495,
    "source_chunk": "CHUNK-001"
  }
]"""
    # 8) Prompt final (se concatena el ejemplo JSON como literal; NO hay llaves sin escapar)
    header = f"""
Eres un analista experto en extracción de entidades y relaciones.Con mucho conocimiento en la creación de Grafos de Conocimiento (Knowledge Graphs) y ontologías.

El selector clasificó el documento con dominio principal **{lead_domain}** 
y los siguientes dominios complementarios (en orden de relevancia descendente): {', '.join(top_domains[1:])}.
Interpreta que el dominio principal aporta el contexto dominante,
pero las entidades y relaciones pueden provenir de cualquiera de los dominios listados,
sin excluir ninguno.

Ontología combinada (Registry):
{ontology_block}



Instrucciones de extracción:

1) Extrae todas las entidades y relaciones **explícitas o semiexplícitas** que aparezcan en el texto 
   y que correspondan a los tipos definidos en la ontología combinada (Registry) 
   para los dominios activos listados arriba.
   Considera semiexplícitas aquellas que se expresan mediante símbolos, unidades, nombres técnicos,
   convenciones del dominio o abreviaturas. 
   No inventes entidades nuevas, pero tampoco omitas las que estén presentes
   de forma implícita en frases o contextos típicos del dominio.


2) Usa los tipos de entidad y relación definidos en los dominios activos del Registry 
   como **esquemas estructurales**, no como filtros de texto.
   Considera que cada entidad está definida por su nombre, descripción y atributos.
   Los aliases sirven solo como ejemplos léxicos, pero debes reconocer menciones que encajen
   con el concepto y atributos descritos para ese tipo, aunque el texto use otras palabras.


3) Los dominios activos ({', '.join(top_domains)}) pueden coexistir y compartir tipos o relaciones. 
   Usa las entidades y relaciones definidas en todos ellos, sin excluir ninguno. 
   Clasifica cada mención en el dominio más coherente con su contexto semántico,
   y usa el dominio principal **{lead_domain}** únicamente como guía de desambiguación,
   nunca como restricción.


4 Usa activamente los aliases del Registry para identificar equivalencias:
   - Fechas, valores, cantidades o porcentajes → tipos `Date`, `Amount`, `Percentage`, `Measurement`.
   - Símbolos bursátiles o índices (p. ej., "^SPX", "S&P 500", "^DJI") → tipos `Ticker`, `Index`.
   - Nombres de organizaciones o compañías (p. ej., "Southern Co.", "Zacks Investment Research") → tipos `Company`, `Organization`.
   - Si hay productos, contratos o métricas, usa los tipos de sus dominios asociados (p. ej., `Product`, `Contract`, `Metric`).

5 Si el texto expresa una relación semántica entre entidades (por ejemplo: “reportó ingresos de”, “pertenece a”, “anunció la compra de”),
   crea un objeto adicional con `"type"` igual al nombre de la relación definida en el Registry 
   (por ejemplo: `"reports"`, `"belongs_to_index"`, `"acquired"`) y el `"domain"` correspondiente.

6 Cada mención debe tener evidencia textual directa. Incluye:
   - `"start_char"` y `"end_char"` con posiciones aproximadas.
   - `"source_chunk"` con el identificador del bloque en que aparece.
   No incluyas inferencias sin texto o información externa.

7 Evita duplicados triviales: si una mención aparece repetida con el mismo `text`, `type` y `domain`, conserva solo una.

8 Devuelve **únicamente un JSON array válido**, con objetos que sigan esta estructura:
   [
     {{"text":"...","type":"...","domain":"...","confidence":0.95,
       "start_char":null,"end_char":null,"source_chunk":"..."}}
   ]
   No incluyas comentarios, backticks, ni texto adicional fuera del JSON.
   Si no hay menciones válidas, devuelve [].
""".strip()

    text_section = f"\n\n Texto a analizar:\n{safe_text}" if safe_text else ""

    prompt = header + "\n\n" + example_json + text_section
    return prompt
