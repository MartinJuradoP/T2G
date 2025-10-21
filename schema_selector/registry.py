# -*- coding: utf-8 -*-
"""
registry.py — Ontología de dominios y entidades para Adaptive Schema Selector.

Este módulo define la ontología base utilizada por el subsistema
**Adaptive Schema Selector** dentro de la pipeline T2G.

Su función principal es proporcionar un **catálogo estructurado de dominios**
(legal, financiero, médico, tecnológico, etc.) con sus correspondientes
entidades, atributos y relaciones, para permitir que el selector determine
de forma automática qué esquema de extracción aplicar a cada documento.

Cada dominio contiene:
- **Aliases:** palabras clave o expresiones asociadas al dominio, usadas
  para la detección contextual en los textos.
- **Negative Aliases:** términos que, si aparecen, penalizan la selección
  de ese dominio (ayudan a reducir falsos positivos).
- **Stopwords:** palabras genéricas que no aportan valor discriminativo.
- **Entity Types:** definiciones de entidades con sus atributos relevantes.
- **Relation Types:** relaciones semánticas entre entidades del mismo dominio.
- **Schema Name:** nombre del esquema que se usará para la extracción NER/RE
  cuando este dominio sea detectado.

Características clave:
----------------------
- **Extensible:** se pueden añadir o modificar dominios sin afectar al resto del sistema.
- **Modular:** cada dominio encapsula su propio conjunto de entidades y relaciones.
- **Auditable:** permite inspeccionar la cobertura léxica y solapamientos entre dominios.
- **Compatibilidad total:** puede ser consumido por el Adaptive Schema Selector
  y otros componentes que requieran información de contexto o estructura semántica.

El dominio **generic** se incluye siempre como fallback universal y actúa como
esquema de respaldo cuando el documento no puede asociarse claramente a un dominio
específico.
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional
from pydantic import BaseModel, Field, model_validator
from collections import defaultdict
import json
import pandas as pd

# Importamos las estructuras base
from .schemas import (
    OntologyDomain,
    OntologyRegistry,
    EntityTypeDef,
    AttributeDef,
    RelationTypeDef
)

# ===========================================================================
#  Clases extendidas con validación y trazabilidad
# ===========================================================================

class OntologyDomain(OntologyDomain):
    """Extiende OntologyDomain con soporte para alias negativos, stopwords y auditoría."""

    stopwords: Set[str] = Field(default_factory=set, description="Palabras genéricas a ignorar.")
    negative_aliases: Set[str] = Field(default_factory=set, description="Términos que penalizan la selección del dominio.")
    weight: float = Field(default=1.0, description="Peso relativo del dominio.")
    schema_name: str = Field(default="generic_text_v1", description="Nombre del esquema asociado al dominio.")
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_aliases(self) -> "OntologyDomain":
        """Valida y normaliza alias, stopwords y negativos."""
        self.aliases = sorted(set(a.strip().lower() for a in self.aliases if a))
        self.stopwords = set(w.lower().strip() for w in self.stopwords)
        self.negative_aliases = set(w.lower().strip() for w in self.negative_aliases)

        # Evita conflictos entre listas
        overlap = set(self.aliases) & set(self.negative_aliases)
        if overlap:
            raise ValueError(f"Alias conflictivos en dominio '{self.domain}': {overlap}")
        return self

    def describe(self, max_entities: int = 3) -> str:
        """Devuelve un resumen legible del dominio y sus componentes."""
        ents = ", ".join(e.name for e in self.entity_types[:max_entities])
        rels = ", ".join(r.name for r in self.relation_types[:max_entities])
        return (
            f"🔹 {self.domain.upper()} — {len(self.aliases)} alias "
            f"({len(self.entity_types)} entidades, {len(self.relation_types)} relaciones)\n"
            f"  Ejemplos entidades: {ents or 'N/A'}\n"
            f"  Relaciones: {rels or 'N/A'}\n"
        )


class OntologyRegistry(OntologyRegistry):
    """Ontología global con validación y funciones de auditoría."""

    domains: List[OntologyDomain] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_domains(self) -> "OntologyRegistry":
        names = [d.domain.lower() for d in self.domains]
        if len(names) != len(set(names)):
            raise ValueError(f"Dominios duplicados detectados: {names}")
        return self

    # ----------------------------------------------------------------------
    # Funciones de auditoría y control de calidad
    # ----------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        """Muestra un resumen tabular de todos los dominios definidos."""
        data = []
        for d in self.domains:
            data.append({
                "Domain": d.domain,
                "#Aliases": len(d.aliases),
                "#Stopwords": len(d.stopwords),
                "#Negatives": len(d.negative_aliases),
                "#Entities": len(d.entity_types),
                "#Relations": len(d.relation_types),
                "Weight": d.weight,
                "Schema": d.schema_name
            })
        return pd.DataFrame(data).sort_values(by="Domain")

    def conflicts_matrix(self) -> pd.DataFrame:
        """Matriz de solapamiento de alias entre dominios (para detectar ambigüedad léxica)."""
        doms = [d.domain for d in self.domains]
        overlap = defaultdict(dict)
        for d1 in self.domains:
            for d2 in self.domains:
                if d1.domain == d2.domain:
                    overlap[d1.domain][d2.domain] = 1.0
                else:
                    inter = len(set(d1.aliases) & set(d2.aliases))
                    total = len(set(d1.aliases) | set(d2.aliases))
                    overlap[d1.domain][d2.domain] = inter / max(1, total)
        return pd.DataFrame(overlap).T.loc[doms, doms]

    def export_json(self, path: str = "registry_audit.json") -> None:
        """Exporta la ontología completa a JSON (para auditoría o versionado)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        print(f"✅ Ontología exportada a {path}")
# ===========================================================================
# 🩺 MEDICAL Domain
# ===========================================================================
MEDICAL = OntologyDomain(
    domain="medical",
    schema_name="medical_note_v1",
    weight=0.95,
    aliases=[
        # ⚕️ Más específicos, menos genéricos
        "diagnóstico", "síntoma", "tratamiento", "terapia",
        "hospital", "médico", "doctor", "paciente", "prescripción",
        "cirugía", "fármaco", "medicina", "receta", "clínico",
        "enfermedad", "historial médico", "análisis clínico",
        # Inglés
        "diagnosis", "symptom", "therapy", "treatment",
        "hospital", "doctor", "patient", "prescription",
        "surgery", "drug", "medication", "clinical trial", "medical record"
    ],
    negative_aliases={
        # ⚠️ Palabras que anulan el contexto médico
        "financial", "contract", "invoice", "policy", "software",
        "product", "order", "payment", "investment", "bank", "insurance",
        "agreement", "article", "press", "review", "customer", "user"
    },
    stopwords={
        # 🧹 Palabras neutras que no ayudan al contexto
        "caso", "registro", "documento", "data", "study", "report",
        "analysis", "record", "case", "note"
    },
    entity_types=[
        EntityTypeDef(
            name="Disease",
            description="Illness or pathology affecting a patient.",
            aliases=["enfermedad", "patología", "disease", "condition"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="icd_code", type="code")
            ]
        ),
        EntityTypeDef(
            name="Symptom",
            description="Sign or indication of a disease.",
            aliases=["síntoma", "signo", "symptom"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Drug",
            description="Medication or compound used in treatment.",
            aliases=["fármaco", "medicamento", "drug"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="dose", type="string")
            ]
        ),
        EntityTypeDef(
            name="Patient",
            description="Person receiving medical treatment.",
            aliases=["paciente", "patient"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="age", type="number"),
                AttributeDef(name="gender", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_symptom", head="Disease", tail="Symptom"),
        RelationTypeDef(name="treated_with", head="Disease", tail="Drug"),
        RelationTypeDef(name="attended_by", head="Patient", tail="Doctor"),
    ],
)


# ===========================================================================
# ⚖️ LEGAL Domain
# ===========================================================================
LEGAL = OntologyDomain(
    domain="legal",
    schema_name="legal_contract_v1",
    weight=1.0,
    aliases=[
        "contrato", "cláusula", "firma", "notario", "juicio", "sentencia",
        "demanda", "acuerdo", "penalización", "contract", "agreement", "clause",
        "signature", "trial", "lawsuit", "court", "penalty", "liability", "claim","jurisdicción","legal","law","compliance"
    ],
    negative_aliases={"hospital", "doctor", "disease","stocks","finance","invoice"},
    stopwords={"documento", "registro", "caso"},
    entity_types=[
        EntityTypeDef(
            name="Party",
            description="Person or organization in a legal agreement.",
            aliases=["parte", "firmante", "persona", "empresa", "party"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="role")]
        ),
        EntityTypeDef(
            name="Contract",
            description="Legal document defining terms and obligations.",
            aliases=["contrato", "acuerdo", "contract"],
            attributes=[
                AttributeDef(name="effective_date", type="date"),
                AttributeDef(name="term", type="string"),
                AttributeDef(name="jurisdiction", type="string")
            ]
        ),
        EntityTypeDef(
            name="Obligation",
            description="Duty or responsibility from a contract.",
            aliases=["obligación", "responsabilidad", "duty"],
            attributes=[AttributeDef(name="description")]
        ),
        EntityTypeDef(
            name="Penalty",
            description="Legal or monetary sanction for breach.",
            aliases=["multa", "sanción", "penalty", "fine"],
            attributes=[
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="currency", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="binds", head="Contract", tail="Party"),
        RelationTypeDef(name="imposes", head="Contract", tail="Obligation"),
        RelationTypeDef(name="penalizes", head="Obligation", tail="Penalty"),
    ],
)

# ===========================================================================
# 💰 FINANCIAL Domain
# ===========================================================================
FINANCIAL = OntologyDomain(
    domain="financial",
    schema_name="financial_tx_v2",
    weight=1.0,
    aliases=[
        "finanzas", "factura", "transacción", "pago", "banco", "seguro", "mercado",
        "divisa", "acción", "presupuesto", "bolsa", "cotización", "exchange",
        "finance", "investment", "stock", "currency", "insurance", "loan",
        "interest", "policy", "equity", "earnings", "revenue", "ingresos","P&L","Margin","ROI",
        "financials", "trading", "comercio","earnings per share", "profit margin", "EPS", "Dividend","foreign exchange","FX","mutual fund","hedge fund"
    ],
    negative_aliases={"hospital", "doctor", "contract", "disease"},
    stopwords={"monto", "total", "fecha"},
    entity_types=[
        EntityTypeDef(
            name="Invoice",
            description="Document for transaction of goods or services.",
            aliases=["factura", "invoice"],
            attributes=[
                AttributeDef(name="invoice_number"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="currency", type="string")
            ]
        ),
        EntityTypeDef(
            name="Transaction",
            description="Movement of money between accounts.",
            aliases=["pago", "transferencia", "transaction"],
            attributes=[
                AttributeDef(name="transaction_id"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="date", type="date")
            ]
        ),
        EntityTypeDef(
            name="Account",
            description="Financial account identifier.",
            aliases=["cuenta", "account", "bank"],
            attributes=[
                AttributeDef(name="account_number"),
                AttributeDef(name="bank", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="paid_by", head="Transaction", tail="Account"),
        RelationTypeDef(name="covered_by", head="Invoice", tail="Policy"),
    ],
)

# ===========================================================================
# 💻 TECH REVIEW Domain
# ===========================================================================
TECH = OntologyDomain(
    domain="tech_review",
    schema_name="tech_review_v1",
    weight=0.9,
    aliases=[
        "benchmark", "reseña", "modelo", "gpu", "cpu", "latencia", "precisión",
        "tecnología", "hardware", "software", "review", "performance", "specs",
        "accuracy", "ai", "model", "technology", "data", "inference", "training","news","artificial intelligence","machine learning","deep learning","neural network","Digital Transformation","emerging technology"
    ],
    negative_aliases={"contract", "disease", "invoice"},
    stopwords={"comparativa", "prueba", "resultado"},
    entity_types=[
        EntityTypeDef(
            name="Product",
            description="Hardware or software under evaluation.",
            aliases=["producto", "modelo", "device", "software", "hardware"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="vendor"),
                AttributeDef(name="category")
            ]
        ),
        EntityTypeDef(
            name="Metric",
            description="Performance or quality measure.",
            aliases=["latencia", "tiempo", "fps", "precisión", "metric", "accuracy"],
            attributes=[
                AttributeDef(name="metric_name"),
                AttributeDef(name="value", type="number"),
                AttributeDef(name="unit", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_metric", head="Product", tail="Metric"),
    ],
)

# ===========================================================================
# 🛒 E-COMMERCE Domain
# ===========================================================================
ECOMMERCE = OntologyDomain(
    domain="ecommerce",
    schema_name="ecommerce_order_v1",
    weight=1.0,
    aliases=[
        "carrito", "pedido", "compra", "precio", "producto", "cliente",
        "order", "purchase", "product", "customer", "store", "review", "seller"
    ],
    negative_aliases={"hospital", "contract", "disease"},
    stopwords={"artículo", "comentario", "item"},
    entity_types=[
        EntityTypeDef(
            name="Order",
            description="Commercial purchase order.",
            aliases=["pedido", "orden", "order"],
            attributes=[
                AttributeDef(name="order_id"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="payment_method", type="string")
            ]
        ),
        EntityTypeDef(
            name="Product",
            description="Item available for sale or review.",
            aliases=["producto", "item", "product"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="brand"),
                AttributeDef(name="category")
            ]
        ),
        EntityTypeDef(
            name="Review",
            description="Customer opinion about a product.",
            aliases=["reseña", "comentario", "review"],
            attributes=[
                AttributeDef(name="rating", type="number"),
                AttributeDef(name="text")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="contains", head="Order", tail="Product"),
        RelationTypeDef(name="reviewed_by", head="Product", tail="Review"),
    ],
)

# ===========================================================================
# 🐾 VETERINARY Domain
# ===========================================================================
VETERINARY = OntologyDomain(
    domain="veterinary",
    schema_name="veterinary_case_v1",
    weight=0.9,
    aliases=[
        "animal", "mascota", "veterinario", "síntoma", "tratamiento",
        "ganado", "pet", "vet", "cattle", "disease", "vacuna", "zoonosis"
    ],
    negative_aliases={"contrato", "invoice", "court"},
    stopwords={"caso", "registro", "historia"},
    entity_types=[
        EntityTypeDef(
            name="Animal",
            description="Animal or pet under veterinary care.",
            aliases=["mascota", "animal", "pet", "dog", "cat"],
            attributes=[
                AttributeDef(name="species"),
                AttributeDef(name="breed")
            ]
        ),
        EntityTypeDef(
            name="Disease",
            description="Condition affecting an animal.",
            aliases=["enfermedad", "zoonosis", "disease"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Treatment",
            description="Medication or vaccine for an animal.",
            aliases=["tratamiento", "vacuna", "treatment", "vaccine"],
            attributes=[AttributeDef(name="name")]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="treated_with", head="Animal", tail="Treatment"),
        RelationTypeDef(name="has_disease", head="Animal", tail="Disease"),
    ],
)

# ===========================================================================
# 🌎 GEOPOLITICAL Domain
# ===========================================================================
GEO = OntologyDomain(
    domain="geopolitical",
    schema_name="geopolitical_event_v1",
    weight=0.8,
    aliases=[
        "país", "ciudad", "estado", "frontera", "conflicto", "tratado",
        "country", "city", "state", "border", "conflict", "treaty", "agreement",
        "nación", "territorio", "guerra", "alianza", "summit"
    ],
    negative_aliases={"contract", "hospital", "invoice"},
    stopwords={"caso", "zona", "región"},
    entity_types=[
        EntityTypeDef(
            name="Country",
            description="Nation or sovereign state.",
            aliases=["país", "nación", "country"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="iso_code")
            ]
        ),
        EntityTypeDef(
            name="City",
            description="Urban or municipal entity.",
            aliases=["ciudad", "municipio", "city"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Event",
            description="Political or international event.",
            aliases=["conflicto", "tratado", "acuerdo", "event", "summit", "war"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="date", type="date")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="located_in", head="City", tail="Country"),
        RelationTypeDef(name="involves", head="Event", tail="Country"),
    ],
)

# ===========================================================================
# 📰 REVIEWS & NEWS Domain
# ===========================================================================
REVIEWS = OntologyDomain(
    domain="reviews_and_opinions",
    schema_name="review_service_v3",
    weight=0.9,
    aliases=[
        # Estructura general de reseñas
        "review", "reseña", "comentario", "opinión", "feedback", "valoración",
        "rating", "calificación", "testimonio", "experiencia",
        # Polaridad positiva
        "good", "great", "excellent", "amazing", "fantastic", "wonderful",
        "awesome", "perfect", "delicious", "tasty", "friendly", "clean",
        "fast", "fresh", "nice", "love", "best", "quick", "affordable",
        # Polaridad negativa
        "bad", "terrible", "awful", "disgusting", "horrible", "rude",
        "slow", "dirty", "cold", "expensive", "worst", "unfriendly",
        "inattentive", "burned", "stale", "mediocre", "poor",
        # Aspectos del servicio
        "food", "comida", "drink", "bebida", "coffee", "pizza", "burger",
        "service", "servicio", "staff", "mesero", "waiter", "waitress",
        "price", "precio", "atmosphere", "ambiente", "decor", "lugar",
        "place", "restaurant", "restaurante", "menu", "porciones",
        "cleanliness", "higiene", "customer", "cliente", "attitude",
        "experience", "recomendado", "recommend",
        # Complementos del contexto Yelp
        "parking", "music", "bar", "dessert", "postre", "drink", "beer",
        "wine", "happy hour", "reservation", "table", "crowded"
    ],
    # Filtros y palabras neutrales
    negative_aliases={"policy", "terms", "privacy", "contract"},
    stopwords={"texto", "nota", "contenido"},
    entity_types=[
        EntityTypeDef(
            name="Review",
            description="Opinión de un usuario sobre un servicio, producto o establecimiento.",
            aliases=["review", "reseña", "comentario", "opinión", "feedback"],
            attributes=[
                AttributeDef(name="review_id"),
                AttributeDef(name="stars", type="number", description="Calificación (1–5 estrellas)."),
                AttributeDef(name="sentiment", type="string", description="Polaridad general del texto."),
                AttributeDef(name="subjectivity", type="number", description="Grado de opinión personal."),
                AttributeDef(name="language", type="string"),
            ],
        ),
        EntityTypeDef(
            name="ServiceAspect",
            description="Categoría específica evaluada dentro de la reseña.",
            aliases=[
                "food", "comida", "drink", "bebida", "service", "servicio", "staff",
                "mesero", "waiter", "waitress", "ambiente", "price", "precio",
                "decor", "cleanliness", "menu", "producto", "entorno"
            ],
            attributes=[
                AttributeDef(name="sentiment", type="string", description="Polaridad del aspecto."),
                AttributeDef(name="intensity", type="number", description="Fuerza emocional de la expresión."),
            ],
        ),
        EntityTypeDef(
            name="Emotion",
            description="Expresión emocional asociada a la experiencia (positiva o negativa).",
            aliases=[
                "happy", "sad", "angry", "satisfied", "frustrated",
                "love", "hate", "amazing", "disgusted", "pleasant"
            ],
            attributes=[
                AttributeDef(name="valence", type="number"),
                AttributeDef(name="arousal", type="number"),
            ],
        ),
        EntityTypeDef(
            name="Business",
            description="Negocio o establecimiento evaluado.",
            aliases=[
                "restaurant", "restaurante", "bar", "café", "hotel", "tienda",
                "local", "empresa", "servicio"
            ],
            attributes=[
                AttributeDef(name="category", type="string"),
                AttributeDef(name="location", type="string"),
            ],
        ),
    ],
    relation_types=[
        RelationTypeDef(name="written_by", head="Review", tail="Reviewer"),
        RelationTypeDef(name="about", head="Review", tail="Business"),
        RelationTypeDef(name="mentions_aspect", head="Review", tail="ServiceAspect"),
        RelationTypeDef(name="expresses_emotion", head="Review", tail="Emotion"),
    ],
    notes=(
        "Optimizado para reseñas cortas de plataformas como Yelp, Google Reviews o TripAdvisor. "
        "Los embeddings se calculan por agrupamiento semántico de alias, "
        "con centroides basados en polaridad y aspectos de servicio, "
        "para evitar sesgos hacia otros dominios y mejorar la coherencia de categorización."
    )
)


# ===========================================================================
# 🧩 GENERIC Domain (universal fallback actualizado)
# ===========================================================================
GENERIC = OntologyDomain(
    domain="generic",
    schema_name="generic_text_v2",
    weight=0.4,
    aliases=[
        # Conceptos transversales
        "general", "documento", "texto", "registro", "mensaje", "post", "comentario",
        "tweet", "publicación", "content", "note", "text", "comment", "article",
        "file", "record", "entry"
    ],
    negative_aliases=set(),  # No penaliza ningún otro dominio
    stopwords={"contenido", "archivo", "formato", "texto", "document"},
    entity_types=[
        # ------------------------------------------------------
        # Identidad y entidades clásicas (fallback semántico)
        # ------------------------------------------------------
        EntityTypeDef(
            name="Person",
            description="Nombre de una persona física o usuario genérico.",
            aliases=["persona", "nombre", "user", "autor", "writer"]
        ),
        EntityTypeDef(
            name="Organization",
            description="Entidad corporativa, institución, o grupo social.",
            aliases=["empresa", "institución", "organization", "compañía", "grupo"]
        ),
        EntityTypeDef(
            name="Date",
            description="Fecha explícita o calendario formal.",
            aliases=["fecha", "día", "año", "mes", "date"]
        ),
        EntityTypeDef(
            name="DateExpression",
            description="Referencia temporal relativa expresada en lenguaje natural.",
            aliases=["ayer", "hoy", "mañana", "semana", "mes", "próximo", "pasado"]
        ),
        EntityTypeDef(
            name="Location",
            description="Lugar o referencia geográfica general.",
            aliases=["ubicación", "ciudad", "país", "address", "lugar", "sitio"]
        ),
        EntityTypeDef(
            name="Amount",
            description="Cantidad numérica con posible valor monetario o métrico.",
            aliases=["monto", "precio", "valor", "cantidad", "amount", "total"]
        ),
        EntityTypeDef(
            name="Currency",
            description="Símbolo o abreviatura de moneda.",
            aliases=["usd", "eur", "mxn", "gbp", "¥", "₿", "$", "€"]
        ),

        # ------------------------------------------------------
        # Entidades de comunicación digital y trazabilidad
        # ------------------------------------------------------
        EntityTypeDef(
            name="URL",
            description="Dirección o enlace web, completo o parcial (http, https, www).",
            aliases=["url", "link", "website", "sitio web", "enlace"]
        ),
        EntityTypeDef(
            name="EmailAddress",
            description="Dirección de correo electrónico.",
            aliases=["correo", "email", "mail", "e-mail"]
        ),
        EntityTypeDef(
            name="PhoneNumber",
            description="Número telefónico o de contacto.",
            aliases=["teléfono", "número", "celular", "phone", "contacto"]
        ),
        EntityTypeDef(
            name="SocialHandle",
            description="Identificador o mención de usuario en redes sociales (@usuario).",
            aliases=["usuario", "@", "handle", "cuenta", "perfil"]
        ),
        EntityTypeDef(
            name="Hashtag",
            description="Etiqueta temática usada en redes sociales (#tema).",
            aliases=["hashtag", "#"]
        ),
        EntityTypeDef(
            name="ReferenceCode",
            description="Código de referencia, folio o identificador alfanumérico.",
            aliases=["id", "folio", "ticket", "ref", "código", "identificador"]
        ),
        EntityTypeDef(
            name="FileReference",
            description="Referencia a nombre o ruta de archivo local o remoto.",
            aliases=["archivo", ".pdf", ".docx", ".xls", ".csv", "documento"]
        ),

        # ------------------------------------------------------
        # Expresiones emocionales o contextuales
        # ------------------------------------------------------
        EntityTypeDef(
            name="Emoji",
            description=(
                "Símbolo Unicode que expresa emoción, reacción o idea (😊, 🚀, ❤️, etc.). "
                "Se detecta mediante rango Unicode o regex, no mediante aliases explícitos."
            ),
            aliases=["emoji", "emoticon", "carita", "símbolo"],
            attributes=[
                AttributeDef(name="symbol", type="string"),
                AttributeDef(
                    name="category",
                    type="string",
                    description="Categoría semántica: emotion, object, activity, symbol, flag."
                )
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(
            name="mentions",
            head="Person",
            tail="SocialHandle",
            description="Una persona menciona o referencia a otra cuenta digital."
        ),
        RelationTypeDef(
            name="links_to",
            head="Document",
            tail="URL",
            description="Un texto o registro contiene un enlace a un sitio web."
        ),
        RelationTypeDef(
            name="refers_to",
            head="Document",
            tail="ReferenceCode",
            description="Documento o texto que incluye una referencia o código identificador."
        ),
        RelationTypeDef(
            name="includes_emoji",
            head="Document",
            tail="Emoji",
            description="Texto que contiene un símbolo emocional o expresivo."
        ),
    ],
    notes=(
        "El dominio GENÉRICO actúa como fallback universal. Detecta patrones transversales "
        "comunes en textos no clasificables (emails, publicaciones, notas, logs, etc.). "
        "Incluye soporte para URLs, correos, teléfonos, hashtags, menciones, emojis y referencias."
    ),
)


# ===========================================================================
# 🌐 GLOBAL REGISTRY
# ===========================================================================
REGISTRY = OntologyRegistry(
    domains=[
        MEDICAL, LEGAL, FINANCIAL, TECH,
        ECOMMERCE, VETERINARY, GEO, REVIEWS, GENERIC
    ]
)


