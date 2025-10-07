# -*- coding: utf-8 -*-
"""
registry.py — Ontología de dominios y entidades para Adaptive Schema Selector.

Define una ontología de dominios comunes (medical, legal, identity, tech_review,
financial, ecommerce, veterinary, geopolitical, reviews_and_news, generic)
que sirven como catálogo base para la detección de entidades y relaciones
en documentos heterogéneos.

Cada dominio:
- Tiene alias (palabras clave para detección de contexto).
- Define entidades (EntityTypeDef) con atributos relevantes.
- Define relaciones entre entidades.
- Es extensible y modular: se pueden añadir nuevos dominios fácilmente.

El dominio genérico (generic) se incluye siempre como fallback universal.
"""

from __future__ import annotations
from .schemas import (
    OntologyRegistry,
    OntologyDomain,
    EntityTypeDef,
    AttributeDef,
    RelationTypeDef,
)

# ===========================================================================
# 🩺 MEDICAL Domain
# ===========================================================================
MEDICAL = OntologyDomain(
    domain="medical",
    aliases=[
        "salud", "clínico", "médico", "paciente", "síntoma", "tratamiento",
        "hospital", "doctor", "medicina", "fármaco", "insulina", "diagnóstico",
        "enfermedad", "health", "clinical", "medical", "patient", "disease",
        "treatment", "therapy", "hospital", "drug", "vaccine"
    ],
    entity_types=[
        EntityTypeDef(
            name="Disease",
            description="Illness or pathology affecting a patient.",
            aliases=["enfermedad", "patología", "dx", "disease", "condition"],
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
            aliases=["fármaco", "medicamento", "medicine", "drug"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="dose", type="string")
            ]
        ),
        EntityTypeDef(
            name="Treatment",
            description="Therapeutic or surgical procedure.",
            aliases=["tratamiento", "terapia", "therapy"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="LabTest",
            description="Laboratory analysis or diagnostic test.",
            aliases=["análisis", "laboratorio", "test", "glucosa", "hemoglobina", "lab"],
            attributes=[
                AttributeDef(name="test_name"),
                AttributeDef(name="value", type="number"),
                AttributeDef(name="unit", type="string")
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
        RelationTypeDef(name="has_symptom", head="Disease", tail="Symptom", description="Disease presents a symptom."),
        RelationTypeDef(name="treated_with", head="Disease", tail="Drug", description="Disease treated with a drug."),
        RelationTypeDef(name="confirmed_by", head="Disease", tail="LabTest", description="Disease confirmed by a lab test."),
        RelationTypeDef(name="attended_by", head="Patient", tail="Doctor", description="Patient attended by a doctor."),
    ],
)

# ===========================================================================
# ⚖️ LEGAL Domain
# ===========================================================================
LEGAL = OntologyDomain(
    domain="legal",
    aliases=[
        "contrato", "cláusula", "firma", "notario", "juicio", "sentencia",
        "demanda", "acuerdo", "penalización", "contract", "agreement", "clause",
        "signature", "trial", "lawsuit", "court", "penalty", "liability", "claim"
    ],
    entity_types=[
        EntityTypeDef(
            name="Party",
            description="Person or organization in a legal agreement.",
            aliases=["parte", "firmante", "persona", "empresa", "party", "signatory"],
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
            aliases=["obligación", "responsabilidad", "duty", "liability"],
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
        RelationTypeDef(name="binds", head="Contract", tail="Party", description="Contract binds a party."),
        RelationTypeDef(name="imposes", head="Contract", tail="Obligation", description="Contract imposes an obligation."),
        RelationTypeDef(name="penalizes", head="Obligation", tail="Penalty", description="Obligation leads to a penalty."),
    ],
)

# ===========================================================================
# 🪪 IDENTITY Domain
# ===========================================================================
IDENTITY = OntologyDomain(
    domain="identity",
    aliases=[
        "identificación", "id", "dni", "ine", "rfc", "curp", "pasaporte", "licencia",
        "identity", "passport", "id card", "driver license", "ssn", "nif"
    ],
    entity_types=[
        EntityTypeDef(
            name="Person",
            description="Individual identified by a name or document.",
            aliases=["nombre", "persona", "person", "individual"],
            attributes=[AttributeDef(name="full_name")]
        ),
        EntityTypeDef(
            name="IDDocument",
            description="Document identifying a person.",
            aliases=["identificación", "id", "passport", "license", "document"],
            attributes=[
                AttributeDef(name="id_type"),
                AttributeDef(name="id_number", type="id")
            ]
        ),
        EntityTypeDef(
            name="Address",
            description="Physical address or residence.",
            aliases=["domicilio", "dirección", "address", "residence"],
            attributes=[
                AttributeDef(name="line1"),
                AttributeDef(name="city"),
                AttributeDef(name="state"),
                AttributeDef(name="postal_code")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="identified_by", head="Person", tail="IDDocument", description="Person identified by an ID."),
        RelationTypeDef(name="resides_at", head="Person", tail="Address", description="Person resides at an address."),
    ],
)

# ===========================================================================
# 💻 TECH REVIEW Domain
# ===========================================================================
TECH = OntologyDomain(
    domain="tech_review",
    aliases=[
        "benchmark", "reseña", "modelo", "gpu", "cpu", "latencia", "precisión", "tecnología",
        "hardware", "software", "review", "performance", "specs", "accuracy", "ai", "model","technology","data"
    ],
    entity_types=[
        EntityTypeDef(
            name="Product",
            description="Hardware or software under evaluation.",
            aliases=["producto", "modelo", "device", "software", "hardware"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="vendor"), AttributeDef(name="category")]
        ),
        EntityTypeDef(
            name="Metric",
            description="Performance or quality measure.",
            aliases=["latencia", "tiempo", "fps", "precisión", "metric", "speed", "accuracy"],
            attributes=[
                AttributeDef(name="metric_name"),
                AttributeDef(name="value", type="number"),
                AttributeDef(name="unit", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_metric", head="Product", tail="Metric", description="Product evaluated by a metric."),
    ],
)

# ===========================================================================
# 💰 FINANCIAL Domain
# ===========================================================================
FINANCIAL = OntologyDomain(
    domain="financial",
    aliases=[
        "finanzas", "factura", "transacción", "pago", "banco", "seguro", "mercado", "divisa",
        "acción", "presupuesto", "bolsa", "cotización", "exchange", "finance", "investment",
        "stock", "currency", "insurance", "loan", "interest", "policy","equity","patrimonio","banco","bank","EPS","acciones","dividendos","dividends"
        ,"earnings","revenue","ingresos","financials","financieros","trading","comercio","trader","inversor","investor"
    ],
    entity_types=[
        EntityTypeDef(
            name="Invoice",
            description="Document for transaction of goods or services.",
            aliases=["factura", "recibo", "invoice"],
            attributes=[
                AttributeDef(name="invoice_number"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="currency", type="string")
            ]
        ),
        EntityTypeDef(
            name="Transaction",
            description="Movement of money between accounts.",
            aliases=["pago", "transferencia", "transaction", "deposit"],
            attributes=[
                AttributeDef(name="transaction_id"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="date", type="date")
            ]
        ),
        EntityTypeDef(
            name="Account",
            description="Financial account identifier.",
            aliases=["cuenta", "iban", "account", "bank"],
            attributes=[
                AttributeDef(name="account_number"),
                AttributeDef(name="bank", type="string")
            ]
        ),
        EntityTypeDef(
            name="Policy",
            description="Insurance or coverage policy.",
            aliases=["póliza", "seguro", "policy", "insurance"],
            attributes=[
                AttributeDef(name="policy_id"),
                AttributeDef(name="coverage", type="string")
            ]
        ),
        EntityTypeDef(
            name="StockIndicator",
            description="Market or stock index performance indicator.",
            aliases=["acción", "índice", "stock", "ticker", "equity"],
            attributes=[
                AttributeDef(name="symbol"),
                AttributeDef(name="price", type="number"),
                AttributeDef(name="change_percent", type="number")
            ]
        ),
        EntityTypeDef(
            name="ExchangeRate",
            description="Conversion rate between currencies.",
            aliases=["tipo de cambio", "exchange rate", "forex"],
            attributes=[
                AttributeDef(name="from_currency"),
                AttributeDef(name="to_currency"),
                AttributeDef(name="rate", type="number")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="paid_by", head="Transaction", tail="Account", description="Transaction paid by account."),
        RelationTypeDef(name="covered_by", head="Invoice", tail="Policy", description="Invoice covered by a policy."),
        RelationTypeDef(name="quoted_in", head="StockIndicator", tail="ExchangeRate", description="Stock quoted in a currency."),
    ],
)

# ===========================================================================
# 🛒 E-COMMERCE Domain
# ===========================================================================
ECOMMERCE = OntologyDomain(
    domain="ecommerce",
    aliases=[
        "carrito", "pedido", "compra", "precio", "producto", "cliente",
        "order", "purchase", "product", "customer", "store"
    ],
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
        RelationTypeDef(name="contains", head="Order", tail="Product", description="Order contains a product."),
        RelationTypeDef(name="reviewed_by", head="Product", tail="Review", description="Product reviewed by a customer."),
    ],
)

# ===========================================================================
# 🐾 VETERINARY Domain
# ===========================================================================
VETERINARY = OntologyDomain(
    domain="veterinary",
    aliases=[
        "animal", "mascota", "veterinario", "síntoma", "tratamiento", "ganado",
        "pet", "vet", "cattle", "disease"
    ],
    entity_types=[
        EntityTypeDef(
            name="Animal",
            description="Animal or pet under veterinary care.",
            aliases=["mascota", "animal", "pet", "dog", "cat"],
            attributes=[AttributeDef(name="species"), AttributeDef(name="breed")]
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
        RelationTypeDef(name="treated_with", head="Animal", tail="Treatment", description="Animal treated with a treatment."),
        RelationTypeDef(name="has_disease", head="Animal", tail="Disease", description="Animal diagnosed with a disease."),
    ],
)

# ===========================================================================
# 🌎 GEOPOLITICAL Domain
# ===========================================================================
GEO = OntologyDomain(
    domain="geopolitical",
    aliases=[
        "país", "ciudad", "estado", "frontera", "conflicto", "tratado",
        "country", "city", "state", "border", "conflict", "treaty", "agreement"
    ],
    entity_types=[
        EntityTypeDef(
            name="Country",
            description="Nation or sovereign state.",
            aliases=["país", "nación", "country"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="iso_code")]
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
        RelationTypeDef(name="located_in", head="City", tail="Country", description="City located in a country."),
        RelationTypeDef(name="involves", head="Event", tail="Country", description="Event involves a country."),
    ],
)

# ===========================================================================
# 📰 REVIEWS & NEWS Domain
# ===========================================================================
REVIEWS = OntologyDomain(
    domain="reviews_and_news",
    aliases=[
        "review", "reseña", "comentario", "opinión", "feedback", "news",
        "noticia", "artículo", "prensa", "report", "headline","calificación","score"
    ],
    entity_types=[
        EntityTypeDef(
            name="Review",
            description="Opinion or evaluation by a user or critic.",
            aliases=["reseña", "review", "comentario"],
            attributes=[
                AttributeDef(name="review_id"),
                AttributeDef(name="stars", type="number"),
                AttributeDef(name="date", type="date"),
                AttributeDef(name="sentiment", type="string")
            ]
        ),
        EntityTypeDef(
            name="NewsArticle",
            description="Published news article or report.",
            aliases=["noticia", "artículo", "news"],
            attributes=[
                AttributeDef(name="title"),
                AttributeDef(name="publisher"),
                AttributeDef(name="date", type="date"),
                AttributeDef(name="sentiment", type="string")
            ]
        ),
        EntityTypeDef(
            name="MarketEvent",
            description="Event affecting market or economy.",
            aliases=["crisis", "subida", "baja", "market event", "announcement"],
            attributes=[
                AttributeDef(name="event_type"),
                AttributeDef(name="impact")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="authored_by", head="Review", tail="Person", description="Review authored by a person."),
        RelationTypeDef(name="published_by", head="NewsArticle", tail="Organization", description="News published by an organization."),
        RelationTypeDef(name="describes_event", head="NewsArticle", tail="MarketEvent", description="News describes a market event."),
    ],
)

# ===========================================================================
# 🧩 GENERIC Domain (Always included)
# ===========================================================================
GENERIC = OntologyDomain(
    domain="generic",
    aliases=[
        "general", "documento", "texto", "registro", "file", "document", "record", "text"
    ],
    entity_types=[
        EntityTypeDef(name="Person", aliases=["persona", "nombre", "user", "patient"]),
        EntityTypeDef(name="Organization", aliases=["empresa", "institución", "organization", "company"]),
        EntityTypeDef(name="IDNumber", aliases=["id", "rfc", "curp", "ssn", "identifier"]),
        EntityTypeDef(name="Date", aliases=["fecha", "día", "mes", "año", "date"]),
        EntityTypeDef(name="Location", aliases=["ubicación", "ciudad", "estado", "país", "address"]),
        EntityTypeDef(name="PhoneNumber", aliases=["teléfono", "móvil", "celular", "phone"]),
        EntityTypeDef(name="Email", aliases=["correo", "email", "mail"]),
        EntityTypeDef(name="Amount", aliases=["monto", "precio", "costo", "valor", "usd", "mxn", "amount"]),
    ],
)

# ===========================================================================
# 🌐 GLOBAL REGISTRY
# ===========================================================================
REGISTRY = OntologyRegistry(
    domains=[
        MEDICAL, LEGAL, IDENTITY, TECH, FINANCIAL,
        ECOMMERCE, VETERINARY, GEO, REVIEWS, GENERIC
    ]
)
