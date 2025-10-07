# -*- coding: utf-8 -*-
"""
registry.py — Ontología de dominios y entidades para Adaptive Schema Selector.

Define una ontología de dominios comunes (medical, legal, identity, tech_review,
financial, ecommerce, veterinary, geopolitical, reviews_and_news, generic)
que sirven como catálogo base para detección de entidades y relaciones
dentro de documentos heterogéneos.

Cada dominio:
- Tiene alias (palabras clave para mapear contexto).
- Define entidades (EntityTypeDef) con atributos relevantes.
- Define relaciones entre entidades.
- Está diseñado para ser fácilmente extensible.

GenericDomain:
- Se usa siempre como capa base, asegurando la detección de entidades 
  universales (persona, email, teléfono, dirección, fecha, monto, id).
"""


from __future__ import annotations
from .schemas import (
    OntologyRegistry,
    OntologyDomain,
    EntityTypeDef,
    AttributeDef,
    RelationTypeDef,
)

# ---------------------------------------------------------------------------
# MEDICAL Domain
# ---------------------------------------------------------------------------
MEDICAL = OntologyDomain(
    domain="medical",
    aliases=[
        "salud", "clínico", "médico", "paciente", "síntoma", "tratamiento", "hospital",
        "doctor", "medicina", "fármaco", "insulina", "diagnóstico",
        "health", "clinical", "medical", "patient", "disease", "treatment", "hospital", "doctor"
    ],
    entity_types=[
        EntityTypeDef(
            name="Disease",
            description="Illness or pathology affecting a patient.",
            aliases=["enfermedad", "patología", "dx", "disease", "illness", "condition"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="icd_code", type="code")
            ]
        ),
        EntityTypeDef(
            name="Symptom",
            description="Sign or indication of a disease.",
            aliases=["síntoma", "signo", "symptom", "sign"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Drug",
            description="Medication or pharmaceutical compound used in treatment.",
            aliases=["fármaco", "medicamento", "medicine", "drug", "tablet", "pill"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="dose", type="string")
            ]
        ),
        EntityTypeDef(
            name="Treatment",
            description="Therapeutic procedure applied to cure or manage a disease.",
            aliases=["tratamiento", "terapia", "therapy", "treatment"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="LabTest",
            description="Laboratory analysis or diagnostic test.",
            aliases=["análisis", "laboratorio", "test", "lab test", "glucose", "hba1c"],
            attributes=[
                AttributeDef(name="test_name"),
                AttributeDef(name="value", type="number"),
                AttributeDef(name="unit", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_symptom", head="Disease", tail="Symptom", description="Disease presents a symptom."),
        RelationTypeDef(name="treated_with", head="Disease", tail="Drug", description="Disease treated with a drug."),
        RelationTypeDef(name="confirmed_by", head="Disease", tail="LabTest", description="Disease confirmed by a lab test."),
    ],
)

# ---------------------------------------------------------------------------
# LEGAL Domain
# ---------------------------------------------------------------------------
LEGAL = OntologyDomain(
    domain="legal",
    aliases=[
        "contrato", "cláusula", "obligación", "firma", "notario", "juicio", "sentencia", "acuerdo",
        "contract", "agreement", "clause", "signature", "trial", "lawsuit", "court", "penalty"
    ],
    entity_types=[
        EntityTypeDef(
            name="Party",
            description="Person or organization participating in a legal agreement.",
            aliases=["parte", "firmante", "persona", "empresa", "party", "signatory"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="role")]
        ),
        EntityTypeDef(
            name="Contract",
            description="Legal document establishing terms and obligations between parties.",
            aliases=["contrato", "acuerdo", "convenio", "contract", "agreement"],
            attributes=[
                AttributeDef(name="effective_date", type="date"),
                AttributeDef(name="term", type="string")
            ]
        ),
        EntityTypeDef(
            name="Obligation",
            description="Specific duty arising from a legal agreement.",
            aliases=["obligación", "responsabilidad", "obligation", "duty", "liability"],
            attributes=[AttributeDef(name="description")]
        ),
        EntityTypeDef(
            name="Penalty",
            description="Monetary or legal sanction due to contract breach.",
            aliases=["multa", "sanción", "penalty", "fine", "sanction"],
            attributes=[
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="currency", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="binds", head="Contract", tail="Party", description="Contract binds a party legally."),
        RelationTypeDef(name="imposes", head="Contract", tail="Obligation", description="Contract imposes an obligation."),
        RelationTypeDef(name="penalizes", head="Obligation", tail="Penalty", description="Obligation leads to a penalty."),
    ],
)

# ---------------------------------------------------------------------------
# IDENTITY Domain
# ---------------------------------------------------------------------------
IDENTITY = OntologyDomain(
    domain="identity",
    aliases=[
        "identificación", "id", "dni", "ine", "rfc", "curp", "pasaporte", "licencia",
        "identity", "passport", "id card", "driver license", "ssn", "nif"
    ],
    entity_types=[
        EntityTypeDef(
            name="Person",
            description="Individual identified by name or document.",
            aliases=["nombre", "titular", "persona", "person", "individual"],
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
            aliases=["domicilio", "dirección", "address", "residence", "home"],
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

# ---------------------------------------------------------------------------
# TECH REVIEW Domain
# ---------------------------------------------------------------------------
TECH = OntologyDomain(
    domain="tech_review",
    aliases=[
        "benchmark", "reseña", "modelo", "gpu", "cpu", "latencia", "precisión", "tecnología",
        "hardware", "software", "review", "tech", "performance", "specs", "accuracy", "fps"
    ],
    entity_types=[
        EntityTypeDef(
            name="Product",
            description="Hardware or software product under evaluation.",
            aliases=["producto", "modelo", "device", "gadget", "software", "hardware"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="vendor")]
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

# ---------------------------------------------------------------------------
# FINANCIAL Domain
# ---------------------------------------------------------------------------
FINANCIAL = OntologyDomain(
    domain="financial",
    aliases=[
        "finanzas", "factura", "transacción", "pago", "banco", "seguro", "mercado", "divisa",
        "acción", "presupuesto", "bolsa", "cotización", "exchange", "finance", "bank", "market",
        "investment", "stock", "forex", "currency", "insurance", "loan", "interest"
    ],
    entity_types=[
        EntityTypeDef(
            name="Invoice",
            description="Commercial document detailing transaction of goods or services.",
            aliases=["factura", "recibo", "invoice", "bill"],
            attributes=[
                AttributeDef(name="invoice_number", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Transaction",
            description="Movement of money between accounts or entities.",
            aliases=["pago", "transferencia", "depósito", "transaction", "payment", "transfer"],
            attributes=[
                AttributeDef(name="transaction_id", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Account",
            description="Financial account or banking identifier.",
            aliases=["cuenta", "iban", "account", "bank account"],
            attributes=[
                AttributeDef(name="account_number", type="string"),
                AttributeDef(name="bank", type="string")
            ]
        ),
        EntityTypeDef(
            name="Policy",
            description="Insurance policy or financial coverage document.",
            aliases=["póliza", "seguro", "policy", "insurance"],
            attributes=[
                AttributeDef(name="policy_id", type="string"),
                AttributeDef(name="coverage", type="string")
            ]
        ),
        EntityTypeDef(
            name="StockIndicator",
            description="Indicator representing market or stock performance.",
            aliases=["acción", "índice", "stock", "share", "equity", "ticker", "indice"],
            attributes=[
                AttributeDef(name="symbol", type="string"),
                AttributeDef(name="price", type="number"),
                AttributeDef(name="change_percent", type="number")
            ]
        ),
        EntityTypeDef(
            name="ExchangeRate",
            description="Rate of conversion between two currencies.",
            aliases=["tipo de cambio", "exchange rate", "forex", "currency"],
            attributes=[
                AttributeDef(name="from_currency", type="string"),
                AttributeDef(name="to_currency", type="string"),
                AttributeDef(name="rate", type="number")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="paid_by", head="Transaction", tail="Account", description="Transaction paid by account."),
        RelationTypeDef(name="covered_by", head="Invoice", tail="Policy", description="Invoice covered by a policy."),
        RelationTypeDef(name="quoted_in", head="StockIndicator", tail="ExchangeRate", description="Stock quoted in specific currency."),
    ],
)

# ---------------------------------------------------------------------------
# E-COMMERCE Domain
# ---------------------------------------------------------------------------
ECOMMERCE = OntologyDomain(
    domain="ecommerce",
    aliases=[
        "carrito", "pedido", "compra", "precio", "producto", "cliente", "review",
        "order", "purchase", "product", "customer", "store", "ecommerce"
    ],
    entity_types=[
        EntityTypeDef(
            name="Order",
            description="Commercial purchase order.",
            aliases=["pedido", "orden", "order", "purchase"],
            attributes=[
                AttributeDef(name="order_id", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Product",
            description="Item available for sale or review.",
            aliases=["producto", "item", "product", "article"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="brand")]
        ),
        EntityTypeDef(
            name="Review",
            description="Customer opinion or feedback about a product.",
            aliases=["reseña", "comentario", "opinión", "review", "feedback"],
            attributes=[
                AttributeDef(name="rating", type="number"),
                AttributeDef(name="text", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="contains", head="Order", tail="Product", description="Order contains a product."),
        RelationTypeDef(name="reviewed_by", head="Product", tail="Review", description="Product reviewed by a customer."),
    ],
)

# ---------------------------------------------------------------------------
# VETERINARY Domain
# ---------------------------------------------------------------------------
VETERINARY = OntologyDomain(
    domain="veterinary",
    aliases=[
        "animal", "mascota", "veterinario", "síntoma", "tratamiento", "ganado",
        "animal", "pet", "veterinary", "vet", "cattle", "disease"
    ],
    entity_types=[
        EntityTypeDef(
            name="Animal",
            description="Animal or pet under veterinary care.",
            aliases=["mascota", "animal", "pet", "dog", "cat", "res"],
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
        RelationTypeDef(name="treated_with", head="Animal", tail="Treatment", description="Animal treated with treatment."),
        RelationTypeDef(name="has_disease", head="Animal", tail="Disease", description="Animal diagnosed with disease."),
    ],
)

# ---------------------------------------------------------------------------
# GEOPOLITICAL Domain
# ---------------------------------------------------------------------------
GEO = OntologyDomain(
    domain="geopolitical",
    aliases=[
        "país", "ciudad", "estado", "frontera", "conflicto", "tratado", "acuerdo",
        "country", "city", "state", "border", "conflict", "treaty", "agreement"
    ],
    entity_types=[
        EntityTypeDef(
            name="Country",
            description="Nation or sovereign state.",
            aliases=["país", "nación", "country", "nation"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="iso_code")]
        ),
        EntityTypeDef(
            name="City",
            description="Urban or municipal entity.",
            aliases=["ciudad", "municipio", "city", "town"],
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
        RelationTypeDef(name="located_in", head="City", tail="Country", description="City located in country."),
        RelationTypeDef(name="involves", head="Event", tail="Country", description="Event involves a country."),
    ],
)

# ---------------------------------------------------------------------------
# REVIEWS & NEWS Domain
# ---------------------------------------------------------------------------
REVIEWS = OntologyDomain(
    domain="reviews_and_news",
    aliases=[
        "review", "reseña", "comentario", "opinion", "feedback", "news", "noticia", "artículo",
        "valoración", "prensa", "artículo financiero", "financial news", "headline", "report"
    ],
    entity_types=[
        EntityTypeDef(
            name="Review",
            description="Opinion or evaluation by a user or critic.",
            aliases=["reseña", "review", "comentario", "feedback", "opinion"],
            attributes=[
                AttributeDef(name="review_id", type="string"),
                AttributeDef(name="stars", type="number"),
                AttributeDef(name="date", type="date"),
                AttributeDef(name="sentiment", type="string")
            ]
        ),
        EntityTypeDef(
            name="NewsArticle",
            description="Published news article or report.",
            aliases=["noticia", "artículo", "reportaje", "news", "article", "report"],
            attributes=[
                AttributeDef(name="title", type="string"),
                AttributeDef(name="publisher", type="string"),
                AttributeDef(name="date", type="date"),
                AttributeDef(name="sentiment", type="string")
            ]
        ),
        EntityTypeDef(
            name="StockIndicator",
            description="Market or stock index referenced in news.",
            aliases=["acción", "índice", "ticker", "stock", "equity"],
            attributes=[
                AttributeDef(name="symbol", type="string"),
                AttributeDef(name="price", type="number"),
                AttributeDef(name="change_percent", type="number")
            ]
        ),
        EntityTypeDef(
            name="MarketEvent",
            description="Event affecting market or economy.",
            aliases=["crisis", "subida", "baja", "market event", "announcement"],
            attributes=[
                AttributeDef(name="event_type", type="string"),
                AttributeDef(name="impact", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="authored_by", head="Review", tail="Person", description="Review authored by person."),
        RelationTypeDef(name="published_by", head="NewsArticle", tail="Organization", description="News published by organization."),
        RelationTypeDef(name="mentions_indicator", head="NewsArticle", tail="StockIndicator", description="News mentions a stock or indicator."),
        RelationTypeDef(name="describes_event", head="NewsArticle", tail="MarketEvent", description="News describes a market event."),
    ],
)

# ---------------------------------------------------------------------------
# GENERIC Domain (Always included)
# ---------------------------------------------------------------------------
GENERIC = OntologyDomain(
    domain="generic",
    aliases=[
        "general", "documento", "texto", "registro", "file", "document", "record", "text"
    ],
    entity_types=[
        EntityTypeDef(name="Person", aliases=["persona", "nombre", "user", "person", "patient"]),
        EntityTypeDef(name="Organization", aliases=["empresa", "institución", "organization", "company"]),
        EntityTypeDef(name="IDNumber", aliases=["id", "rfc", "curp", "ssn", "identifier"]),
        EntityTypeDef(name="Date", aliases=["fecha", "día", "mes", "año", "date", "day", "month", "year"]),
        EntityTypeDef(name="Location", aliases=["ubicación", "ciudad", "estado", "país", "address", "city", "country"]),
        EntityTypeDef(name="PhoneNumber", aliases=["teléfono", "móvil", "celular", "phone", "mobile"]),
        EntityTypeDef(name="Email", aliases=["correo", "email", "mail"]),
        EntityTypeDef(name="Amount", aliases=["monto", "precio", "costo", "valor", "usd", "mxn", "price", "amount"]),
    ],
)

# ---------------------------------------------------------------------------
# GLOBAL REGISTRY
# ---------------------------------------------------------------------------
REGISTRY = OntologyRegistry(
    domains=[
        MEDICAL, LEGAL, IDENTITY, TECH, FINANCIAL,
        ECOMMERCE, VETERINARY, GEO, REVIEWS, GENERIC
    ]
)
