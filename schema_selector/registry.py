# -*- coding: utf-8 -*-
"""
registry.py — Ontología de dominios y entidades para Adaptive Schema Selector.

Este archivo define múltiples dominios comunes (medical, legal, identity, tech,
financial, ecommerce, veterinary, geopolitical, generic) que sirven como 
catálogo de entidades y relaciones a detectar dentro de documentos.

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
# Dominio Médico
# ---------------------------------------------------------------------------
MEDICAL = OntologyDomain(
    domain="medical",
    aliases=["salud", "clínico", "médico", "paciente", "síntomas", "tratamiento",
             "fármaco", "insulina", "metformina", "diagnóstico", "hospital"],
    entity_types=[
        EntityTypeDef(
            name="Disease",
            aliases=["enfermedad", "patología", "dx"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="icd_code", type="code")
            ]
        ),
        EntityTypeDef(
            name="Symptom",
            aliases=["síntoma", "signo"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Drug",
            aliases=["fármaco", "medicamento", "tableta", "inyección"],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="dose", type="string")
            ]
        ),
        EntityTypeDef(
            name="Treatment",
            aliases=["terapia", "manejo", "tratamiento"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="LabTest",
            aliases=["análisis", "laboratorio", "glucosa", "hba1c"],
            attributes=[
                AttributeDef(name="test_name"),
                AttributeDef(name="value", type="number"),
                AttributeDef(name="unit", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_symptom", head="Disease", tail="Symptom"),
        RelationTypeDef(name="treated_with", head="Disease", tail="Drug"),
        RelationTypeDef(name="confirmed_by", head="Disease", tail="LabTest"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio Legal
# ---------------------------------------------------------------------------
LEGAL = OntologyDomain(
    domain="legal",
    aliases=["contrato", "cláusula", "obligación", "firma", "notario", "juicio", "sentencia"],
    entity_types=[
        EntityTypeDef(
            name="Party",
            aliases=["parte", "firmante", "empresa", "persona"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="role")]
        ),
        EntityTypeDef(
            name="Contract",
            aliases=["contrato", "acuerdo", "convenio"],
            attributes=[
                AttributeDef(name="effective_date", type="date"),
                AttributeDef(name="term", type="string")
            ]
        ),
        EntityTypeDef(
            name="Obligation",
            aliases=["obligación", "responsabilidad"],
            attributes=[AttributeDef(name="description")]
        ),
        EntityTypeDef(
            name="Penalty",
            aliases=["multa", "sanción"],
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

# ---------------------------------------------------------------------------
# Dominio Identidad
# ---------------------------------------------------------------------------
IDENTITY = OntologyDomain(
    domain="identity",
    aliases=["ine", "pasaporte", "licencia", "rfc", "curp", "nss", "id", "dni"],
    entity_types=[
        EntityTypeDef(
            name="Person",
            aliases=["nombre", "titular"],
            attributes=[AttributeDef(name="full_name")]
        ),
        EntityTypeDef(
            name="IDDocument",
            aliases=["identificación", "id", "ine", "pasaporte"],
            attributes=[
                AttributeDef(name="id_type"),
                AttributeDef(name="id_number", type="id")
            ]
        ),
        EntityTypeDef(
            name="Address",
            aliases=["domicilio", "dirección", "calle"],
            attributes=[
                AttributeDef(name="line1"),
                AttributeDef(name="city"),
                AttributeDef(name="state"),
                AttributeDef(name="postal_code")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="identified_by", head="Person", tail="IDDocument"),
        RelationTypeDef(name="resides_at", head="Person", tail="Address"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio Tecnología
# ---------------------------------------------------------------------------
TECH = OntologyDomain(
    domain="tech_review",
    aliases=["benchmark", "reseña", "modelo", "gpu", "cpu", "latencia", "accuracy", "fps"],
    entity_types=[
        EntityTypeDef(
            name="Product",
            aliases=["producto", "modelo", "dispositivo"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="vendor")]
        ),
        EntityTypeDef(
            name="Metric",
            aliases=["latencia", "tiempo", "fps", "precisión"],
            attributes=[
                AttributeDef(name="metric_name"),
                AttributeDef(name="value", type="number")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="has_metric", head="Product", tail="Metric"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio Financiero
# ---------------------------------------------------------------------------
FINANCIAL = OntologyDomain(
    domain="financial",
    aliases=["factura", "transacción", "pago", "banco", "seguro", "presupuesto", "ingreso", "gasto"],
    entity_types=[
        EntityTypeDef(
            name="Invoice",
            aliases=["factura", "recibo"],
            attributes=[
                AttributeDef(name="invoice_number", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Transaction",
            aliases=["pago", "transferencia", "depósito"],
            attributes=[
                AttributeDef(name="transaction_id", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Account",
            aliases=["cuenta", "número de cuenta", "iban"],
            attributes=[
                AttributeDef(name="account_number", type="string"),
                AttributeDef(name="bank", type="string")
            ]
        ),
        EntityTypeDef(
            name="Policy",
            aliases=["póliza", "seguro"],
            attributes=[
                AttributeDef(name="policy_id", type="string"),
                AttributeDef(name="coverage", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="paid_by", head="Transaction", tail="Account"),
        RelationTypeDef(name="covered_by", head="Invoice", tail="Policy"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio E-commerce
# ---------------------------------------------------------------------------
ECOMMERCE = OntologyDomain(
    domain="ecommerce",
    aliases=["carrito", "pedido", "compra", "precio", "reseña", "producto", "cliente"],
    entity_types=[
        EntityTypeDef(
            name="Order",
            aliases=["pedido", "orden"],
            attributes=[
                AttributeDef(name="order_id", type="string"),
                AttributeDef(name="amount", type="number")
            ]
        ),
        EntityTypeDef(
            name="Product",
            aliases=["artículo", "producto", "item"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="brand")]
        ),
        EntityTypeDef(
            name="Review",
            aliases=["reseña", "comentario", "opinión"],
            attributes=[
                AttributeDef(name="rating", type="number"),
                AttributeDef(name="text", type="string")
            ]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="contains", head="Order", tail="Product"),
        RelationTypeDef(name="reviewed_by", head="Product", tail="Review"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio Veterinaria
# ---------------------------------------------------------------------------
VETERINARY = OntologyDomain(
    domain="veterinary",
    aliases=["animal", "mascota", "perro", "gato", "ganado", "veterinario", "síntoma"],
    entity_types=[
        EntityTypeDef(
            name="Animal",
            aliases=["mascota", "animal", "perro", "gato", "res"],
            attributes=[AttributeDef(name="species"), AttributeDef(name="breed")]
        ),
        EntityTypeDef(
            name="Disease",
            aliases=["enfermedad", "zoonosis"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Treatment",
            aliases=["tratamiento", "vacuna"],
            attributes=[AttributeDef(name="name")]
        ),
    ],
    relation_types=[
        RelationTypeDef(name="treated_with", head="Animal", tail="Treatment"),
        RelationTypeDef(name="has_disease", head="Animal", tail="Disease"),
    ],
)

# ---------------------------------------------------------------------------
# Dominio Geopolítico
# ---------------------------------------------------------------------------
GEO = OntologyDomain(
    domain="geopolitical",
    aliases=["país", "ciudad", "estado", "frontera", "conflicto", "tratado"],
    entity_types=[
        EntityTypeDef(
            name="Country",
            aliases=["país", "nación"],
            attributes=[AttributeDef(name="name"), AttributeDef(name="iso_code")]
        ),
        EntityTypeDef(
            name="City",
            aliases=["ciudad", "municipio"],
            attributes=[AttributeDef(name="name")]
        ),
        EntityTypeDef(
            name="Event",
            aliases=["conflicto", "tratado", "acuerdo"],
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
# ---------------------------------------------------------------------------
# Dominio Reseñas
# ---------------------------------------------------------------------------
# --- Dominio Reviews / Opiniones de Usuarios ---
# --- Dominio Reviews / Opiniones de Usuarios ---
REVIEWS = OntologyDomain(
    domain="reviews",
    aliases=[
        "review", "reseña", "comentario", "opinion", "feedback",
        "criticism", "rating", "testimonial", "experience",
        "stars", "customer review", "user review", "valoración", "crítica"
    ],
    entity_types=[
        # Entidad principal: la reseña en sí misma
        EntityTypeDef(
            name="Review",
            aliases=["reseña", "comentario", "opinión", "feedback", "review"],
            attributes=[
                AttributeDef(name="review_id", type="string"),
                AttributeDef(name="stars", type="number"),   # rating 1–5
                AttributeDef(name="date", type="date"),
                AttributeDef(name="useful", type="number"),
                AttributeDef(name="funny", type="number"),
                AttributeDef(name="cool", type="number"),
                AttributeDef(name="sentiment", type="string", description="Resultado de análisis de sentimiento"),
                AttributeDef(name="type", type="string", description="Tipo de reseña: texto, foto, video, etc.")
        ]
        ),

        # Usuario que emite la reseña
        EntityTypeDef(
            name="Reviewer",
            aliases=["usuario", "cliente", "autor", "user", "customer", "reviewer", "critic"],
            attributes=[
                AttributeDef(name="user_id", type="string"),
                AttributeDef(name="name", type="string")
            ]
        ),

        # Negocio, producto o servicio reseñado
        EntityTypeDef(
            name="EntityReviewed",
            aliases=["negocio", "producto", "servicio", "empresa", "tienda",
                     "restaurante", "bar", "gimnasio", "iglesia", "hotel", "app", "cafe", "store", "shop"],
            attributes=[
                AttributeDef(name="business_id", type="string"),
                AttributeDef(name="name", type="string"),
                AttributeDef(name="category", type="string"),
                AttributeDef(name="price_range", type="string"),
                AttributeDef(name="brand", type="string")
            ]
        ),

        # Ítems específicos mencionados (ej: platos, bebidas, productos)
        EntityTypeDef(
            name="Item",
            aliases=["plato", "comida", "bebida", "producto", "dish", "meal", "servicio",
                     "drink", "cocktail", "pizza", "calzone", "toast", "gyro", "skillet", "bread"],
            attributes=[
                AttributeDef(name="name", type="string"),
                AttributeDef(name="type", type="string")
            ]
        ),

        # Aspectos de servicio (mesero, delivery, atención)
        EntityTypeDef(
            name="ServiceAspect",
            aliases=["servicio", "mesero", "staff", "manager", "atención", "delivery",
                     "soporte", "waiter", "waitress", "service"],
            attributes=[
                AttributeDef(name="aspect", type="string"),
                AttributeDef(name="quality", type="string")
            ]
        ),

        # Experiencia subjetiva (ambiente, limpieza, música)
        EntityTypeDef(
            name="Experience",
            aliases=["ambiente", "atmósfera", "entorno", "limpieza", "comodidad", "música",
                     "decoración", "ambience", "atmosphere", "cleanliness", "music",
                     "decor", "friendly", "pleasant", "excellent", "amazing", "phenomenal"],
            attributes=[
                AttributeDef(name="aspect", type="string"),
                AttributeDef(name="sentiment", type="string")
            ]
        ),

        # Localización (ej: ciudad, sucursal, parque)
        EntityTypeDef(
            name="Location",
            aliases=["ubicación", "dirección", "ciudad", "lugar", "branch", "sucursal"]
        ),
    ],
    relation_types=[
        # Review → Usuario / Entidad
        RelationTypeDef(name="authored_by", head="Review", tail="Reviewer"),
        RelationTypeDef(name="reviews", head="Review", tail="EntityReviewed"),

        # Review → Ítems, Aspectos y Experiencia
        RelationTypeDef(name="about_item", head="Review", tail="Item"),
        RelationTypeDef(name="mentions_aspect", head="Review", tail="ServiceAspect"),
        RelationTypeDef(name="describes_experience", head="Review", tail="Experience"),

        # Entidad reseñada → Localización
        RelationTypeDef(name="located_in", head="EntityReviewed", tail="Location"),

        # Producto ↔ Negocio
        RelationTypeDef(name="offered_by", head="Item", tail="EntityReviewed"),
    ],
)


# ---------------------------------------------------------------------------
# Dominio Genérico (Siempre incluido)
# ---------------------------------------------------------------------------
GENERIC = OntologyDomain(
    domain="generic",
    aliases=["general", "documento", "texto", "registro"],
    entity_types=[
        EntityTypeDef(name="Person", aliases=["persona", "nombre", "usuario", "doctor", "paciente"]),
        EntityTypeDef(name="Organization", aliases=["empresa", "institución", "organización"]),
        EntityTypeDef(name="IDNumber", aliases=["id", "rfc", "curp", "nss", "folio"]),
        EntityTypeDef(name="Date", aliases=["fecha", "día", "mes", "año"]),
        EntityTypeDef(name="Location", aliases=["ubicación", "ciudad", "estado", "país", "dirección"]),
        EntityTypeDef(name="PhoneNumber", aliases=["teléfono", "móvil", "celular"]),
        EntityTypeDef(name="Email", aliases=["correo", "email", "mail"]),
        EntityTypeDef(name="Amount", aliases=["monto", "precio", "costo", "valor", "usd", "mxn"]),
    ],
)

# ---------------------------------------------------------------------------
# Registro Global
# ---------------------------------------------------------------------------
REGISTRY = OntologyRegistry(
    domains=[MEDICAL, LEGAL, IDENTITY, TECH, FINANCIAL, ECOMMERCE, VETERINARY, GEO, REVIEWS, GENERIC]
)
