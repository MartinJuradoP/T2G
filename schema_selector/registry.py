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
 - Mapear nombres del selector -> dominios internos del registry (sinónimos).
  * Exponer entidades, relaciones y aliases por dominio de forma uniforme.
  * Construir bloques compactos para inyectar en prompts.
- No rompe compatibilidad: mantiene las clases/constantes anteriores (MEDICAL, LEGAL,

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
from typing import List, Dict, Set, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from collections import defaultdict
import json
import pandas as pd

# Importa contratos base del selector (compatibles)
from .schemas import (
    OntologyDomain as _BaseOntologyDomain,
    OntologyRegistry as _BaseOntologyRegistry,
    EntityTypeDef,
    AttributeDef,
    RelationTypeDef,
)


# ===========================================================================
#  Clases extendidas con validación y trazabilidad
# ===========================================================================
class OntologyDomain(_BaseOntologyDomain):
    """Extiende para alias negativos, stopwords, peso y schema_name (sin romper contratos base)."""
    stopwords: Set[str] = Field(default_factory=set)
    negative_aliases: Set[str] = Field(default_factory=set)
    weight: float = Field(default=1.0)
    schema_name: str = Field(default="generic_text_v1")
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _normalize_lists(self) -> "OntologyDomain":
        self.aliases = sorted(set(a.strip().lower() for a in self.aliases if a))
        self.stopwords = set(w.lower().strip() for w in self.stopwords)
        self.negative_aliases = set(w.lower().strip() for w in self.negative_aliases)
        # Evita choques básicos
        overlap = set(self.aliases) & set(self.negative_aliases)
        if overlap:
            raise ValueError(f"Alias conflictivos en dominio '{self.domain}': {overlap}")
        return self

    # Utilidades legibles (no usadas por pydantic)
    def entity_names(self) -> List[str]:
        return [e.name for e in self.entity_types]

    def relation_names(self) -> List[str]:
        return [r.name for r in self.relation_types]

    def to_prompt_block(
        self,
        alias_limit: int = 15,
        include_entities: bool = True,
        include_relations: bool = True,
        include_aliases: bool = True
    ) -> str:
        """
        Devuelve un bloque textual compacto para inyección en prompts.
        Permite controlar si se incluyen entidades, relaciones y/o aliases.

        Parámetros
        ----------
        alias_limit : int
            Número máximo de aliases a mostrar por dominio.
            Si es 0, no muestra ninguno.
        include_entities : bool
            Si False, omite la línea de entidades.
        include_relations : bool
            Si False, omite la línea de relaciones.
        include_aliases : bool
            Si False, omite completamente la línea de aliases.
        """
        lines = [f"- **{self.domain.upper()}**:"]

        # 👇 Solo imprime si está habilitado
        if include_entities:
            ents = ", ".join(self.entity_names()) or "(sin entidades)"
            lines.append(f"    • Entidades → {ents}")

        if include_relations:
            rels = ", ".join(self.relation_names()) or "(sin relaciones)"
            lines.append(f"    • Relaciones → {rels}")

        if include_aliases and alias_limit and alias_limit > 0 and self.aliases:
            subset = list(self.aliases)[:alias_limit]
            if subset:
                aliases_str = ", ".join(subset)
                lines.append(f"    • Aliases → {aliases_str}")

        return "\n".join(lines)



class OntologyRegistry(_BaseOntologyRegistry):
    """Ontología global con extras de auditoría (mantiene API get_domain original)."""

    domains: List[OntologyDomain] = Field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for d in self.domains:
            rows.append({
                "Domain": d.domain,
                "#Aliases": len(d.aliases),
                "#Stopwords": len(d.stopwords),
                "#Negatives": len(d.negative_aliases),
                "#Entities": len(d.entity_types),
                "#Relations": len(d.relation_types),
                "Weight": d.weight,
                "Schema": d.schema_name,
            })
        return pd.DataFrame(rows).sort_values(by="Domain")

    def conflicts_matrix(self) -> pd.DataFrame:
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

# ===========================================================================
# 🩺 MEDICAL Domain
# ===========================================================================
MEDICAL = OntologyDomain(
    domain="medical",
    schema_name="medical",
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
    schema_name="legal",
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
            aliases=["parte", "firmante", "persona", "empresa", "party","contratante", "el contratante",
            "prestador", "el prestador",
            "proveedor", "cliente"],
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
            name="Representative",
            description="Legal representative or attorney-in-fact of a party.",
            aliases=[
                "apoderado", "apoderador", "apoderamiento",
                "representante legal", "representante", "signatario", "apod."
            ],
            attributes=[
                AttributeDef(name="name"),
                AttributeDef(name="title", type="string"),
                AttributeDef(name="power_scope", type="string", description="Alcance del poder")
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
    schema_name="financial",
    weight=1.0,
    aliases=[
        # ----------------------------
        # Conceptos generales de finanzas y economía
        # ----------------------------
        "finanzas", "economía", "mercado", "bolsa", "cotización", "acción", "acciones","ticker","index"
        "capital", "inversión", "divisa", "interés", "seguro", "pago", "banco",
        "presupuesto", "loan", "credito", "interés compuesto", "policy", "póliza",
        "beneficio", "loss", "profit", "revenue", "ingresos", "gasto", "expense",
        "income statement", "balance sheet", "cash flow", "financial statement",
        "finance", "accounting", "investment", "fund", "trading", "exchange",
        "foreign exchange", "FX", "hedge fund", "mutual fund", "ETF", "bond",
        "derivative", "futures", "options", "swap",
        "stock", "ticker", "index", "indice", "benchmark",
        "NASDAQ", "Dow Jones", "S&P 500", "SPX", "DJI", "NYSE",
        "Russell 2000", "FTSE", "Nikkei", "IBEX", "Bovespa", "DAX", "CAC 40",

        # ----------------------------
        # Indicadores y métricas financieras
        # ----------------------------
        "EPS", "earnings per share", "P/E", "PE ratio", "PEG", "ROE", "ROI", "ROA",
        "EBITDA", "EBIT", "margin", "profit margin", "gross margin",
        "revenue growth", "operating income", "net income", "cash flow", "valuation",
        "market cap", "price target", "forecast", "guidance", "outlook",
        "consensus estimate", "Zacks Rank", "rating", "buy", "sell", "hold",
        "outperform", "underperform", "dividend", "yield", "return on investment",
        "leverage", "liquidity ratio", "P&L", "ROI", "ROIC", "EPS estimate",

        # ----------------------------
        # Índices, tickers y entidades de mercado
        # ----------------------------
        "stock", "ticker", "index", "indice", "benchmark",
        "NASDAQ", "Dow Jones", "S&P 500", "SPX", "DJI", "NYSE",
        "Russell 2000", "FTSE", "Nikkei", "IBEX", "Bovespa", "DAX", "CAC 40",
        
        # ----------------------------
        # Documentos, operaciones y entidades financieras
        # ----------------------------
        "factura", "invoice", "transacción", "transaction", "transfer", "payment",
        "budget", "presupuesto", "loan", "credit", "interest rate", "claim",
        "compensation", "policy", "insurance", "premium", "account", "bank account",

        # ----------------------------
        # Negocios, startups, fusiones y adquisiciones
        # ----------------------------
        "corporate", "business", "startup", "enterprise", "company", "corporation",
        "IPO", "secondary share sale", "funding", "round", "series A", "series B",
        "investment round", "valuation", "market capitalization",
        "merger", "acquisition", "deal", "partnership", "infrastructure deal",
        "private equity", "venture capital", "capital raise", "investor", "shareholder",

             
    ],
    negative_aliases={"hospital", "doctor", "contract", "disease", "recipe"},
    stopwords={"monto", "total", "fecha", "número", "porcentaje"},

    # =====================================================
    # ENTITY TYPES
    # =====================================================
    entity_types=[
        # -----------------------------------------------------
        # Entidades tradicionales (documentos y cuentas)
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Entidades de mercado y bursátiles
        # -----------------------------------------------------
        EntityTypeDef(
            name="Stock",
            description="Public company share traded on the market.",
            aliases=["acción", "stock", "equity"],
            attributes=[
                AttributeDef(name="symbol", type="string"),
                AttributeDef(name="price", type="number"),
                AttributeDef(name="index", type="string"),
                AttributeDef(name="change_percent", type="number"),
            ]
        ),
        EntityTypeDef(
            name="Ticker",
            description=(
                "Unique symbol used to identify a publicly traded stock or index on an exchange. "
                "Includes both company tickers (e.g., AAPL, MSFT) and index symbols (e.g., ^SPX, ^DJI)."
            ),
            aliases=["ticker", "símbolo", "stock symbol", "market symbol", "trading symbol","^","ticker symbol"],
            attributes=[
                AttributeDef(name="symbol", type="string"),
                AttributeDef(name="exchange", type="string"),
                AttributeDef(name="price", type="number"),
                AttributeDef(name="change_percent", type="number"),
                AttributeDef(name="as_of_date", type="date"),
            ]
        ),

        EntityTypeDef(
            name="Index",
            description="Market index aggregating multiple stocks.",
            aliases=["index", "indice", "benchmark", "SPX", "DJI", "NASDAQ","Dow Jones","S&P 500"],
            attributes=[
                AttributeDef(name="name", type="string"),
                AttributeDef(name="change_percent", type="number"),
            ]
        ),
        EntityTypeDef(
            name="EarningsReport",
            description="Company financial disclosure (quarterly or annual).",
            aliases=["earnings", "quarterly report", "financial results", "EPS", "guidance"],
            attributes=[
                AttributeDef(name="eps", type="number"),
                AttributeDef(name="revenue", type="number"),
                AttributeDef(name="forecast", type="number"),
                AttributeDef(name="date", type="date"),
            ]
        ),

        # -----------------------------------------------------
        # Entidades corporativas y de negocio
        # -----------------------------------------------------
        EntityTypeDef(
            name="Company",
            description="An organization or corporation involved in financial or commercial activity.",
            aliases=["company", "corporation", "startup", "business", "enterprise"],
            attributes=[
                AttributeDef(name="name", type="string"),
                AttributeDef(name="industry", type="string"),
                AttributeDef(name="valuation", type="number"),
                AttributeDef(name="revenue", type="number"),
                AttributeDef(name="headquarters", type="string"),
            ]
        ),
        EntityTypeDef(
            name="Executive",
            description="Corporate leader or senior official.",
            aliases=["CEO", "CFO", "COO", "CTO", "executive", "founder", "chairman"],
            attributes=[
                AttributeDef(name="name", type="string"),
                AttributeDef(name="role", type="string"),
                AttributeDef(name="company", type="string"),
            ]
        ),
        EntityTypeDef(
            name="InvestmentRound",
            description="A financial event involving capital raising or share sale.",
            aliases=["funding", "round", "series A", "series B", "IPO", "secondary share sale"],
            attributes=[
                AttributeDef(name="round_type", type="string"),
                AttributeDef(name="amount", type="number"),
                AttributeDef(name="investors", type="list"),
                AttributeDef(name="date", type="date"),
            ]
        ),
        EntityTypeDef(
            name="Partnership",
            description="Collaboration or deal between companies.",
            aliases=["deal", "partnership", "agreement", "infrastructure deal"],
            attributes=[
                AttributeDef(name="partners", type="list"),
                AttributeDef(name="sector", type="string"),
                AttributeDef(name="value", type="number"),
            ]
        ),
    ],

    # =====================================================
    # RELATION TYPES
    # =====================================================
    relation_types=[
        # Relaciones básicas de transacción
        RelationTypeDef(name="paid_by", head="Transaction", tail="Account"),
        RelationTypeDef(name="covered_by", head="Invoice", tail="Policy"),

        # Relaciones bursátiles y de reporte
        RelationTypeDef(name="belongs_to_index", head="Stock", tail="Index"),
        RelationTypeDef(name="reports", head="Organization", tail="EarningsReport"),
        RelationTypeDef(name="reported", head="Company", tail="EarningsReport"),

        # Relaciones corporativas y de inversión
        RelationTypeDef(name="led_by", head="Company", tail="Executive"),
        RelationTypeDef(name="raised_in", head="Company", tail="InvestmentRound"),
        RelationTypeDef(name="partnered_with", head="Company", tail="Company"),
        RelationTypeDef(name="invested_in", head="Investor", tail="Company"),
        RelationTypeDef(name="analyzed_by", head="Stock", tail="Analyst"),
    ],
)

# ===========================================================================
# 💻 TECH REVIEW Domain
# ===========================================================================
TECH = OntologyDomain(
    domain="tech_review",
    schema_name="tech_review",
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
    schema_name="ecommerce",
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
    schema_name="veterinary",
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
    schema_name="geopolitical",
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
    schema_name="reviews_and_opinions",
    weight=0.9,
    aliases=[
        # Estructura general de reseñas
        "review", "reseña", "comentario", "opinión", "feedback", "valoración",
        "rating", "calificación", "testimonio", "experiencia","food","comida",
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
    schema_name="generic",
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
            aliases=["ubicación", "ciudad", "address", "lugar", "sitio"]
        ),
        EntityTypeDef(
            name="Country",
            description="Nación, país o región.",
            aliases=[
                "país", "nación", "estado", "reino", "territorio", "China", "México", 
                "Estados Unidos", "USA", "EE.UU.", "Francia", "Alemania", "Brasil", "Japón"
            ]
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
        # Identificadores y trazabilidad
        # ------------------------------------------------------
        EntityTypeDef(
            name="Identifier",
            description="Código o número de identificación genérico (ID, RFC, folio, ticket, referencia).",
            aliases=["id", "rfc", "folio", "ticket", "ref", "identificador", "código", "codigo", "no."]
        ),
        EntityTypeDef(
            name="ReferenceCode",
            description="Código de referencia o identificador alfanumérico dentro del documento.",
            aliases=["referencia", "reference", "code", "clave", "número", "num", "serie"]
        ),
        EntityTypeDef(
            name="DocumentID",
            description="Número o clave de documento formal (RFC, INE, pasaporte, etc.).",
            aliases=["rfc", "ine", "passport", "dni", "id", "identificación", "identidad"]
        ),
        EntityTypeDef(
            name="AccountNumber",
            description="Número de cuenta o referencia bancaria.",
            aliases=["cuenta", "account", "iban", "clabe", "bank", "banco"]
        ),
        EntityTypeDef(
            name="TransactionCode",
            description="Código de transacción o folio de operación.",
            aliases=["transacción", "transaction", "operación", "folio", "txid"]
        ),
        EntityTypeDef(
            name="Barcode",
            description="Código de barras o QR detectado textual o visualmente.",
            aliases=["barcode", "código de barras", "qr", "qr code", "etiqueta"]
        ),

        # ------------------------------------------------------
        # Comunicación digital, contacto y trazabilidad técnica
        # ------------------------------------------------------
        EntityTypeDef(
            name="URL",
            description="Dirección o enlace web, completo o parcial (http, https, www).",
            aliases=["url", "link", "website", "sitio", "web", "enlace", "http", "https", "www"]
        ),
        EntityTypeDef(
            name="EmailAddress",
            description="Dirección de correo electrónico.",
            aliases=["correo", "correo electrónico", "email", "mail", "e-mail", "contacto@"]
        ),
        EntityTypeDef(
            name="PhoneNumber",
            description="Número telefónico o de contacto.",
            aliases=["teléfono", "telefono", "número", "celular", "móvil", "movil", "phone", "contacto", "whatsapp", "fax"]
        ),
        EntityTypeDef(
            name="Address",
            description="Dirección física o postal completa o parcial (calle, colonia, ciudad, CP).",
            aliases=["domicilio", "dirección", "address", "calle", "avenida", "colonia", "cp", "código postal", "postal"]
        ),
        EntityTypeDef(
            name="IPAddress",
            description="Dirección IP (IPv4 o IPv6) usada en trazabilidad o metadatos técnicos.",
            aliases=["ip", "ipv4", "ipv6", "ip address"]
        ),
        EntityTypeDef(
            name="SocialHandle",
            description="Identificador o mención de usuario en redes sociales (@usuario).",
            aliases=["usuario", "@", "handle", "cuenta", "perfil", "user", "nickname"]
        ),
        EntityTypeDef(
            name="Hashtag",
            description="Etiqueta temática usada en redes sociales (#tema).",
            aliases=["hashtag", "#", "etiqueta"]
        ),
        EntityTypeDef(
            name="FileReference",
            description="Referencia a nombre o ruta de archivo local o remoto.",
            aliases=["archivo", "documento", "imagen", "foto", "adjunto", "attachment", ".pdf", ".docx", ".xls", ".csv", ".jpg", ".png"]
        ),

        # ------------------------------------------------------
        # Valores, medidas y unidades
        # ------------------------------------------------------
        EntityTypeDef(
            name="Percentage",
            description="Valor porcentual expresado en el texto.",
            aliases=["porcentaje", "%", "percent"]
        ),
        EntityTypeDef(
            name="Measurement",
            description="Unidad de medida genérica (kg, m, cm, L, etc.).",
            aliases=["medida", "kg", "m", "cm", "litro", "unidad", "measurement"]
        ),

        # ------------------------------------------------------
        # Temporalidad extendida
        # ------------------------------------------------------
        EntityTypeDef(
            name="Time",
            description="Hora o expresión temporal específica (hh:mm, AM/PM, etc.).",
            aliases=["hora", "minuto", "segundo", "am", "pm", "tiempo"]
        ),
        EntityTypeDef(
            name="Duration",
            description="Periodo de tiempo o duración expresada naturalmente.",
            aliases=["duración", "periodo", "semana", "meses", "años", "horas", "días"]
        ),

        # ------------------------------------------------------
        # Expresiones emocionales, valorativas o de acción
        # ------------------------------------------------------
        EntityTypeDef(
            name="Sentiment",
            description="Expresión de emoción o valoración (positivo, negativo, neutral).",
            aliases=["bueno", "malo", "increíble", "terrible", "excelente", "horrible", "positivo", "negativo"]
        ),
        EntityTypeDef(
            name="ActionVerb",
            description="Verbos de acción genéricos en texto (comprar, enviar, cancelar, etc.).",
            aliases=["comprar", "pagar", "enviar", "cancelar", "registrar", "firmar", "aceptar"]
        ),
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

    # ------------------------------------------------------
    # Relaciones genéricas frecuentes
    # ------------------------------------------------------
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
        RelationTypeDef(
            name="has_contact",
            head="Organization",
            tail="PhoneNumber",
            description="Una organización tiene asociado un número de contacto o teléfono."
        ),
        RelationTypeDef(
            name="located_at",
            head="Organization",
            tail="Address",
            description="Entidad o persona localizada en una dirección física."
        ),
    ],

    # ------------------------------------------------------
    # Notas de dominio
    # ------------------------------------------------------
    notes=(
        "El dominio GENÉRICO actúa como fallback universal. Detecta patrones transversales "
        "comunes en textos no clasificables (emails, publicaciones, notas, logs, formularios, etc.). "
        "Incluye soporte para teléfonos, direcciones, URLs, correos, identificadores, emojis, "
        "y referencias de archivo o cuenta."
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
# ===========================================================================
# 🧭 RegistryHelper — Capa de compatibilidad con el Selector
# ===========================================================================

class RegistryHelper:
    """
    Provee utilidades para:
    - Normalizar nombres de dominio (selector -> registry).
    - Resolver sinónimos.
    - Exponer entidades/relaciones/aliases para prompts.
    - Construir listas de dominios permitidos (allowed) con fallback 'generic'.
    """

    # Sinónimos aceptados desde el selector hacia el registry
    _DOMAIN_SYNONYMS: Dict[str, str] = {
        "finance": "financial",
        "finanzas": "financial",
        "legal": "legal",
        "law": "legal",
        "tech": "tech_review",
        "technology": "tech_review",
        "reviews": "reviews_and_opinions",
        "opinions": "reviews_and_opinions",
        "opinion": "reviews_and_opinions",
        "geopolitics": "geopolitical",
        "geo": "geopolitical",
        "e-commerce": "ecommerce",
        "generic": "generic",
        "med": "medical",
        "health": "medical",
        "vet": "veterinary",
        "veterinaria": "veterinary",
    }

    def __init__(self, registry: OntologyRegistry):
        self.registry = registry
        # índice por nombre canónico
        self._by_name = {d.domain.lower(): d for d in registry.domains}

    def normalize_domain(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        key = name.strip().lower()
        if key in self._by_name:
            return key
        if key in self._DOMAIN_SYNONYMS:
            mapped = self._DOMAIN_SYNONYMS[key]
            return mapped if mapped in self._by_name else None
        # búsqueda por substring leve
        for dom in self._by_name:
            if key in dom or dom in key:
                return dom
        return None

    def match_domains(self, names: List[str]) -> List[str]:
        out = []
        for n in names or []:
            canon = self.normalize_domain(n)
            if canon and canon not in out:
                out.append(canon)
        # Siempre incluir 'generic' al final
        if "generic" not in out:
            out.append("generic")
        return out

    def get(self, name: str) -> Optional[OntologyDomain]:
        canon = self.normalize_domain(name)
        return self._by_name.get(canon) if canon else None

    def ensure_domains(self, preferred: List[str], extras: List[str] | None = None) -> List[OntologyDomain]:
        """
        Devuelve objetos dominio en orden: preferred (normalizados) + extras (si existen) + generic.
        Sin duplicados.
        """
        order = self.match_domains(preferred or [])
        if extras:
            order += [d for d in self.match_domains(extras) if d not in order]
        # materializa
        uniq: List[OntologyDomain] = []
        seen = set()
        for dn in order:
            d = self._by_name.get(dn)
            if d and dn not in seen:
                uniq.append(d); seen.add(dn)
        return uniq

    def prompt_blocks_for(
        self,
        domains: List[str],
        alias_limit: int = 15,
        include_entities: bool = True,
        include_relations: bool = True,
        include_aliases: bool = True
    ) -> str:
        """
        Construye bloques de texto para dominios activos.
        Controla qué secciones se incluyen (entidades, relaciones, aliases).
        """
        dd = self.ensure_domains(domains)
        if not dd:
            return "  • (sin dominios registrados en Registry)"

        blocks = [
            d.to_prompt_block(
                alias_limit=alias_limit,
                include_entities=include_entities,
                include_relations=include_relations,
                include_aliases=include_aliases,
            )
            for d in dd
        ]
        return "\n".join(blocks)


    def allowed_domains(self, top_domains: List[str]) -> List[str]:
        """Lista blanca para el post-proceso de menciones."""
        return self.match_domains(top_domains)

    def hint_map(self, top_domains: List[str], alias_limit: int = 40) -> Dict[str, List[str]]:
        """Mapa dominio → pistas léxicas (para reetiquetar genéricas)."""
        hints: Dict[str, List[str]] = {}
        for d in self.ensure_domains(top_domains):
            hints[d.domain] = [a for a in list(d.aliases)[:alias_limit] if a and len(a) >= 3]
        return hints

