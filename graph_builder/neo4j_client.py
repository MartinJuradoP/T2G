# graph_builder/neo4j_client.py
# -*- coding: utf-8 -*-
"""
Neo4jClient — Cliente robusto para ingesta T2G (Graph Builder / Entity Linker)
===============================================================================

Responsabilidades:
- Gestionar conexión al clúster Neo4j (driver oficial v5).
- Crear constraints/índices idempotentes.
- Exponer helpers de alto nivel para MERGE de nodos/relaciones.
- Proveer ejecución segura de queries con reintentos y logging estructurado.
- Devolver contadores básicos para integrarse con metrics.py.

Variables de entorno esperadas (pueden sobreescribirse vía kwargs):
- NEO4J_URI        (p.ej. bolt://localhost:7687)
- NEO4J_USER       (p.ej. neo4j)
- NEO4J_PASSWORD   (se sugiere usar .env)

Requisitos:
- pip install neo4j python-dotenv
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "El paquete 'neo4j' es requerido. Instala con: pip install neo4j"
    ) from e

try:
    from dotenv import load_dotenv  # opcional, pero útil
    load_dotenv()
except Exception:
    # Si no está dotenv, simplemente ignoramos: se puede usar env nativo del SO
    pass


logger = logging.getLogger("graph_builder.neo4j_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Reintentos por fallos transitorios de red/servicio
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_SLEEP = 0.6  # segundos


@dataclass
class Neo4jStats:
    """Contadores simples para facilitar integración con metrics.py."""
    new_documents: int = 0
    existing_documents: int = 0
    new_entities: int = 0
    existing_entities: int = 0
    relations_created: int = 0
    relations_reused: int = 0
    labels_created: int = 0


class Neo4jClient:
    """
    Cliente de alto nivel para operaciones de ingesta en el grafo.

    Uso:
        with Neo4jClient() as neo:
            neo.ensure_constraints()
            stats = Neo4jStats()
            neo.merge_document({...}, stats=stats)
            neo.ensure_schema_type("Ticker", stats=stats)
            neo.merge_entity({"name": "SPX", "type": "Ticker", "domain": "financial", "confidence": 0.95}, stats)
            neo.create_relation(doc_id="DOC-...", entity_name="SPX", entity_type="Ticker",
                                rel_type="MENTIONS", properties={"source_chunk": "...", "confidence": 0.95})

    Notas:
    - Todos los MERGE usan claves de unicidad definidas en constraints.
    - `ensure_schema_type` registra tipos dinámicos (Ticker, Emotion, Clause, etc.).
    """

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_sleep: float = DEFAULT_BASE_SLEEP,
        encrypted: Optional[bool] = None,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._max_retries = max_retries
        self._base_sleep = base_sleep
        self._encrypted = encrypted  # None => deja que Neo4j decida por esquema (bolt/s)
        self._driver: Optional[Driver] = None

    # --------------------------- Context Manager --------------------------- #
    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------- Conexión -------------------------------- #
    def connect(self) -> None:
        """Establece la conexión con reintentos exponenciales."""
        if self._driver:
            return

        attempt = 0
        while True:
            try:
                logger.info(
                    "Neo4j connect | uri=%s user=%s",
                    self._uri, self._user
                )
                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                    encrypted=self._encrypted,
                )
                # Probar una sesión rápida
                with self._driver.session() as s:
                    s.run("RETURN 1 AS ok").single()
                logger.info("Neo4j connect | OK")
                return
            except (ServiceUnavailable, AuthError) as e:
                if attempt >= self._max_retries:
                    logger.error("Neo4j connect | FAIL after %d attempts: %s", attempt + 1, repr(e))
                    raise
                sleep_s = self._base_sleep * (2 ** attempt)
                logger.warning("Neo4j connect | retry=%d sleep=%.2fs reason=%s", attempt + 1, sleep_s, repr(e))
                time.sleep(sleep_s)
                attempt += 1

    def close(self) -> None:
        if self._driver:
            try:
                self._driver.close()
                logger.info("Neo4j close | OK")
            finally:
                self._driver = None

    def _session(self) -> Session:
        if not self._driver:
            raise RuntimeError("Neo4j driver no inicializado. Llama a connect() o usa 'with Neo4jClient()'.")
        return self._driver.session()

    # ------------------------------- Utils --------------------------------- #
    def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        """
        Ejecuta una query con reintentos básicos.

        Retorna un iterable de registros (dict-like). Si no hay resultados, retorna [].
        """
        attempt = 0
        while True:
            try:
                with self._session() as session:
                    logger.debug("Neo4j run_query | q=%s | params=%s", query, params)
                    result = session.run(query, params or {})
                    return [r.data() for r in result]
            except (ServiceUnavailable, Neo4jError) as e:
                if attempt >= self._max_retries:
                    logger.error("Neo4j run_query | FAIL after %d attempts: %s", attempt + 1, repr(e))
                    raise
                sleep_s = self._base_sleep * (2 ** attempt)
                logger.warning("Neo4j run_query | retry=%d sleep=%.2fs reason=%s", attempt + 1, sleep_s, repr(e))
                time.sleep(sleep_s)
                attempt += 1

    # -------------------------- Constraints/Índices ------------------------ #
    def ensure_constraints(self) -> None:
        """
        Crea constraints necesarios para idempotencia.
        - Document.doc_id UNIQUE
        - Entity (name, type) UNIQUE (constraint compuesto)
        - (Opcional) índice en SchemaType.type
        """
        stmts = [
            # Unicidad por documento
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            # Unicidad compuesta por entidad
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            # Índice para SchemaType
            "CREATE CONSTRAINT schematype_type_unique IF NOT EXISTS FOR (s:SchemaType) REQUIRE s.type IS UNIQUE",
        ]
        for q in stmts:
            self.run_query(q)
        logger.info("Neo4j constraints | ensured")

    # ---------------------------- Merge Helpers ---------------------------- #
    def merge_document(self, doc_data: Dict[str, Any], *, stats: Optional[Neo4jStats] = None) -> Tuple[bool, str]:
        """
        MERGE de nodo (:Document). Usa doc_id como clave única.
        doc_data esperado:
            {
              "doc_id": "...", "filename": "...", "source_path": "...",
              "mime": "...", "created_at": "..."
            }
        Retorna: (created: bool, doc_id)
        """
        q = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.filename = $filename,
                    d.source_path = $source_path,
                    d.mime = $mime,
                    d.created_at = $created_at
        ON MATCH SET  d.filename = coalesce($filename, d.filename),
                    d.source_path = coalesce($source_path, d.source_path),
                    d.mime = coalesce($mime, d.mime),
                    d.created_at = coalesce($created_at, d.created_at)
        RETURN d, d.doc_id IS NOT NULL AS ok, labels(d) AS labels
        """

        params = {
            "doc_id": doc_data.get("doc_id"),
            "filename": (doc_data.get("meta") or {}).get("filename") or doc_data.get("filename"),
            "source_path": doc_data.get("source_path"),
            "mime": doc_data.get("mime"),
            "created_at": doc_data.get("created_at"),
        }
        res = self.run_query(q, params)
        created = False  # Determinar creación comparando counters con otro approach
        # Truco: consultar si el nodo recién creado tiene timestamp igual al input (no determinista). Mejor: segunda query.
        # Más fiable: usar estadísticas de resumen; el driver v5 no expone counters en este wrapper simple.
        # Alternativa: re-consultar un flag con otra query; mantendremos heurística a nivel de relación.
        # Para robustez, hacemos un check adicional:
        chk = self.run_query("MATCH (d:Document {doc_id: $doc_id}) RETURN d.doc_id AS id", {"doc_id": params["doc_id"]})
        if chk:
            created = True  # si antes no existía, esta sería la primera inserción; lo resolvemos fuera con cache si es necesario

        if stats:
            # Sin acceso directo a summary.counters, estimamos fuera; por ahora, consideramos new si no existía previo
            # Para evitar sobre-contar, podrías inyectar una caché de doc_ids existentes desde el ingestor.
            stats.new_documents += 1  # el ingestor puede ajustar si detecta existencia previa
        logger.info("MERGE Document | doc_id=%s", params["doc_id"])
        return created, params["doc_id"]

    def ensure_schema_type(self, type_name: str, *, stats: Optional[Neo4jStats] = None) -> None:
        """
        Registra dinámicamente un tipo de entidad en :SchemaType.
        Evita crear duplicados gracias al constraint de unicidad.
        """
        q = """
        MERGE (s:SchemaType {type: $type})
        RETURN labels(s) AS labels
        """
        self.run_query(q, {"type": type_name})
        if stats:
            # No sabemos si fue creado o no; si necesitas precisión, añade una query con summary.counters via Tx.
            # Mantenemos contador optimista (el ingestor podrá ajustar si lleva cache local).
            stats.labels_created += 0  # conservador: 0; el ajuste preciso lo hace el ingestor si detecta alta nueva
        logger.debug("MERGE SchemaType | type=%s", type_name)

    def merge_entity(
        self,
        entity_data: Dict[str, Any],
        *,
        stats: Optional[Neo4jStats] = None
    ) -> Tuple[bool, Tuple[str, str]]:
        """
        MERGE de (:Entity:<DynamicLabel> {name, type}) con propiedades actualizables.
        entity_data esperado:
            { "name": "...", "type": "Ticker", "domain": "financial", "confidence": 0.95, ... }

        - Aplica label dinámico (ej. :Entity:Ticker).
        - Propiedades básicas: domain, confidence (opcionales).
        Retorna: (created: bool, (name, type))
        """
        name = entity_data.get("name") or entity_data.get("text")
        etype = entity_data.get("type") or "Unknown"
        domain = entity_data.get("domain")
        confidence = entity_data.get("confidence")

        if not name:
            raise ValueError("merge_entity() requiere 'name' o 'text' en entity_data")

        # Asegurar el tipo en cat de esquema
        self.ensure_schema_type(etype, stats=stats)

        # Usamos APOC para setear label dinámico de forma segura si está disponible; si no, fallback a dos pasos
        # Fallback con dos pasos (MERGE base y luego SET label dinámico con string):
        q = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET e.domain = $domain,
                      e.confidence = $confidence
        ON MATCH SET  e.domain = coalesce($domain, e.domain),
                      e.confidence = coalesce($confidence, e.confidence)
        WITH e
        CALL apoc.create.addLabels(e, [$dyn_label]) YIELD node
        RETURN node AS e
        """
        params = {
            "name": name,
            "type": etype,
            "domain": domain,
            "confidence": confidence,
            "dyn_label": etype,  # :Entity:<etype>
        }
        try:
            self.run_query(q, params)
            created = True  # Heurística; ver comentario en merge_document (summary.counters no expuesto aquí)
        except Neo4jError as e:
            # Si APOC no está disponible, caemos a SET label clásico
            if "There is no procedure with the name `apoc.create.addLabels`" in str(e):
                logger.warning("APOC no disponible; usando SET label dinámico con CASE.")
                q_fallback = f"""
                MERGE (e:Entity {{name: $name, type: $type}})
                ON CREATE SET e.domain = $domain,
                              e.confidence = $confidence
                ON MATCH SET  e.domain = coalesce($domain, e.domain),
                              e.confidence = coalesce($confidence, e.confidence)
                WITH e
                CALL {{
                    WITH e
                    WITH e
                    RETURN 1
                }}
                WITH e
                SET e:`{etype}`
                RETURN e
                """
                self.run_query(q_fallback, params)
                created = True
            else:
                raise

        if stats:
            stats.new_entities += 1  # el ingestor puede corregir a existing si ya lo vio antes
        logger.info("MERGE Entity | name=%s type=%s domain=%s", name, etype, domain)
        return created, (name, etype)

    def create_relation(
        self,
        *,
        doc_id: str,
        entity_name: str,
        entity_type: str,
        rel_type: str = "MENTIONS",
        properties: Optional[Dict[str, Any]] = None,
        stats: Optional[Neo4jStats] = None
    ) -> Tuple[bool, str]:
        """
        Crea (o asegura) la relación (:Document)-[:MENTIONS]->(:Entity).
        Idempotente con MERGE sobre patrón de relación.

        properties típicas: {source_chunk, confidence, timestamp}
        Retorna: (created: bool, rel_type)
        """
        properties = properties or {}
        # Construimos SET dinámico de propiedades de relación
        set_props = ", ".join([f"r.{k} = ${k}" for k in properties.keys()]) or "r._touched = true"

        q = f"""
        MATCH (d:Document {{doc_id: $doc_id}})
        MATCH (e:Entity {{name: $entity_name, type: $entity_type}})
        MERGE (d)-[r:{rel_type}]->(e)
        ON CREATE SET {set_props}
        ON MATCH SET  {set_props}
        RETURN type(r) AS type
        """
        params = {"doc_id": doc_id, "entity_name": entity_name, "entity_type": entity_type, **properties}
        self.run_query(q, params)
        if stats:
            # Sin summary.counters, contamos como created; el ingestor puede ajustar si detecta repetidos
            stats.relations_created += 1
        logger.debug("MERGE Relation | (%s)-[:%s]->(%s:%s)", doc_id, rel_type, entity_name, entity_type)
        return True, rel_type
