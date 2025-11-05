# graph_builder/graph_ingestor.py
# -*- coding: utf-8 -*-
"""
Graph Ingestor — Etapa de construcción de grafo (T2G)
======================================================

Responsabilidad:
----------------
- Cargar archivos JSON de outputs_ir/ y outputs_mentions/
- Validar correspondencia doc_id → menciones
- Insertar documentos, entidades y relaciones en Neo4j
- Mantener métricas e idempotencia (sin duplicados)
- Registrar resultados en outputs_graph/graph_metrics.json
"""

from __future__ import annotations
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from graph_builder.neo4j_client import Neo4jClient, Neo4jStats
from graph_builder.metrics import append_metrics_record
from graph_builder.utils import load_json_safe

logger = logging.getLogger("graph_builder.ingestor")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def ingest_graph(
    ir_dir: str = "outputs_ir",
    mentions_dir: str = "outputs_mentions",
    outdir: str = "outputs_graph",
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta la ingesta completa en Neo4j combinando outputs_ir + outputs_mentions.

    Devuelve un resumen de métricas globales para persistir en graph_metrics.json.
    """

    t0 = time.time()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    graph_metrics_path = Path(outdir) / "graph_metrics.json"

    ir_files = sorted(Path(ir_dir).glob("*.json"))
    mentions_files = sorted(Path(mentions_dir).glob("*.json"))
    if not ir_files or not mentions_files:
        logger.warning("No hay archivos IR o Mentions disponibles.")
        return {}

    # Cache local para idempotencia y conteos finos
    seen_docs = set()
    seen_entities = set()  # (name, type)
    seen_rels = set()      # (doc_id, name, type)

    n_docs_total = 0
    errors = []

    with Neo4jClient(uri=neo4j_uri or os.getenv("NEO4J_URI"),
                     user=neo4j_user or os.getenv("NEO4J_USER"),
                     password=neo4j_password or os.getenv("NEO4J_PASSWORD")) as neo:

        neo.ensure_constraints()

        for ir_path in ir_files:
            try:
                doc_ir = load_json_safe(ir_path)
                doc_id = doc_ir.get("doc_id")
                if not doc_id:
                    logger.warning("Archivo IR sin doc_id: %s", ir_path)
                    continue

                # Buscar menciones correspondientes
                mentions_path = next(
                    (m for m in mentions_files if Path(m).stem.startswith(doc_id)), None
                )
                if not mentions_path or not Path(mentions_path).exists():
                    logger.info("Sin menciones para %s", doc_id)
                    continue

                mentions_data = load_json_safe(mentions_path)
                mentions = mentions_data.get("mentions", [])
                if not mentions:
                    logger.info("Documento %s no tiene menciones válidas", doc_id)
                    continue

                n_docs_total += 1
                stats = Neo4jStats()
                logger.info("[GRAPH] Ingestando doc_id=%s | %d menciones", doc_id, len(mentions))

                # 1️⃣ Documento
                if doc_id not in seen_docs:
                    neo.merge_document(doc_ir, stats=stats)
                    seen_docs.add(doc_id)
                else:
                    stats.existing_documents += 1

                # 2️⃣ Entidades y relaciones
                for m in mentions:
                    name = m.get("text") or m.get("name")
                    etype = m.get("type") or "Unknown"
                    key = (name, etype)
                    rel_key = (doc_id, name, etype)

                    # Evitar duplicados
                    if not name:
                        continue

                    # Crear entidad
                    if key not in seen_entities:
                        neo.merge_entity(m, stats=stats)
                        seen_entities.add(key)
                    else:
                        stats.existing_entities += 1

                    # Crear relación
                    if rel_key not in seen_rels:
                        neo.create_relation(
                            doc_id=doc_id,
                            entity_name=name,
                            entity_type=etype,
                            rel_type="MENTIONS",
                            properties={
                                "source_chunk": m.get("source_chunk"),
                                "confidence": m.get("confidence"),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                            stats=stats,
                        )
                        seen_rels.add(rel_key)
                    else:
                        stats.relations_reused += 1

                # 3️⃣ Métricas por documento
                elapsed = round(time.time() - t0, 2)
                record = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "doc_id": doc_id,
                    "n_new_documents": stats.new_documents,
                    "n_existing_documents": stats.existing_documents,
                    "n_new_entities": stats.new_entities,
                    "n_existing_entities": stats.existing_entities,
                    "n_relations_created": stats.relations_created,
                    "n_relations_reused": stats.relations_reused,
                    "n_labels_created": stats.labels_created,
                    "errors": [],
                    "elapsed_time": elapsed,
                }
                append_metrics_record(graph_metrics_path, record)
                logger.info("[GRAPH]  %s Ingestado con éxito", doc_id)

            except Exception as e:
                err_msg = f"Error procesando {ir_path.name}: {repr(e)}"
                logger.error(err_msg)
                errors.append(err_msg)
                if not continue_on_error:
                    raise

    total_time = round(time.time() - t0, 2)
    summary = {
        "total_docs": n_docs_total,
        "total_entities": len(seen_entities),
        "total_relations": len(seen_rels),
        "errors": errors,
        "elapsed_time": total_time,
    }
    logger.info("[GRAPH] Finalizado | docs=%d entities=%d relations=%d | %.2fs",
                n_docs_total, len(seen_entities), len(seen_rels), total_time)
    return summary
