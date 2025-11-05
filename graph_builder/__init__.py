# graph_builder/__init__.py
# -*- coding: utf-8 -*-
"""
graph_builder — Módulo de construcción del grafo T2G
====================================================

Incluye:
- neo4j_client.py: conexión y operaciones básicas
- graph_ingestor.py: lógica de ingesta de documentos y entidades
- metrics.py: registro y consolidación de métricas
- utils.py: utilidades de carga/validación de JSON
"""

__all__ = ["neo4j_client", "graph_ingestor", "metrics", "utils"]
