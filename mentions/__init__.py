# -*- coding: utf-8 -*-
"""
mentions — Extracción de Entidades (LLM-only, Schema-Aware)
===========================================================

Módulo que implementa la etapa "Mentions" del pipeline T2G.

Propósito:
----------
Extraer menciones de entidades (Person, Organization, Date, Contract, etc.)
desde los documentos ya enriquecidos con contexto y esquemas seleccionados.

Entrada:
    - outputs_schema/*.json  (Adaptive Schema Selector)
    - outputs_chunks/*.json  (HybridChunker + Contextizer)

Salida:
    - outputs_mentions/{doc_id}_mentions.json

Este módulo es LLM-only, schema-aware (usa el registry), y fallback-safe.
"""
