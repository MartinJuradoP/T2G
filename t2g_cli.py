#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t2g_cli.py — CLI unificado del proyecto T2G
===========================================

Etapas implementadas hoy:
- parse              Parser (PDF/DOCX/IMG) → DocumentIR(.json)
- contextize-doc     Añade contexto global al IR
- chunk              Segmenta IR+Topics en DocumentChunks (heredando contexto)
- pipeline-yaml      Ejecuta pipeline declarativo en YAML

Etapas preparadas (stub, no implementadas aún):
- contextize-chunks  Añade contexto local a chunks
"""

from __future__ import annotations
import argparse
import glob
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------
from parser.parsers import Parser as DocParser
from contextizer.contextizer import run_contextizer_on_doc, TopicModelConfig
from contextizer.contextizer import run_contextizer_on_chunks
from chunker import run as run_chunker, load_ir_with_topics, save_chunks
from chunker.schemas import ChunkingConfig
# Adaptive Schema Selector
from schema_selector.selector import select_schemas
from schema_selector.registry import REGISTRY
from schema_selector.schemas import SelectorConfig


# Stubs (para no romper imports; implementar en siguientes entregas)
def run_contextizer_on_chunks_stub(ch_path: str, cfg) -> None:
    logger.warning("[CONTEXTIZER-CHUNKS] 🚧 Stub activo. Aún no implementado. Input: %s", ch_path)

# ---------------------------------------------------------------------------
# Logging global
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("t2g_cli")

# ============================================================================
# Helpers
# ============================================================================
def _expand_inputs(args: Dict[str, Any], key_glob: str, key_list: str) -> List[str]:
    """Expande rutas con globs y listas explícitas."""
    results: List[str] = []
    patterns = args.get(key_glob)
    if patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            results.extend(sorted(glob.glob(pattern)))
    listed = args.get(key_list)
    if listed:
        results.extend([listed] if isinstance(listed, str) else listed)
    # dedup y orden estable
    return sorted(list(dict.fromkeys(results)))


def _maybe_clean(outdir: str | Path, tag: str) -> None:
    """Elimina contenidos previos de un directorio."""
    p_out = Path(outdir)
    if not p_out.exists():
        return
    logger.info("[%s] limpiando carpeta: %s", tag, outdir)
    for f in p_out.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

# ============================================================================
# Commands
# ============================================================================
def cmd_parse(args: argparse.Namespace) -> None:
    if args.clean_outdir:
        _maybe_clean(args.outdir, "parse")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    parser = DocParser()
    for path in args.inputs:
        doc_ir = parser.parse(path)
        out_path = Path(args.outdir) / f"{doc_ir.doc_id}.json"
        out_path.write_text(
            json.dumps(doc_ir.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("[PARSE OK] %s → %s", path, out_path)


def cmd_contextize_doc(args: argparse.Namespace) -> None:
    outdir = Path(getattr(args, "outdir", "outputs_doc_topics"))
    if getattr(args, "clean_outdir", False):
        _maybe_clean(outdir, "contextize-doc")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TopicModelConfig(
        embedding_model=args.embedding_model,
        nr_topics=(None if args.nr_topics in (None, -1) else args.nr_topics),
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    for ir in args.ir_files:
        run_contextizer_on_doc(ir, cfg, outdir=outdir)


def cmd_chunk(args: argparse.Namespace) -> None:
    """Ejecuta el HybridChunker sobre IR+Topics (outputs_doc_topics)."""
    outdir = Path(getattr(args, "outdir", "outputs_chunks"))
    if getattr(args, "clean_outdir", False):
        _maybe_clean(outdir, "chunk")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ChunkingConfig(
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
        use_embeddings=args.use_embeddings,
        embedding_model=args.embedding_model,
        spacy_model=args.spacy_model,
        seed=args.seed,
    )

    for ir_path in args.ir_files:
        ir = load_ir_with_topics(ir_path)
        chunks = run_chunker(ir, cfg)
        save_chunks(chunks, outdir)


def cmd_contextize_chunks(args: argparse.Namespace) -> None:
    """Contextualización a nivel chunk (usa BERTopic + fallbacks)."""
    cfg = TopicModelConfig(
        embedding_model=args.embedding_model,
        nr_topics=(None if args.nr_topics in (None, -1) else args.nr_topics),
        seed=args.seed,
        cache_dir=args.cache_dir if hasattr(args, "cache_dir") else None,
    )
    for ch in args.chunk_files:
        run_contextizer_on_chunks(ch, cfg)


def cmd_schema_select(args: argparse.Namespace) -> None:
    """
    Ejecuta Adaptive Schema Selector 2.0.
    - Entrada: DocumentChunks con topics (outputs_chunks/*.json)
    - Salida:  SchemaSelection JSON (outputs_schema/*.json)
    """
    outdir = Path(getattr(args, "outdir", "outputs_schema"))
    if getattr(args, "clean_outdir", False):
        _maybe_clean(outdir, "schema-select")
    outdir.mkdir(parents=True, exist_ok=True)

    # Configuración del selector v2
    cfg = SelectorConfig(
        alpha_kw=args.alpha_kw,
        beta_emb=args.beta_emb,
        gamma_ctx=args.gamma_ctx,
        delta_top=args.delta_top,
        epsilon_prior=args.epsilon_prior,
        ambiguity_threshold=args.ambig_threshold,
        fallback_threshold=args.fallback_threshold,
        softmax_temperature=args.softmax_temp,
        topk=args.topk_domains,
        max_domains=args.max_domains,
        allow_fallback_generic=not args.disable_generic,
    )

    for ch_path in args.chunk_files:
        ch_path = Path(ch_path)
        if not ch_path.exists():
            logger.warning("[SCHEMA-SELECT] Archivo no encontrado: %s", ch_path)
            continue

        doc = json.loads(ch_path.read_text(encoding="utf-8"))
        schema_sel = select_schemas(doc, registry=REGISTRY, config=cfg)

        out_path = outdir / f"{doc.get('doc_id', 'UNKNOWN')}_schema.json"
        out_path.write_text(
            json.dumps(schema_sel.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("[SCHEMA-SELECT OK] %s → %s", ch_path, out_path)

# ============================================================================
# Pipeline YAML
# ============================================================================
def cmd_pipeline_yaml(args: argparse.Namespace) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML requerido (pip install pyyaml)")

    conf = yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}
    stages = conf.get("stages", [])

    for idx, stage in enumerate(stages, 1):
        name, sargs = stage["name"], stage.get("args", {})
        logger.info("[%d] ▶ Ejecutando etapa: %s", idx, name)

        if name == "parse":
            inputs = _expand_inputs(sargs, "inputs_glob", "inputs")
            ns = argparse.Namespace(
                inputs=inputs,
                outdir=sargs.get("outdir", "outputs_ir"),
                clean_outdir=sargs.get("clean_outdir", False),
            )
            cmd_parse(ns)

        elif name == "contextize-doc":
            ir_files = _expand_inputs(sargs, "ir_glob", "ir_files")
            ns = argparse.Namespace(
                ir_files=ir_files,
                embedding_model=sargs.get("embedding_model", "all-MiniLM-L6-v2"),
                nr_topics=sargs.get("nr_topics", None),
                seed=sargs.get("seed", 42),
                cache_dir=sargs.get("cache_dir", None),
                outdir=sargs.get("outdir", "outputs_doc_topics"),
                clean_outdir=sargs.get("clean_outdir", False),
            )
            cmd_contextize_doc(ns)

        elif name == "chunk":
            ir_files = _expand_inputs(sargs, "ir_glob", "ir_files")
            ns = argparse.Namespace(
                ir_files=ir_files,
                outdir=sargs.get("outdir", "outputs_chunks"),
                clean_outdir=sargs.get("clean_outdir", False),
                max_tokens=sargs.get("max_tokens", 2048),
                min_chars=sargs.get("min_chars", 280),
                use_embeddings=sargs.get("use_embeddings", True),
                embedding_model=sargs.get("embedding_model", "all-MiniLM-L6-v2"),
                spacy_model=sargs.get("spacy_model", "es_core_news_sm"),
                seed=sargs.get("seed", 42),
            )
            cmd_chunk(ns)

        elif name == "contextize-chunks":
            chunk_files = _expand_inputs(sargs, "chunks_glob", "chunk_files")
            ns = argparse.Namespace(
                chunk_files=chunk_files,
                embedding_model=sargs.get("embedding_model", "all-MiniLM-L6-v2"),
                nr_topics=sargs.get("nr_topics", None),       
                seed=sargs.get("seed", 42),
                cache_dir=sargs.get("cache_dir", None),       
                outdir=sargs.get("outdir", "outputs_chunks")
            )
            cmd_contextize_chunks(ns)
        
        elif name == "schema-select":
            chunk_files = _expand_inputs(sargs, "chunks_glob", "chunk_files")
            ns = argparse.Namespace(
                chunk_files=chunk_files,
                outdir=sargs.get("outdir", "outputs_schema"),
                clean_outdir=sargs.get("clean_outdir", False),

                # --- Pesos de señales ---
                alpha_kw=sargs.get("alpha_kw", 0.30),
                beta_emb=sargs.get("beta_emb", 0.25),
                gamma_ctx=sargs.get("gamma_ctx", 0.25),
                delta_top=sargs.get("delta_top", 0.15),
                epsilon_prior=sargs.get("epsilon_prior", 0.05),

                # --- Umbrales ---
                ambig_threshold=sargs.get("ambig_threshold", 0.12),
                fallback_threshold=sargs.get("fallback_threshold", 0.40),
                softmax_temp=sargs.get("softmax_temp", 0.85),
                disable_generic=sargs.get("disable_generic", False),

                # --- Control de dominios ---
                topk_domains=sargs.get("topk_domains", 5),
                max_domains=sargs.get("max_domains", 5),
            )
            cmd_schema_select(ns)



# ============================================================================
# CLI Entrypoint
# ============================================================================
def build_t2g_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="t2g", description="T2G Pipeline CLI")
    cmds = cli.add_subparsers(dest="cmd", required=True)

    # parse
    p = cmds.add_parser("parse", help="Parsea documentos a IR JSON")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--outdir", default="outputs_ir")
    p.add_argument("--clean-outdir", action="store_true")
    p.set_defaults(func=cmd_parse)

    # contextize-doc
    c = cmds.add_parser("contextize-doc", help="Añade contexto global al IR")
    c.add_argument("ir_files", nargs="+")
    c.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    c.add_argument("--nr-topics", type=int, default=None, dest="nr_topics")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--outdir", default="outputs_doc_topics")
    c.add_argument("--clean-outdir", action="store_true")
    c.set_defaults(func=cmd_contextize_doc)

    # chunk
    ch = cmds.add_parser("chunk", help="Segmenta IR+Topics en chunks")
    ch.add_argument("ir_files", nargs="+")
    ch.add_argument("--outdir", default="outputs_chunks")
    ch.add_argument("--clean-outdir", action="store_true")
    ch.add_argument("--max-tokens", type=int, default=2048)
    ch.add_argument("--min-chars", type=int, default=280)
    ch.add_argument("--use-embeddings", action="store_true", default=True)
    ch.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ch.add_argument("--spacy-model", default="es_core_news_sm")
    ch.add_argument("--seed", type=int, default=42)
    ch.set_defaults(func=cmd_chunk)

    # contextize-chunks (stub)
    cc = cmds.add_parser("contextize-chunks", help="Añade contexto local a chunks")
    cc.add_argument("chunk_files", nargs="+")
    cc.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    cc.add_argument("--nr-topics", type=int, default=None, dest="nr_topics")
    cc.add_argument("--seed", type=int, default=42)
    cc.add_argument("--cache-dir", default=None)
    cc.set_defaults(func=cmd_contextize_chunks)

    # schema-select
    ss = cmds.add_parser("schema-select", help="Ejecuta Adaptive Schema Selector 2.0")
    ss.add_argument("chunk_files", nargs="+", help="Archivos JSON de DocumentChunks")
    ss.add_argument("--outdir", default="outputs_schema", help="Directorio de salida")
    ss.add_argument("--clean-outdir", action="store_true", help="Limpiar directorio antes de generar salida")

    # --- Pesos de señales ---
    ss.add_argument("--alpha-kw", type=float, default=0.30, help="Peso de keywords (léxica)")
    ss.add_argument("--beta-emb", type=float, default=0.25, help="Peso de embeddings (semántica densa)")
    ss.add_argument("--gamma-ctx", type=float, default=0.25, help="Peso de métricas contextuales (cohesión, health, novelty)")
    ss.add_argument("--delta-top", type=float, default=0.15, help="Peso de afinidad de tópicos globales/locales")
    ss.add_argument("--epsilon-prior", type=float, default=0.05, help="Peso de priors (sesgos organizacionales)")

    # --- Umbrales y control ---
    ss.add_argument("--ambig-threshold", type=float, default=0.12, help="Margen de ambigüedad (|S1−S2|<τ)")
    ss.add_argument("--fallback-threshold", type=float, default=0.40, help="Confianza mínima antes de fallback genérico")
    ss.add_argument("--softmax-temp", type=float, default=0.85, help="Temperatura del softmax (confianza)")
    ss.add_argument("--disable-generic", action="store_true", help="Desactivar fallback genérico")

    # --- Control de dominios ---
    ss.add_argument("--topk-domains", type=int, default=5, help="Cuántos dominios se devuelven en doc-level")
    ss.add_argument("--max-domains", type=int, default=5, help="Máximo de dominios a evaluar")

    ss.set_defaults(func=cmd_schema_select)



    # pipeline-yaml
    py = cmds.add_parser("pipeline-yaml", help="Ejecuta pipeline desde YAML")
    py.add_argument("--file", default="pipelines/pipeline.yaml")
    py.set_defaults(func=cmd_pipeline_yaml)

    return cli


def main():
    cli = build_t2g_cli()
    args = cli.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
