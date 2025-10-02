#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t2g_cli.py — CLI unificado del proyecto T2G
===========================================

Etapas implementadas hoy:
- parse              Parser (PDF/DOCX/IMG) → DocumentIR(.json)
- contextize-doc     Añade contexto global al IR
- pipeline-yaml      Ejecuta pipeline declarativo en YAML

Etapas preparadas (stub, no implementadas aún):
- chunk              Segmenta IR en chunks (heredando contexto global)
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
from contextizer.contextizer import run_contextizer_on_doc  # procedural (doc-level)
from contextizer.contextizer import TopicModelConfig
# Stubs (para no romper imports; implementar en siguientes entregas)
def run_chunker_stub(ir_path: str, outdir: str) -> None:
    logger.warning("[CHUNKER] 🚧 Stub activo. Aún no implementado. Input: %s → %s", ir_path, outdir)

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
    """Stub de chunker (no implementado aún)."""
    if args.clean_outdir:
        _maybe_clean(args.outdir, "chunk")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for ir in args.ir_files:
        run_chunker_stub(ir, args.outdir)


def cmd_contextize_chunks(args: argparse.Namespace) -> None:
    """Stub de contextizer-chunks (no implementado aún)."""
    cfg = TopicModelConfig(embedding_model=args.embedding_model, seed=args.seed)
    for ch in args.chunk_files:
        run_contextizer_on_chunks_stub(ch, cfg)

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
            )
            cmd_chunk(ns)

        elif name == "contextize-chunks":
            chunk_files = _expand_inputs(sargs, "chunks_glob", "chunk_files")
            ns = argparse.Namespace(
                chunk_files=chunk_files,
                embedding_model=sargs.get("embedding_model", "all-MiniLM-L6-v2"),
                seed=sargs.get("seed", 42),
            )
            cmd_contextize_chunks(ns)

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
    c.add_argument("--outdir", default="outputs_doc_topics")  # ← faltaba
    c.add_argument("--clean-outdir", action="store_true")
    c.set_defaults(func=cmd_contextize_doc)

    # chunk (stub)
    ch = cmds.add_parser("chunk", help="🚧 Stub — segmenta IR en chunks (no implementado aún)")
    ch.add_argument("ir_files", nargs="+")
    ch.add_argument("--outdir", default="outputs_chunks")
    ch.add_argument("--clean-outdir", action="store_true")
    ch.set_defaults(func=cmd_chunk)

    # contextize-chunks (stub)
    cc = cmds.add_parser("contextize-chunks", help="🚧 Stub — añade contexto local a chunks (no implementado aún)")
    cc.add_argument("chunk_files", nargs="+")
    cc.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    cc.add_argument("--seed", type=int, default=42)
    cc.set_defaults(func=cmd_contextize_chunks)

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
