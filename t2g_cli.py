#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t2g_cli.py — CLI unificado del proyecto T2G

Subcomandos:
- parse           Parser → DocumentIR(.json)
- chunk           DocumentIR(.json) → DocumentChunks(.json)
- sentences       DocumentChunks(.json) → DocumentSentences(.json)
- pipeline        Ejecuta parse + chunk en una sola corrida (imperativo)
- pipeline-yaml   Ejecuta un pipeline declarativo desde YAML

Requisitos:
- Estructura de paquetes con __init__.py en parser/ y chunker/
- Ejecutar desde la raíz del repo:  python t2g_cli.py <subcomando> ...
- Para pipeline-yaml: PyYAML (pip install pyyaml)

Sugerencias:
- Para cortes por oración con mejor calidad, instala spaCy y un modelo:
    pip install spacy
    python -m spacy download es_core_news_sm
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil          # ← se usa en limpieza de carpetas en pipeline-yaml
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

# YAML es opcional (solo para pipeline-yaml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Subsistemas actuales
from parser.parsers import Parser as DocParser
from parser.schemas import DocumentIR, DocumentChunks  # contratos de IR y Chunks
from chunker import HybridChunker, ChunkerConfig

# Etapa 3: Sentence/Filter
from sentence_filter.sentence_filter import SentenceFilter, SentenceFilterConfig
from sentence_filter.schemas import DocumentSentences


# ============================================================================
# UTILIDADES COMUNES
# ============================================================================

def _write_json(path: Path, obj: Any) -> None:
    """
    Serializa un objeto Pydantic v2 (o un dict) a JSON con pretty-print UTF-8.
    Escritura atómica para evitar archivos vacíos/corruptos si el proceso se interrumpe.
    """
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump(mode="json")  # baja datetime/otros a JSON-safe
    else:
        payload = obj
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    os.replace(tmp_path, path)  # atómico en POSIX


def _expand_inputs(args: Dict[str, Any], key_glob: str, key_list: str) -> List[str]:
    """
    Resuelve entradas a partir de:
      - key_glob: patrón(es) glob (str o list[str])
      - key_list: rutas directas (str o list[str])
    Retorna lista ordenada y sin duplicados.
    """
    results: List[str] = []

    # 1) Globs
    patterns = args.get(key_glob)
    if patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            results.extend(sorted(glob.glob(pattern)))

    # 2) Rutas directas
    listed = args.get(key_list)
    if listed:
        if isinstance(listed, str):
            results.append(listed)
        else:
            results.extend(listed)

    # 3) Dedup + orden estable
    return sorted(list(dict.fromkeys(results)))


def _ns(**kwargs) -> argparse.Namespace:
    """Atajo para crear un argparse.Namespace desde kwargs (útil en YAML runner)."""
    return argparse.Namespace(**kwargs)


# ============================================================================
# SUBCOMANDO: PARSE
# ============================================================================

def cmd_parse(args: argparse.Namespace) -> None:
    """
    Parser → DocumentIR(.json)
    - Lee documentos heterogéneos (PDF/DOCX/IMG).
    - Emite una IR homogénea (DocumentIR) en JSON.
    """
    parser = DocParser(
        # Mapea flags si deseas exponerlos en el CLI:
        # ocr_lang=args.ocr_lang,
        # ocr_resolution=args.ocr_dpi,
        # enable_pdf_ocr_fallback=not args.no_pdf_ocr_fallback,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        doc_ir = parser.parse(path)
        out_path = outdir / f"{doc_ir.doc_id}.json"
        _write_json(out_path, doc_ir)
        print(f"[PARSE OK] {path} → {out_path}  (pages={len(doc_ir.pages)})")


# ============================================================================
# SUBCOMANDO: CHUNK
# ============================================================================

def cmd_chunk(args: argparse.Namespace) -> None:
    """
    DocumentIR(.json) → DocumentChunks(.json)
    """
    from json import JSONDecodeError

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ChunkerConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap,
        table_policy=args.table_policy,
        sentence_splitter=args.sentence_splitter,
    )
    chunker = HybridChunker(cfg)

    for ir_path in args.ir_files:
        p = Path(ir_path)

        # Saltar archivos no .json o vacíos
        if p.suffix.lower() != ".json":
            print(f"[SKIP] {p} (no .json)")
            continue
        try:
            if p.stat().st_size == 0:
                print(f"[SKIP] {p} (archivo vacío)")
                continue
        except FileNotFoundError:
            print(f"[SKIP] {p} (no existe)")
            continue

        # Cargar IR con manejo de errores
        try:
            ir_json = json.loads(p.read_text(encoding="utf-8"))
        except JSONDecodeError as e:
            print(f"[SKIP] {p} (JSON inválido: {e})")
            continue

        doc_ir = DocumentIR(**ir_json)
        doc_chunks = chunker.chunk_document(doc_ir)

        out_path = outdir / f"{doc_ir.doc_id}_chunks.json"
        _write_json(out_path, doc_chunks)
        print(f"[CHUNK OK] {ir_path} → {out_path}  (#chunks={len(doc_chunks.chunks)})")


# ============================================================================
# SUBCOMANDO: SENTENCE FILTER
# ============================================================================

def cmd_sentences(args: argparse.Namespace) -> None:
    """
    DocumentChunks(.json) → DocumentSentences(.json)
    - Divide chunks en oraciones y filtra ruido configurable.
    """
    from json import JSONDecodeError

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SentenceFilterConfig(
        sentence_splitter=args.sentence_splitter,
        normalize_whitespace=not args.no_normalize_whitespace,
        dehyphenate=not args.no_dehyphenate,
        strip_bullets=not args.no_strip_bullets,
        min_chars=args.min_chars,
        drop_stopword_only=not args.keep_stopword_only,
        drop_numeric_only=not args.keep_numeric_only,
        dedupe=args.dedupe,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    filt = SentenceFilter(cfg)

    for path in args.chunk_files:
        p = Path(path)
        if p.suffix.lower() != ".json":
            print(f"[SKIP] {p} (no .json)")
            continue
        try:
            if p.stat().st_size == 0:
                print(f"[SKIP] {p} (archivo vacío)")
                continue
        except FileNotFoundError:
            print(f"[SKIP] {p} (no existe)")
            continue

        try:
            dc_json = json.loads(p.read_text(encoding="utf-8"))
        except JSONDecodeError as e:
            print(f"[SKIP] {p} (JSON inválido: {e})")
            continue

        # Cargar y procesar
        from parser.schemas import DocumentChunks as _DC  # evitar colisiones de nombres
        dc = _DC(**dc_json)
        ds = filt.sentences_from_chunks(dc)

        out_path = outdir / f"{dc.doc_id}_sentences.json"
        _write_json(out_path, ds)
        kept = len(ds.sentences)
        keep_rate = ds.meta.get("counters", {}).get("keep_rate") or 0.0  # robustez
        print(f"[SENTENCES OK] {p} → {out_path}  (#sents={kept}, keep_rate={keep_rate:.2f})")


# ============================================================================
# SUBCOMANDO: PIPELINE (imperativo, parse+chunk)
# ============================================================================

def cmd_pipeline(args: argparse.Namespace) -> None:
    """
    Ejecuta parse y luego chunk sobre los mismos documentos, guardando ambas salidas.
    Ideal para una corrida rápida sin YAML.
    """
    ir_outdir = Path(args.ir_outdir)
    ch_outdir = Path(args.chunks_outdir)
    ir_outdir.mkdir(parents=True, exist_ok=True)
    ch_outdir.mkdir(parents=True, exist_ok=True)

    # 1) Parse
    parser = DocParser()
    ir_paths: List[Path] = []
    for path in args.inputs:
        doc_ir = parser.parse(path)
        ir_path = ir_outdir / f"{doc_ir.doc_id}.json"
        _write_json(ir_path, doc_ir)
        ir_paths.append(ir_path)
        print(f"[PARSE OK] {path} → {ir_path}  (pages={len(doc_ir.pages)})")

    # 2) Chunk
    cfg = ChunkerConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap,
        table_policy=args.table_policy,
        sentence_splitter=args.sentence_splitter,
    )
    chunker = HybridChunker(cfg)

    for irp in ir_paths:
        ir_json = json.loads(irp.read_text(encoding="utf-8"))
        doc_ir = DocumentIR(**ir_json)
        doc_chunks = chunker.chunk_document(doc_ir)

        out_path = ch_outdir / f"{doc_ir.doc_id}_chunks.json"
        _write_json(out_path, doc_chunks)
        print(f"[CHUNK OK] {irp} → {out_path}  (#chunks={len(doc_chunks.chunks)})")


# ============================================================================
# SUBCOMANDO: PIPELINE-YAML (declarativo)
# ============================================================================

def cmd_pipeline_yaml(args: argparse.Namespace) -> None:
    """
    Ejecuta un pipeline definido en YAML (por defecto: pipelines/pipeline.yaml).

    Estructura esperada del YAML:
    ---
    pipeline:
      dry_run: false            # si true, imprime lo que haría y no ejecuta
      continue_on_error: true   # si una etapa falla, continuar con las siguientes

    stages:
      - name: parse
        args:
          clean_outdir: true
          inputs_glob:
            - "docs/*.pdf"
            - "docs/*.png"
          outdir: "outputs_ir"

      - name: chunk
        args:
          ir_glob: "outputs_ir/*.json"
          outdir: "outputs_chunks"
          target_chars: 1400
          max_chars: 2048
          min_chars: 400
          overlap: 120
          table_policy: "isolate"       # isolate | merge
          sentence_splitter: "auto"     # auto | spacy | regex

      - name: sentences
        args:
          chunks_glob: "outputs_chunks/*.json"
          outdir: "outputs_sentences"
          sentence_splitter: "auto"
          min_chars: 25
          dedupe: "fuzzy"
          fuzzy_threshold: 0.92
          # normalización/filtros:
          no_normalize_whitespace: false
          no_dehyphenate: false
          no_strip_bullets: false
          keep_stopword_only: false
          keep_numeric_only: false
    """
    if yaml is None:
        raise RuntimeError("PyYAML no está instalado. Ejecuta: pip install pyyaml")

    # Por defecto usa pipelines/pipeline.yaml, o uno explícito si se pasa por --file
    ypath = Path(args.file or "pipelines/pipeline.yaml")
    if not ypath.exists():
        raise FileNotFoundError(f"No existe el YAML: {ypath}")

    conf = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    pipeline_conf = conf.get("pipeline", {}) or {}
    stages_conf = conf.get("stages", []) or []

    dry_run = bool(pipeline_conf.get("dry_run", False))
    continue_on_error = bool(pipeline_conf.get("continue_on_error", True))

    # Registro de etapas → funciones que mapean args YAML a Namespaces de subcomandos
    stage_registry: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def run_parse(stage_args: Dict[str, Any]) -> None:
        # 1) Entradas
        inputs = _expand_inputs(stage_args, "inputs_glob", "inputs")
        outdir = stage_args.get("outdir", "outputs_ir")

        # 2) Limpieza opcional del outdir
        if stage_args.get("clean_outdir"):
            if dry_run:
                print(f"[DRY RUN] parse  (limpieza) outdir={outdir}")
            else:
                p_out = Path(outdir)
                if p_out.exists():
                    print(f"[parse] limpiando carpeta: {outdir}")
                    for p in p_out.iterdir():
                        try:
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                shutil.rmtree(p)
                        except Exception as e:
                            print(f"[parse] aviso: no se pudo eliminar {p}: {e}")

        # 3) Dry run
        if dry_run:
            print(f"[DRY RUN] parse  inputs={len(inputs)} → outdir={outdir}")
            return

        # 4) Validación y ejecución
        if not inputs:
            print("[parse] ⚠️ No hay inputs (inputs_glob/inputs vacíos). Saltando.")
            return
        ns = _ns(inputs=inputs, outdir=outdir)
        cmd_parse(ns)

    def run_chunk(stage_args: Dict[str, Any]) -> None:
        ir_files = _expand_inputs(stage_args, "ir_glob", "ir_files")
        ir_files = [p for p in ir_files if str(p).lower().endswith(".json")]

        outdir = stage_args.get("outdir", "outputs_chunks")
        ns = _ns(
            ir_files=ir_files,
            outdir=outdir,
            target_chars=int(stage_args.get("target_chars", 1400)),
            max_chars=int(stage_args.get("max_chars", 2048)),
            min_chars=int(stage_args.get("min_chars", 400)),
            overlap=int(stage_args.get("overlap", 120)),
            table_policy=str(stage_args.get("table_policy", "isolate")),
            sentence_splitter=str(stage_args.get("sentence_splitter", "auto")),
        )
        if dry_run:
            print(f"[DRY RUN] chunk  ir_files={len(ir_files)} → outdir={outdir}")
            return
        if not ir_files:
            print("[chunk] ⚠️ No hay ir_files (ir_glob/ir_files vacíos). Saltando.")
            return
        cmd_chunk(ns)

    def run_sentences(stage_args: Dict[str, Any]) -> None:
        chunk_files = _expand_inputs(stage_args, "chunks_glob", "chunk_files")
        chunk_files = [p for p in chunk_files if str(p).lower().endswith(".json")]

        outdir = stage_args.get("outdir", "outputs_sentences")
        ns = _ns(
            chunk_files=chunk_files,
            outdir=outdir,
            sentence_splitter=str(stage_args.get("sentence_splitter", "auto")),
            min_chars=int(stage_args.get("min_chars", 25)),
            dedupe=str(stage_args.get("dedupe", "fuzzy")),
            fuzzy_threshold=float(stage_args.get("fuzzy_threshold", 0.92)),
            no_normalize_whitespace=bool(stage_args.get("no_normalize_whitespace", False)),
            no_dehyphenate=bool(stage_args.get("no_dehyphenate", False)),
            no_strip_bullets=bool(stage_args.get("no_strip_bullets", False)),
            keep_stopword_only=bool(stage_args.get("keep_stopword_only", False)),
            keep_numeric_only=bool(stage_args.get("keep_numeric_only", False)),
        )
        if dry_run:
            print(f"[DRY RUN] sentences  chunks={len(chunk_files)} → outdir={outdir}")
            return
        if not chunk_files:
            print("[sentences] ⚠️ No hay chunk_files (chunks_glob/chunk_files vacíos). Saltando.")
            return
        cmd_sentences(ns)

    # Registrar etapas disponibles
    stage_registry["parse"] = run_parse
    stage_registry["chunk"] = run_chunk
    stage_registry["sentences"] = run_sentences

    # Ejecutar en orden declarado
    for idx, stage in enumerate(stages_conf, start=1):
        name = str(stage.get("name", "")).strip()
        sargs = stage.get("args", {}) or {}
        if name not in stage_registry:
            print(f"[{idx}] ❌ Etapa desconocida: '{name}'. Saltando.")
            if not continue_on_error:
                break
            continue
        print(f"[{idx}] ▶ Ejecutando etapa: {name}")
        try:
            stage_registry[name](sargs)
        except Exception as e:
            print(f"[{idx}] ❌ Error en etapa '{name}': {e}")
            if not continue_on_error:
                raise


# ============================================================================
# CONSTRUCCIÓN DEL CLI (argparse)
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t2g",
        description="T2G Pipeline CLI — Parser + HybridChunker + Sentence/Filter + YAML runner",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- parse ---
    sp = sub.add_parser("parse", help="Parsea documentos a IR JSON")
    sp.add_argument("inputs", nargs="+", help="Rutas de documentos (PDF/DOCX/IMG)")
    sp.add_argument("--outdir", default="outputs_ir", help="Carpeta de salida de IRs")
    # (futuro) flags de OCR:
    # sp.add_argument("--ocr-lang", default="spa", help="Idiomas OCR (tesseract)")
    # sp.add_argument("--ocr-dpi", type=int, default=220, help="DPI para fallback OCR")
    # sp.add_argument("--no-pdf-ocr-fallback", action="store_true", help="Deshabilita OCR fallback en PDF")
    sp.set_defaults(func=cmd_parse)

    # --- chunk ---
    sc = sub.add_parser("chunk", help="Genera chunks desde IR (.json)")
    sc.add_argument("ir_files", nargs="+", help="Rutas IR JSON (salidas del parser)")
    sc.add_argument("--outdir", default="outputs_chunks", help="Carpeta de salida de chunks")
    sc.add_argument("--target-chars", type=int, default=1400, help="Tamaño objetivo del chunk")
    sc.add_argument("--max-chars", type=int, default=2048, help="Límite duro del chunk")
    sc.add_argument("--min-chars", type=int, default=400, help="Tamaño mínimo preferido")
    sc.add_argument("--overlap", type=int, default=120, help="Solapamiento entre chunks consecutivos")
    sc.add_argument("--table-policy", choices=["isolate", "merge"], default="isolate",
                    help="Aislar cada tabla como chunk o fusionarla con el contexto")
    sc.add_argument("--sentence-splitter", choices=["auto", "spacy", "regex"], default="auto",
                    help="Segmentador de oraciones (auto usa spaCy si está disponible)")
    sc.set_defaults(func=cmd_chunk)

    # --- sentences ---
    ss = sub.add_parser("sentences", help="Genera oraciones desde Chunks (.json)")
    ss.add_argument("chunk_files", nargs="+", help="Rutas de Chunks JSON (salidas del chunker)")
    ss.add_argument("--outdir", default="outputs_sentences", help="Carpeta de salida de oraciones")
    ss.add_argument("--sentence-splitter", choices=["auto","spacy","regex"], default="auto",
                    help="Segmentador de oraciones (auto usa spaCy si está disponible)")
    ss.add_argument("--min-chars", type=int, default=25, help="Descarta oraciones demasiado cortas")
    ss.add_argument("--dedupe", choices=["none","exact","fuzzy"], default="fuzzy",
                    help="Estrategia de deduplicación (exact o fuzzy)")
    ss.add_argument("--fuzzy-threshold", type=float, default=0.92, help="Umbral de similitud para fuzzy dedupe")
    # toggles de normalización/filtros
    ss.add_argument("--no-normalize-whitespace", action="store_true", help="No colapsar espacios")
    ss.add_argument("--no-dehyphenate", action="store_true", help="No unir palabras cortadas por guion")
    ss.add_argument("--no-strip-bullets", action="store_true", help="No remover bullets al inicio de línea")
    ss.add_argument("--keep-stopword-only", action="store_true", help="NO descartar oraciones de solo stopwords")
    ss.add_argument("--keep-numeric-only", action="store_true", help="NO descartar oraciones sin letras")
    ss.set_defaults(func=cmd_sentences)

    # --- pipeline (imperativo, parse+chunk) ---
    pp = sub.add_parser("pipeline", help="Parsea y chunquea en una corrida")
    pp.add_argument("inputs", nargs="+", help="Rutas de documentos (PDF/DOCX/IMG)")
    pp.add_argument("--ir-outdir", default="outputs_ir", help="Carpeta de salida IR")
    pp.add_argument("--chunks-outdir", default="outputs_chunks", help="Carpeta de salida chunks")
    pp.add_argument("--target-chars", type=int, default=1400)
    pp.add_argument("--max-chars", type=int, default=2048)
    pp.add_argument("--min-chars", type=int, default=400)
    pp.add_argument("--overlap", type=int, default=120)
    pp.add_argument("--table-policy", choices=["isolate", "merge"], default="isolate")
    pp.add_argument("--sentence-splitter", choices=["auto", "spacy", "regex"], default="auto")
    pp.set_defaults(func=cmd_pipeline)

    # --- pipeline-yaml (declarativo) ---
    py = sub.add_parser("pipeline-yaml", help="Ejecuta un pipeline desde YAML")
    py.add_argument("--file", default="pipelines/pipeline.yaml",
                    help="Ruta al YAML (default: pipelines/pipeline.yaml)")
    py.set_defaults(func=cmd_pipeline_yaml)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
