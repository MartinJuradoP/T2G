#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t2g_cli.py — CLI unificado del proyecto T2G

Subcomandos:
- parse            Parser → DocumentIR(.json)
- chunk            DocumentIR(.json) → DocumentChunks(.json)
- sentences        DocumentChunks(.json) → DocumentSentences(.json)
- triples          DocumentSentences(.json) → DocumentTriples(.json)
- mentions         DocumentSentences(.json) → DocumentMentions(.json)
- ie               Orquestador por doc: (triples si faltan) → mentions con boost
- pipeline-yaml    Pipeline declarativo desde YAML

Puntos clave:
- `mentions` NO recalcula `triples`. Si pasas --boost-from-triples, reusa esas salidas.
- `ie` por documento: si NO existe el triple → lo calcula; luego corre mentions con boost.
- Paralelización a nivel de documento (ThreadPoolExecutor).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# YAML es opcional (solo para pipeline-yaml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ==============================
# Imports de los subsistemas
# ==============================

# Parser & Chunker
from parser.parsers import Parser as DocParser
from parser.schemas import DocumentIR
from chunker import HybridChunker, ChunkerConfig

# Sentence/Filter
from sentence_filter.sentence_filter import SentenceFilter, SentenceFilterConfig
from sentence_filter.schemas import DocumentSentences as DSModel  # contrato de entrada

# Triples
from triples.dep_triples import DepTripleExtractor, DepTripleConfig

# Mentions (NER/RE)
from mentions.ner_re import NERREExtractor, MentionConfig


# ============================================================================
# UTILIDADES COMUNES
# ============================================================================

def _write_json(path: Path, obj: Any) -> None:
    """
    Serializa un objeto Pydantic v2 (o un dict) a JSON con pretty-print UTF-8.
    Escritura atómica para evitar archivos vacíos/corruptos si el proceso se interrumpe.
    """
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump(mode="json")
    else:
        payload = obj
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _expand_inputs(args: Dict[str, Any], key_glob: str, key_list: str) -> List[str]:
    """
    Resuelve entradas a partir de:
      - key_glob: patrón(es) glob (str o list[str])
      - key_list: rutas directas (str o list[str])
    Retorna lista ordenada y sin duplicados.
    """
    results: List[str] = []

    patterns = args.get(key_glob)
    if patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            results.extend(sorted(glob.glob(pattern)))

    listed = args.get(key_list)
    if listed:
        if isinstance(listed, str):
            results.append(listed)
        else:
            results.extend(listed)

    return sorted(list(dict.fromkeys(results)))


def _ns(**kwargs) -> argparse.Namespace:
    """Atajo para crear un argparse.Namespace desde kwargs (útil en YAML runner)."""
    return argparse.Namespace(**kwargs)


def _maybe_clean(outdir: str, tag: str) -> None:
    """
    Limpia (rm -rf) un directorio si existe, con logs homogéneos.
    - No borra la carpeta raíz, solo su contenido.
    """
    if not outdir:
        return
    p_out = Path(outdir)
    if not p_out.exists():
        return
    print(f"[{tag}] limpiando carpeta: {outdir}")
    for p in p_out.iterdir():
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"[{tag}] aviso: no se pudo eliminar {p}: {e}")


# ============================================================================
# SUBCOMANDO: PARSE
# ============================================================================

def cmd_parse(args: argparse.Namespace) -> None:
    """Parser → DocumentIR(.json)"""
    parser = DocParser()
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
    """DocumentIR(.json) → DocumentChunks(.json)"""
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
    """DocumentChunks(.json) → DocumentSentences(.json)"""
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

        from parser.schemas import DocumentChunks as _DC
        dc = _DC(**dc_json)
        ds = filt.sentences_from_chunks(dc)

        out_path = outdir / f"{dc.doc_id}_sentences.json"
        _write_json(out_path, ds)
        kept = len(ds.sentences)
        keep_rate = ds.meta.get("counters", {}).get("keep_rate") or 0.0
        print(f"[SENTENCES OK] {p} → {out_path}  (#sents={kept}, keep_rate={keep_rate:.2f})")


# ============================================================================
# SUBCOMANDO: TRIPLES
# ============================================================================

def cmd_triples(args: argparse.Namespace) -> None:
    """DocumentSentences(.json) → DocumentTriples(.json)"""
    from json import JSONDecodeError

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Mapear “auto|force|off” → None|True|False para DepTripleConfig
    use_spacy_flag = None if args.spacy == "auto" else (True if args.spacy == "force" else False)
    cfg = DepTripleConfig(
        use_spacy=use_spacy_flag,
        lang_pref=str(args.lang or "auto"),
        ruleset=str(args.ruleset or "default-bilingual"),
        max_triples_per_sentence=int(args.max_triples_per_sentence or 4),
        drop_pronoun_subjects=not bool(getattr(args, "keep_pronouns", False)),
        enable_ner=bool(getattr(args, "enable_ner", False)),
        canonicalize_relations=not bool(getattr(args, "no_canonicalize_relations", False)),
        min_conf_keep=args.min_conf_keep,
    )
    extractor = DepTripleExtractor(cfg)

    processed = 0
    for spath in args.sent_files:
        p = Path(spath)
        if p.suffix.lower() != ".json":
            print(f"[TRIPLES SKIP] {p} (no .json)")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[TRIPLES ERR] {p}: JSON inválido ({e})")
            continue

        # Validar DocumentSentences (robusto)
        try:
            ds = DSModel(**data)
        except Exception:
            try:
                ds = DSModel.model_validate({
                    "doc_id": data.get("doc_id", "DOC-UNKNOWN"),
                    "sentences": data.get("sentences", []),
                    "meta": data.get("meta", {}),
                })
            except Exception as e2:
                print(f"[TRIPLES ERR] {p}: DocumentSentences inválido ({e2})")
                continue

        dt = extractor.extract_document(ds)
        out_path = outdir / f"{ds.doc_id}_triples.json"
        _write_json(out_path, dt)

        c = dt.meta.get("counters", {})
        print(f"[TRIPLES OK] {p} → {out_path}  (#triples={len(dt.triples)}, used_sents={c.get('used_sents', 0)})")
        processed += 1

    print(f"[TRIPLES OK] archivos={processed} → {outdir}")


# ============================================================================
# SUBCOMANDO: MENTIONS (NER/RE)
# ============================================================================

def cmd_mentions(args: argparse.Namespace) -> None:
    """
    DocumentSentences(.json) → DocumentMentions(.json)
    - Reusa triples previos si se indica --boost-from-triples (glob o ruta exacta).
    - NO recalcula triples en este subcomando.
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = MentionConfig(
        use_spacy=args.spacy,  # 'auto' | 'force' | 'off'
        use_transformers=bool(args.use_transformers),
        lang_pref=args.lang,
        min_conf_keep=args.min_conf_keep,
        max_relations_per_sentence=args.max_relations_per_sentence,
        canonicalize_labels=not bool(args.no_canonicalize_labels),
        boost_from_triples_glob=args.boost_from_triples,
        boost_conf=args.boost_conf,
        # HF opcional
        hf_rel_model_path=args.hf_rel_model_path,
        hf_device=args.hf_device,
        hf_batch_size=getattr(args, "hf_batch_size", 16),
        hf_min_prob=getattr(args, "hf_min_prob", 0.55),
        transformer_weight=getattr(args, "transformer_weight", 0.30),
        triples_boost_weight=getattr(args, "triples_boost_weight", 1.0),
    )
    extractor = NERREExtractor(cfg)

    processed = 0
    for spath in args.sent_files:
        p = Path(spath)
        if p.suffix.lower() != ".json":
            print(f"[MENTIONS SKIP] {p} (no .json)")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[MENTIONS ERR] {p}: JSON inválido ({e})")
            continue

        # Validar DocumentSentences (o aceptar dict)
        try:
            ds = DSModel(**data)
            ds_payload = ds.model_dump(mode="json")
        except Exception:
            ds_payload = data  # extractor acepta dict equivalente

        dm = extractor.extract_document(ds_payload)
        out_path = outdir / p.name.replace("_sentences.json", "_mentions.json")
        _write_json(out_path, dm)
        print(f"[MENTIONS OK] {p} → {out_path}  (#entities={len(dm.entities)}, #relations={len(dm.relations)})")
        processed += 1

    print(f"[MENTIONS OK] archivos={processed} → {outdir}")


# ============================================================================
# SUBCOMANDO: IE (orquestador por-doc)
# ============================================================================

def cmd_ie(args: argparse.Namespace) -> None:
    """
    Orquestador por documento:
      1) Usa --sents-glob si existe; si no y hay --chunks-glob, genera sentences.
      2) Reutiliza triples si existen; si faltan (o limpiaste), los genera.
      3) Corre mentions (NER/RE) con:
         - consenso/boost desde triples (si se pasan o existen),
         - re-rank local HF opcional (sin red).
      4) (Opcional) Valida con tools/validate_ie.py.
    """
    from json import JSONDecodeError

    # --- Preparar rutas destino
    sent_dir = Path(args.outdir_sentences)
    tri_dir  = Path(args.outdir_triples)
    men_dir  = Path(args.outdir_mentions)
    for d in (sent_dir, tri_dir, men_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Limpiezas previas
    if getattr(args, "clean_outdir_mentions", False):
        _maybe_clean(args.outdir_mentions, "ie.mentions")
    if getattr(args, "clean_outdir_triples", False):
        _maybe_clean(args.outdir_triples, "ie.triples")

    # --- Resolver oraciones de entrada
    sents_glob = getattr(args, "sents_glob", None) or getattr(args, "sent_glob", None)
    chunks_glob = getattr(args, "chunks_glob", None)

    sents_paths: List[str] = []

    if sents_glob:
        # Caso A: el usuario ya pasó oraciones
        sents_paths = sorted(glob.glob(sents_glob))
        if not sents_paths:
            print(f"[IE] ⚠️ No hay archivos que coincidan con --sents-glob='{sents_glob}'.")
            return

    elif chunks_glob:
        # Caso B: generar oraciones desde chunks
        chunk_paths = sorted(glob.glob(chunks_glob))
        if not chunk_paths:
            print(f"[IE] ⚠️ No hay archivos que coincidan con --chunks-glob='{chunks_glob}'.")
            return
        # Generar sentences aquí
        ns_sent = _ns(
            chunk_files=chunk_paths,
            outdir=str(sent_dir),
            sentence_splitter=args.spacy,
            min_chars=25,
            dedupe="fuzzy",
            fuzzy_threshold=0.92,
            no_normalize_whitespace=False,
            no_dehyphenate=False,
            no_strip_bullets=False,
            keep_stopword_only=False,
            keep_numeric_only=False,
        )
        cmd_sentences(ns_sent)
        sents_paths = [str(p) for p in sorted(sent_dir.glob("*_sentences.json"))]

    else:
        # Caso C: intentar reusar outdir_sentences por defecto
        default_glob = str(sent_dir / "*_sentences.json")
        sents_paths = sorted(glob.glob(default_glob))
        if not sents_paths:
            print(
                "[IE] ⚠️ No se proporcionó --sents-glob ni --chunks-glob, "
                f"y no se encontraron oraciones en '{default_glob}'."
            )
            return

    if not sents_paths:
        print("[IE] ⚠️ No se encontraron oraciones para procesar.")
        return

    # --- Config de Triples
    use_spacy_flag = None if args.spacy == "auto" else (True if args.spacy == "force" else False)
    tcfg = DepTripleConfig(
        use_spacy=use_spacy_flag,
        lang_pref=args.lang,
        ruleset="default-bilingual",
        max_triples_per_sentence=4,
        drop_pronoun_subjects=True,
        enable_ner=False,
        canonicalize_relations=True,
        min_conf_keep=args.min_conf_keep_triples,
    )
    triple_extractor = DepTripleExtractor(tcfg)

    # --- Config de Mentions
    # Si el usuario no pasó boost_from_triples, por defecto usa la carpeta destino
    boost_glob = getattr(args, "boost_from_triples", None)
    if not boost_glob:
        boost_glob = str(Path(args.outdir_triples) / "*_triples.json")

    mcfg = MentionConfig(
        use_spacy=("off" if args.spacy == "off" else ("force" if args.spacy == "force" else "auto")),
        lang_pref=args.lang,
        min_conf_keep=float(getattr(args, "min_conf_keep_mentions", 0.66)),
        canonicalize_labels=bool(getattr(args, "canonicalize_labels", True)),
        max_relations_per_sentence=int(getattr(args, "max_relations_per_sentence", 6)),
        # Consenso/boost con triples
        boost_from_triples_glob=boost_glob,
        boost_conf=float(getattr(args, "boost_conf", 0.05)),
        triples_boost_weight=float(getattr(args, "triples_boost_weight", 1.0)),
        # HF opcional
        use_transformers=bool(getattr(args, "use_transformers", False)),
        hf_rel_model_path=getattr(args, "hf_rel_model_path", None),
        hf_device=str(getattr(args, "hf_device", "cpu") or "cpu"),
        hf_batch_size=int(getattr(args, "hf_batch_size", 16)),
        hf_min_prob=float(getattr(args, "hf_min_prob", 0.55)),
        transformer_weight=float(getattr(args, "transformer_weight", 0.30)),
    )
    mention_extractor = NERREExtractor(mcfg)

    print(f"[IE] Procesando {len(sents_paths)} documentos (paralelo por doc, workers={args.workers})")

    def _process_one(spath: str) -> Tuple[str, int, int]:
        """Procesa un DocumentSentences → (triples si faltan) + mentions. Retorna contadores."""
        p = Path(spath)
        # Cargar DocumentSentences con fallback suave
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[IE] ⚠️ JSON inválido en {p.name}: {e}")
            return (p.stem, 0, 0)

        try:
            ds = DSModel(**data)
        except Exception:
            # Fallback: intentar normalizar a DSModel
            try:
                ds = DSModel.model_validate({
                    "doc_id": data.get("doc_id", p.stem.replace("_sentences","")),
                    "sentences": data.get("sentences", []),
                    "meta": data.get("meta", {}),
                })
            except Exception as e2:
                print(f"[IE] ⚠️ {p.name} no es DocumentSentences válido: {e2}")
                return (p.stem, 0, 0)

        doc_id = ds.doc_id

        # 1) Triples: si faltan (o limpiaste), generarlos
        triples_path = tri_dir / f"{doc_id}_triples.json"
        if not triples_path.exists():
            dt = triple_extractor.extract_document(ds)
            _write_json(triples_path, dt)
            c = dt.meta.get("counters", {})
            print(f"[TRIPLES OK] {p.name} → {triples_path}  (#triples={len(dt.triples)}, used_sents={c.get('used_sents', 0)})")

        # 2) Mentions
        dm = mention_extractor.extract_document(ds)
        mentions_path = men_dir / f"{doc_id}_mentions.json"
        _write_json(mentions_path, dm)
        print(f"[MENTIONS OK] {p.name} → {mentions_path}  (#entities={len(dm.entities)}, #relations={len(dm.relations)})")

        return (doc_id, len(getattr(dm, "entities", [])), len(getattr(dm, "relations", [])))

    # --- Paralelismo por documento
    ents_total = 0
    rels_total = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_process_one, sp) for sp in sents_paths]
        for fut in as_completed(futs):
            try:
                _, ecount, rcount = fut.result()
                ents_total += ecount
                rels_total += rcount
            except Exception as e:
                print(f"[IE] ⚠️ Error procesando doc: {e}")

    print(f"[IE DONE] triples: {tri_dir} | mentions: {men_dir} | totals → entities={ents_total}, relations={rels_total}")

    # --- Validación global (opcional)
    if args.validate:
        cmd = [
            "python", "tools/validate_ie.py",
            "--mentions", str(men_dir),
            "--triples",  str(tri_dir),
        ]
        print("[IE] Validating:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[IE] ⚠️ validate_ie.py error: {e}")


# ============================================================================
# SUBCOMANDO: PIPELINE-YAML (declarativo)
# ============================================================================

def cmd_pipeline_yaml(args: argparse.Namespace) -> None:
    """
    Ejecuta un pipeline definido en YAML (por defecto: pipelines/pipeline.yaml).
    Soporta etapas: parse, chunk, sentences, triples, mentions, ie
    """
    if yaml is None:
        raise RuntimeError("PyYAML no está instalado. Ejecuta: pip install pyyaml")

    ypath = Path(args.file or "pipelines/pipeline.yaml")
    if not ypath.exists():
        raise FileNotFoundError(f"No existe el YAML: {ypath}")

    conf = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    pipeline_conf = conf.get("pipeline", {}) or {}
    stages_conf = conf.get("stages", []) or []

    dry_run = bool(pipeline_conf.get("dry_run", False))
    continue_on_error = bool(pipeline_conf.get("continue_on_error", True))

    stage_registry: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    # ---- Stage runners ----

    def run_parse(stage_args: Dict[str, Any]) -> None:
        inputs = _expand_inputs(stage_args, "inputs_glob", "inputs")
        outdir = stage_args.get("outdir", "outputs_ir")
        if stage_args.get("clean_outdir"):
            if dry_run: print(f"[DRY RUN] parse (clean) outdir={outdir}")
            else: _maybe_clean(outdir, "parse")
        if dry_run:
            print(f"[DRY RUN] parse inputs={len(inputs)} → outdir={outdir}")
            return
        if not inputs:
            print("[parse] ⚠️ No hay inputs. Saltando.")
            return
        ns = _ns(inputs=inputs, outdir=outdir)
        cmd_parse(ns)

    def run_chunk(stage_args: Dict[str, Any]) -> None:
        ir_files = _expand_inputs(stage_args, "ir_glob", "ir_files")
        ir_files = [p for p in ir_files if str(p).lower().endswith(".json")]
        outdir = stage_args.get("outdir", "outputs_chunks")
        if stage_args.get("clean_outdir"):
            if dry_run: print(f"[DRY RUN] chunk (clean) outdir={outdir}")
            else: _maybe_clean(outdir, "chunk")
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
            print(f"[DRY RUN] chunk ir_files={len(ir_files)} → outdir={outdir}")
            return
        if not ir_files:
            print("[chunk] ⚠️ No hay ir_files. Saltando.")
            return
        cmd_chunk(ns)

    def run_sentences(stage_args: Dict[str, Any]) -> None:
        chunk_files = _expand_inputs(stage_args, "chunks_glob", "chunk_files")
        chunk_files = [p for p in chunk_files if str(p).lower().endswith(".json")]
        outdir = stage_args.get("outdir", "outputs_sentences")
        if stage_args.get("clean_outdir"):
            if dry_run: print(f"[DRY RUN] sentences (clean) outdir={outdir}")
            else: _maybe_clean(outdir, "sentences")
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
            print(f"[DRY RUN] sentences chunks={len(chunk_files)} → outdir={outdir}")
            return
        if not chunk_files:
            print("[sentences] ⚠️ No hay chunk_files. Saltando.")
            return
        cmd_sentences(ns)

    def run_triples(stage_args: Dict[str, Any]) -> None:
        sent_files = _expand_inputs(stage_args, "sents_glob", "sent_files")
        sent_files = [p for p in sent_files if str(p).lower().endswith(".json")]
        outdir = stage_args.get("outdir", "outputs_triples")
        if stage_args.get("clean_outdir"):
            if dry_run: print(f"[DRY RUN] triples (clean) outdir={outdir}")
            else: _maybe_clean(outdir, "triples")

        canonicalize_rel = bool(stage_args.get("canonicalize_relations", True))
        ns = _ns(
            sent_files=sent_files,
            outdir=outdir,
            lang=str(stage_args.get("lang", "auto")),
            ruleset=str(stage_args.get("ruleset", "default-bilingual")),
            spacy=str(stage_args.get("spacy", "auto")),
            max_triples_per_sentence=int(stage_args.get("max_triples_per_sentence", 4)),
            keep_pronouns=bool(stage_args.get("keep_pronouns", False)),
            enable_ner=bool(stage_args.get("enable_ner", False)),
            no_canonicalize_relations=not canonicalize_rel,
            min_conf_keep=stage_args.get("min_conf_keep", None),
        )
        if dry_run:
            print(f"[DRY RUN] triples sents={len(sent_files)} → outdir={outdir}")
            return
        if not sent_files:
            print("[triples] ⚠️ No hay sent_files. Saltando.")
            return
        cmd_triples(ns)

    def run_mentions(stage_args: Dict[str, Any]) -> None:
        sent_files = _expand_inputs(stage_args, "sent_glob", "sent_files")
        sent_files = [p for p in sent_files if str(p).lower().endswith(".json")]
        outdir = stage_args.get("outdir", "outputs_mentions")
        if stage_args.get("clean_outdir"):
            if dry_run: print(f"[DRY RUN] mentions (clean) outdir={outdir}")
            else: _maybe_clean(outdir, "mentions")

        canonicalize_labels = bool(stage_args.get("canonicalize_labels", True))
        ns = _ns(
            sent_files=sent_files,
            outdir=outdir,
            lang=str(stage_args.get("lang", "auto")),
            spacy=str(stage_args.get("spacy", "auto")),
            min_conf_keep=float(stage_args.get("min_conf_keep", 0.66)),
            max_relations_per_sentence=int(stage_args.get("max_relations_per_sentence", 6)),
            no_canonicalize_labels=not canonicalize_labels,
            boost_from_triples=stage_args.get("boost_from_triples", None),
            boost_conf=float(stage_args.get("boost_conf", 0.05)),
            # HF opcional (para mantener homogeneidad con IE)
            use_transformers=bool(stage_args.get("use_transformers", False)),
            hf_rel_model_path=stage_args.get("hf_rel_model_path", None),
            hf_device=stage_args.get("hf_device", "cpu"),
            hf_batch_size=int(stage_args.get("hf_batch_size", 16)),
            hf_min_prob=float(stage_args.get("hf_min_prob", 0.55)),
            transformer_weight=float(stage_args.get("transformer_weight", 0.30)),
            triples_boost_weight=float(stage_args.get("triples_boost_weight", 1.0)),
        )
        if dry_run:
            print(f"[DRY RUN] mentions sents={len(sent_files)} → outdir={outdir}")
            return
        if not sent_files:
            print("[mentions] ⚠️ No hay sent_files. Saltando.")
            return
        cmd_mentions(ns)

    def run_ie(stage_args: Dict[str, Any]) -> None:
        """
        Orquestador por doc:
        - Usa sents_glob si está; si no, usa chunks_glob para generar oraciones primero.
        - Por doc: (reusa o crea triples) → corre mentions con boost (y opcional HF).
        """
        ns = _ns(
            # fuentes
            chunks_glob=stage_args.get("chunks_glob", None),
            sents_glob=stage_args.get("sents_glob", None) or stage_args.get("sent_glob", None),

            # destinos
            outdir_sentences=str(stage_args.get("outdir_sentences", "outputs_sentences")),
            outdir_triples=str(stage_args.get("outdir_triples", "outputs_triples")),
            outdir_mentions=str(stage_args.get("outdir_mentions", "outputs_mentions")),

            # config general
            lang=str(stage_args.get("lang", "auto")),
            spacy=str(stage_args.get("spacy", "auto")),

            # umbrales / límites
            min_conf_keep_triples=stage_args.get("min_conf_keep_triples", None),
            min_conf_keep_mentions=float(stage_args.get("min_conf_keep_mentions", 0.66)),
            max_relations_per_sentence=int(stage_args.get("max_relations_per_sentence", 6)),
            canonicalize_labels=bool(stage_args.get("canonicalize_labels", True)),

            # consenso/boost
            boost_conf=float(stage_args.get("boost_conf", 0.05)),
            boost_from_triples=stage_args.get("boost_from_triples", None),
            triples_boost_weight=float(stage_args.get("triples_boost_weight", 1.0)),

            # HF opcional
            use_transformers=bool(stage_args.get("use_transformers", False)),
            hf_rel_model_path=stage_args.get("hf_rel_model_path", None),
            hf_device=str(stage_args.get("hf_device", "cpu")),
            hf_batch_size=int(stage_args.get("hf_batch_size", 16)),
            hf_min_prob=float(stage_args.get("hf_min_prob", 0.55)),
            transformer_weight=float(stage_args.get("transformer_weight", 0.30)),

            # ejecución
            workers=int(stage_args.get("workers", 4)),
            validate=bool(stage_args.get("validate", False)),

            # limpieza (la ejecuta cmd_ie)
            clean_outdir_triples=bool(stage_args.get("clean_outdir_triples", False)),
            clean_outdir_mentions=bool(stage_args.get("clean_outdir_mentions", True)),
        )

        if dry_run:
            print(f"[DRY RUN] ie  sents_glob={bool(ns.sents_glob)} chunks_glob={bool(ns.chunks_glob)} "
                  f"→ triples={ns.outdir_triples} | mentions={ns.outdir_mentions}")
            return

        cmd_ie(ns)

    # Registrar etapas YAML
    stage_registry.update({
        "parse": run_parse,
        "chunk": run_chunk,
        "sentences": run_sentences,
        "triples": run_triples,
        "mentions": run_mentions,
        "ie": run_ie,
    })

    # Ejecutar en orden
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
# CONSTRUCCIÓN DEL CLI
# ============================================================================

def build_t2g_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="t2g",
        description="T2G Pipeline CLI — parse · chunk · sentences · triples · mentions · ie · pipeline-yaml",
    )
    cmds = cli.add_subparsers(dest="cmd", required=True)

    # --- parse ---
    parse_cmd = cmds.add_parser("parse", help="Parsea documentos a IR JSON")
    parse_cmd.add_argument("inputs", nargs="+", help="Rutas de documentos (PDF/DOCX/IMG)")
    parse_cmd.add_argument("--outdir", default="outputs_ir", help="Carpeta de salida de IRs")
    parse_cmd.set_defaults(func=cmd_parse)

    # --- chunk ---
    chunk_cmd = cmds.add_parser("chunk", help="Genera chunks desde IR (.json)")
    chunk_cmd.add_argument("ir_files", nargs="+", help="Rutas de IR JSON")
    chunk_cmd.add_argument("--outdir", default="outputs_chunks")
    chunk_cmd.add_argument("--target-chars", type=int, default=1400)
    chunk_cmd.add_argument("--max-chars", type=int, default=2048)
    chunk_cmd.add_argument("--min-chars", type=int, default=400)
    chunk_cmd.add_argument("--overlap", type=int, default=120)
    chunk_cmd.add_argument("--table-policy", choices=["isolate", "merge"], default="isolate")
    chunk_cmd.add_argument("--sentence-splitter", choices=["auto", "spacy", "regex"], default="auto")
    chunk_cmd.set_defaults(func=cmd_chunk)

    # --- sentences ---
    sent_cmd = cmds.add_parser("sentences", help="Genera oraciones desde Chunks (.json)")
    sent_cmd.add_argument("chunk_files", nargs="+", help="Rutas de Chunks JSON")
    sent_cmd.add_argument("--outdir", default="outputs_sentences")
    sent_cmd.add_argument("--sentence-splitter", choices=["auto","spacy","regex"], default="auto")
    sent_cmd.add_argument("--min-chars", type=int, default=25)
    sent_cmd.add_argument("--dedupe", choices=["none","exact","fuzzy"], default="fuzzy")
    sent_cmd.add_argument("--fuzzy-threshold", type=float, default=0.92)
    sent_cmd.add_argument("--no-normalize-whitespace", action="store_true")
    sent_cmd.add_argument("--no-dehyphenate", action="store_true")
    sent_cmd.add_argument("--no-strip-bullets", action="store_true")
    sent_cmd.add_argument("--keep-stopword-only", action="store_true")
    sent_cmd.add_argument("--keep-numeric-only", action="store_true")
    sent_cmd.set_defaults(func=cmd_sentences)

    # --- triples ---
    triples_cmd = cmds.add_parser("triples", help="Genera triples (S,R,O) desde Sentences (.json)")
    triples_cmd.add_argument("sent_files", nargs="+", help="Rutas de DocumentSentences JSON")
    triples_cmd.add_argument("--outdir", default="outputs_triples")
    triples_cmd.add_argument("--lang", default="auto", choices=["auto","es","en"])
    triples_cmd.add_argument("--ruleset", default="default-bilingual")
    triples_cmd.add_argument("--spacy", default="auto", choices=["auto","force","off"])
    triples_cmd.add_argument("--max-triples-per-sentence", type=int, default=4)
    triples_cmd.add_argument("--keep-pronouns", action="store_true")
    triples_cmd.add_argument("--continue-on-error", action="store_true", default=True)
    triples_cmd.add_argument("--enable-ner", action="store_true")
    triples_cmd.add_argument("--no-canonicalize-relations", action="store_true")
    triples_cmd.add_argument("--min-conf-keep", type=float, default=None)
    triples_cmd.set_defaults(func=cmd_triples)

    # --- mentions ---
    mentions_cmd = cmds.add_parser("mentions", help="Genera menciones (NER/RE) desde Sentences (.json)")
    mentions_cmd.add_argument("sent_files", nargs="+", help="Rutas de DocumentSentences JSON")
    mentions_cmd.add_argument("--outdir", default="outputs_mentions")
    mentions_cmd.add_argument("--lang", default="auto", choices=["auto","es","en"])
    mentions_cmd.add_argument("--spacy", default="auto", choices=["auto","force","off"])
    mentions_cmd.add_argument("--min-conf-keep", type=float, default=0.66)
    mentions_cmd.add_argument("--max-relations-per-sentence", type=int, default=6)
    mentions_cmd.add_argument("--no-canonicalize-labels", action="store_true")
    mentions_cmd.add_argument("--boost-from-triples", default=None, help="Glob o ruta exacta a triples para boost")
    mentions_cmd.add_argument("--boost-conf", type=float, default=0.05)
    # HF opcional (mismos flags que IE para homogeneidad)
    mentions_cmd.add_argument("--use-transformers", action="store_true")
    mentions_cmd.add_argument("--hf-rel-model-path", default=None)
    mentions_cmd.add_argument("--hf-device", default="cpu")
    mentions_cmd.add_argument("--hf-batch-size", type=int, default=16)
    mentions_cmd.add_argument("--hf-min-prob", type=float, default=0.55)
    mentions_cmd.add_argument("--transformer-weight", type=float, default=0.30)
    mentions_cmd.add_argument("--triples-boost-weight", type=float, default=1.0)
    mentions_cmd.set_defaults(func=cmd_mentions)

    # --- ie (orquestador) ---
    ie_cmd = cmds.add_parser("ie", help="Orquesta por-doc: (triples si faltan) → mentions (con boost)")
    ie_cmd.add_argument("--chunks-glob", default=None, help="Si no hay oraciones, usa esto para generarlas")
    ie_cmd.add_argument("--sents-glob", dest="sents_glob", default=None, help="Glob de oraciones si ya existen")
    ie_cmd.add_argument("--outdir-sentences", default="outputs_sentences")
    ie_cmd.add_argument("--outdir-triples", default="outputs_triples")
    ie_cmd.add_argument("--outdir-mentions", default="outputs_mentions")
    ie_cmd.add_argument("--lang", default="auto", choices=["auto","es","en"])
    ie_cmd.add_argument("--spacy", default="auto", choices=["auto","force","off"])
    # Umbrales / límites
    ie_cmd.add_argument("--min-conf-keep-triples", type=float, default=None)
    ie_cmd.add_argument("--min-conf-keep-mentions", type=float, default=0.66)
    ie_cmd.add_argument("--max-relations-per-sentence", type=int, default=6)
    ie_cmd.add_argument("--canonicalize-labels", action="store_true", default=True)
    # Boost
    ie_cmd.add_argument("--boost-conf", type=float, default=0.05)
    ie_cmd.add_argument("--boost-from-triples", default=None,
                        help="Glob de triples para boost; por defecto usa {outdir_triples}/*_triples.json")
    ie_cmd.add_argument("--triples-boost-weight", type=float, default=1.0,
                        help="Multiplicador aplicado al boost_conf cuando hay consenso con triples [1.0]")
    # HF opcional
    ie_cmd.add_argument("--use-transformers", action="store_true")
    ie_cmd.add_argument("--hf-rel-model-path", default=None)
    ie_cmd.add_argument("--hf-device", default="cpu")
    ie_cmd.add_argument("--hf-batch-size", type=int, default=16)
    ie_cmd.add_argument("--hf-min-prob", type=float, default=0.55)
    ie_cmd.add_argument("--transformer-weight", type=float, default=0.30)
    # Paralelismo / validación / limpieza
    ie_cmd.add_argument("--workers", type=int, default=4)
    ie_cmd.add_argument("--validate", action="store_true", default=False)
    ie_cmd.add_argument("--clean-outdir-triples", action="store_true", default=False)
    ie_cmd.add_argument("--clean-outdir-mentions", action="store_true", default=False)
    ie_cmd.set_defaults(func=cmd_ie)

    # --- pipeline-yaml ---
    py_cmd = cmds.add_parser("pipeline-yaml", help="Ejecuta un pipeline desde YAML")
    py_cmd.add_argument("--file", default="pipelines/pipeline.yaml",
                        help="Ruta al YAML (default: pipelines/pipeline.yaml)")
    py_cmd.set_defaults(func=cmd_pipeline_yaml)

    return cli


def main() -> None:
    cli = build_t2g_cli()
    args = cli.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
