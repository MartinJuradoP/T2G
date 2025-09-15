# --- (add near top) ---
import argparse
import json
from pathlib import Path

from schemas import DocumentIR  # ya existe
from schemas import DocumentChunks  # nuevo import para escribir salida de chunks
from chunker import HybridChunker, ChunkerConfig  # nuevo

# --- (dentro de tu main o equivalente) ---
def build_arg_parser():
    p = argparse.ArgumentParser(description="T2G Pipeline CLI (Parser + Chunker)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # parse command (ya lo tienes; ajusta nombres según tu CLI actual)
    p_parse = sub.add_parser("parse", help="Parsea documentos a IR JSON")
    p_parse.add_argument("inputs", nargs="+", help="Rutas de documentos a parsear")
    p_parse.add_argument("--outdir", default="outputs_ir", help="Directorio de salida IR")

    # chunk command: desde IR existentes
    p_chunk = sub.add_parser("chunk", help="Genera chunks desde archivos IR (.json)")
    p_chunk.add_argument("ir_files", nargs="+", help="Rutas IR JSON (salida del parser)")
    p_chunk.add_argument("--outdir", default="outputs_chunks", help="Directorio de salida de chunks")
    p_chunk.add_argument("--target-chars", type=int, default=1400)
    p_chunk.add_argument("--max-chars", type=int, default=2048)
    p_chunk.add_argument("--min-chars", type=int, default=400)
    p_chunk.add_argument("--overlap", type=int, default=120)
    p_chunk.add_argument("--table-policy", choices=["isolate","merge"], default="isolate")
    p_chunk.add_argument("--sentence-splitter", choices=["auto","spacy","regex"], default="auto")

    # pipeline command: parse + chunk en una corrida
    p_pipe = sub.add_parser("pipeline", help="Parsea y chunquea en una sola corrida")
    p_pipe.add_argument("inputs", nargs="+", help="Rutas de documentos a procesar")
    p_pipe.add_argument("--ir-outdir", default="outputs_ir", help="Salida IR")
    p_pipe.add_argument("--chunks-outdir", default="outputs_chunks", help="Salida chunks")
    p_pipe.add_argument("--target-chars", type=int, default=1400)
    p_pipe.add_argument("--max-chars", type=int, default=2048)
    p_pipe.add_argument("--min-chars", type=int, default=400)
    p_pipe.add_argument("--overlap", type=int, default=120)
    p_pipe.add_argument("--table-policy", choices=["isolate","merge"], default="isolate")
    p_pipe.add_argument("--sentence-splitter", choices=["auto","spacy","regex"], default="auto")

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "parse":
        # (tu lógica existente) parsea y guarda IR en outputs_ir/
        # ...
        return

    if args.cmd == "chunk":
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
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
            with open(ir_path, "r", encoding="utf-8") as f:
                ir_json = json.load(f)
            doc_ir = DocumentIR(**ir_json)

            dc = chunker.chunk_document(doc_ir)

            out = Path(args.outdir) / f"{doc_ir.doc_id}_chunks.json"
            with open(out, "w", encoding="utf-8") as f:
                f.write(dc.model_dump_json(indent=2, ensure_ascii=False))
            print(f"[OK] {ir_path} → {out}  (#chunks={len(dc.chunks)})")
        return

    if args.cmd == "pipeline":
        # Ejecuta parse y luego chunk sobre la IR recién generada
        Path(args.ir_outdir).mkdir(parents=True, exist_ok=True)
        Path(args.chunks_outdir).mkdir(parents=True, exist_ok=True)

        # 1) Parse (reuse tu código existente)
        from parser import Parser as DocParser  # ajusta ruta real si es 'parser.parsers'
        p = DocParser()
        ir_paths = []
        for path in args.inputs:
            doc_ir = p.parse(path)
            ir_out = Path(args.ir_outdir) / f"{doc_ir.doc_id}.json"
            with open(ir_out, "w", encoding="utf-8") as f:
                f.write(doc_ir.model_dump_json(indent=2, ensure_ascii=False))
            ir_paths.append(ir_out)
            print(f"[PARSE OK] {path} → {ir_out} (pages={len(doc_ir.pages)})")

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
            with open(irp, "r", encoding="utf-8") as f:
                ir_json = json.load(f)
            doc_ir = DocumentIR(**ir_json)

            dc = chunker.chunk_document(doc_ir)
            out = Path(args.chunks_outdir) / f"{doc_ir.doc_id}_chunks.json"
            with open(out, "w", encoding="utf-8") as f:
                f.write(dc.model_dump_json(indent=2, ensure_ascii=False))
            print(f"[CHUNK OK] {irp} → {out}  (#chunks={len(dc.chunks)})")

if __name__ == "__main__":
    main()
