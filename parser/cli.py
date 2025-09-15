"""
cli.py — Ejecución por lote del Parser y reporte de métricas

Uso:
    python -m kg_pipeline.parser.cli docs/a.pdf docs/b.docx --outdir outputs_ir --golden-tables 2

Qué hace:
- Parsea cada archivo y guarda la IR en JSON (un archivo por doc).
- Imprime métricas por documento (layout_loss, tablas encontradas/ratio).
- Al final, imprime %docs_ok del lote.

Tips:
- Integra este CLI a un pipeline de CI para detectar regresiones.
- Agrega logging a archivo si lo corres en servidor (por defecto, STDOUT).
"""

import argparse
import json
from pathlib import Path
import logging

from parsers import Parser
from metrics import percent_docs_ok, layout_loss, table_consistency

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def main():
    ap = argparse.ArgumentParser("kg-parser")
    ap.add_argument("inputs", nargs="+", help="Rutas de archivos a parsear (PDF/DOCX/IMG)")
    ap.add_argument("--outdir", default="outputs_ir", help="Directorio donde guardar IR JSON")
    ap.add_argument("--golden-tables", type=int, default=None, help="Tablas esperadas por doc (opcional)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    parser = Parser()
    docs = []
    for in_path in args.inputs:
        try:
            doc = parser.parse(in_path)
            docs.append(doc)

            out_path = outdir / f"{doc.doc_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

            logger.info("[OK] %s → %s (pages=%d)", in_path, out_path.name, len(doc.pages))

            # Métricas por doc
            ll = layout_loss(doc)
            tc = table_consistency(doc, args.golden_tables)
            logger.info("   layout_loss=%.3f tables_found=%s ratio=%.2f", ll, tc["found"], tc["ratio"])

        except Exception as e:
            docs.append(None)
            logger.error("[ERR] %s: %s", in_path, e)

    # Métrica de lote
    ok_pct = percent_docs_ok(docs)
    logger.info("== LOTE ==  %%docs_ok = %.2f%%  (total=%d)", ok_pct, len(docs))

if __name__ == "__main__":
    main()
