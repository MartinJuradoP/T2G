#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_conll_pdfs.py — Genera PDFs de consumo a partir de CoNLL03 (solo texto)

Propósito
---------
- Descargar un subconjunto de CoNLL03, detokenizarlo y empaquetarlo en PDFs.
- Ubicar los PDFs en `docs/conll03_test_1k/` para que el pipeline T2G los procese.
- Mantener un manifest mínimo con seed/indices para reproducibilidad.

Lo que NO hace
--------------
- No altera el pipeline, ni añade etapas.
- No genera ground truth ni evalúa; solo produce PDFs de entrada.

Requisitos
----------
pip install datasets reportlab

Uso
---
python benchmarks/conll03_pdfgen/scripts/make_conll_pdfs.py
"""

from __future__ import annotations
import os
import json
import random
from datetime import datetime
from typing import List

from datasets import load_dataset
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet

# Configuración
SPLIT = "test"
N_SAMPLES = 1000          # muestras a usar
GROUP_SIZE = 5            # muestras por PDF (controla nº de PDFs)
SEED = 42

PDF_DIR = "docs/conll03_test_1k"
META_DIR = "benchmarks/conll03_pdfgen/meta"


def ensure_dirs() -> None:
    for d in [PDF_DIR, META_DIR]:
        os.makedirs(d, exist_ok=True)


def detokenize(tokens: List[str]) -> str:
    text = " ".join(tokens)
    fixes = [
        (" ,", ","), (" .", "."), (" :", ":"), (" ;", ";"),
        (" !", "!"), (" ?", "?"), (" )", ")"), ("( ", "("),
        (" 's", "'s"), (" n't", "n't"),
    ]
    for a, b in fixes:
        text = text.replace(a, b)
    return text


def render_pdf(pdf_path: str, paragraphs: List[str]) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=LETTER)
    story = []
    for p in paragraphs:
        story.append(Paragraph(p.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 10))
    doc.build(story)


def main() -> None:
    ensure_dirs()
    random.seed(SEED)

    ds = load_dataset("conll2003", split=SPLIT)

    all_idx = list(range(len(ds)))
    random.shuffle(all_idx)
    chosen = all_idx[:N_SAMPLES]

    # Guardamos manifest de reproducibilidad
    manifest = {
        "dataset": "conll2003",
        "split": SPLIT,
        "n_samples": N_SAMPLES,
        "group_size": GROUP_SIZE,
        "seed": SEED,
        "pdf_dir": os.path.abspath(PDF_DIR),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "indices": chosen,
    }
    os.makedirs(META_DIR, exist_ok=True)
    with open(os.path.join(META_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    n_docs = 0
    for batch_start in range(0, len(chosen), GROUP_SIZE):
        batch = chosen[batch_start:batch_start + GROUP_SIZE]
        if not batch:
            continue

        doc_id = f"CONLL03_{SPLIT}_{n_docs:04d}"
        pdf_filename = f"{doc_id}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        paragraphs = []
        for j, idx in enumerate(batch):
            row = ds[idx]
            tokens = row["tokens"]
            text = detokenize(tokens)
            para = f"<b>Sample {j+1}</b><br/>{text}"
            paragraphs.append(para)

        render_pdf(pdf_path, paragraphs)
        n_docs += 1

    print(f"✅ PDFs generados: {n_docs} en {PDF_DIR}")
    print(f"ℹ️  Manifest: {os.path.join(META_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
