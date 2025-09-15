# 📚 Proyecto T2G — Knowledge Graph a partir de Documentos

**T2G** es una *pipeline modular* para convertir documentos heterogéneos (PDF, DOCX, imágenes) en una **Representación Intermedia (IR) homogénea**, segmentarlos en **chunks** semánticos y luego en **oraciones filtradas**, listas para RAG/IE/grafos.

* **Entrada:** PDF / DOCX / IMG
* **Salidas (hoy):** **IR (JSON)** → **Chunks (JSON)** → **Sentences (JSON)**
* **Diseño:** subsistemas desacoplados, contratos claros, ejecución CLI/YAML

---

## ✨ Objetivos

* Convertir documentos heterogéneos en una **IR homogénea JSON** con bloques y tablas.
* Desarrollar, probar y orquestar los **subsistemas**: Parser, Chunker, Sentence/Filter, (NER/RE), Normalización, Publicación, Retriever, Evaluación.
* Sentar base para **grafos de conocimiento**, **QA empresarial** y **compliance**.
* Mantener una arquitectura **escalable y modular**, con **contratos Pydantic** y CLIs consistentes.

---

## 🧩 Subsistemas (vivos y planeados)

| Nº | Subsistema          | Rol                                                      | I/O                                     | Estado |
| -: | ------------------- | -------------------------------------------------------- | --------------------------------------- | ------ |
|  1 | **Parser**          | Unificar formatos a **IR JSON/MD** con layout y tablas   | Doc → **IR**                            | ✅      |
|  2 | **HybridChunker**   | Chunks **cohesivos** con tamaños estables y solapamiento | IR → **Chunks**                         | ✅      |
|  3 | **Sentence/Filter** | Dividir en **oraciones** y filtrar ruido antes de IE     | Chunks → **Sentences**                  | ✅      |
|  4 | Triples (dep.)      | (S,R,O) ligeros (dependency-based)                       | Sentences → Triples                     | 🕒     |
|  5 | Extracción (NER/RE) | Entidades y relaciones                                   | Sentences → Menciones                   | 🕒     |
|  6 | Normalización       | Fechas, montos, IDs, orgs                                | Menciones → Entidades                   | 🕒     |
|  7 | Publicación         | Índices / grafo 1-hop                                    | Chunks/Ent/Triples → ES/Qdrant/PG/Grafo | 🕒     |
|  8 | Retriever (cascada) | Recall → Precisión                                       | Query → Contexto                        | 🕒     |
|  9 | Evaluación & HITL   | Calidad / drift / lazo humano                            | Respuestas → Scores                     | 🕒     |

---

## 📂 Estructura del proyecto

```
project_T2G/
├── docs/                      # Documentos de prueba (PDF, DOCX, PNG, JPG)
├── parser/                    # Subsistema 1: Parser
│   ├── parsers.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── chunker/                   # Subsistema 2: HybridChunker
│   ├── hybrid_chunker.py
│   ├── metrics.py
│   └── __init__.py
├── sentence_filter/           # Subsistema 3: Sentence/Filter
│   ├── sentence_filter.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── pipelines/
│   └── pipeline.yaml          # Pipeline declarativo (YAML)
├── outputs_ir/                # Salidas IR (JSON)
├── outputs_chunks/            # Salidas Chunks (JSON)
├── outputs_sentences/         # Salidas Sentences (JSON)
├── t2g_cli.py                 # CLI unificado (parse, chunk, sentences, pipeline-yaml)
├── requirements.txt
└── README.md
```

> **Nota:** asegúrate de que `docs/` exista; ahí van tus archivos de prueba. Ejemplos típicos:
>
> ```
> docs/
> ├── Resume Martin Jurado_CDAO_24.pdf
> ├── tabla.png
> └── ejemplo.docx
> ```

---

## 🧠 Qué hace cada etapa

### Parser (Doc → IR)

Convierte PDF/DOCX/IMG a una **IR homogénea** con bloques de texto y tablas:

* **Detección de tipo** por MIME/extensión y envío a parser especializado.
* **PDF (pdfplumber):**
  texto por líneas, tablas básicas (`vertical/horizontal_strategy=lines`) y **fallback OCR** (Tesseract) por página si no hay texto/tabla.
* **DOCX (python-docx):**
  párrafos, *headings* por estilo, tablas por celda; devuelve una **página lógica**.
* **IMG (Pillow + Tesseract):**
  OCR directo con normalización.
* **Normalización:** `normalize_whitespace`, `dehyphenate`.
* **Metadatos:** `size_bytes`, `page_count`, `mime`, `source_path`, `created_at`.

**Contrato (IR):**
`DocumentIR.pages[*].blocks` puede contener **dicts** (rápidos) o **modelos Pydantic** (`TextBlock`, `TableBlock`, `FigureBlock`). Campos de layout/OCR/provenance están listos para enriquecerse luego.

---

### HybridChunker (IR → Chunks)

Genera **chunks semánticos** ≤ `max_chars`, respetando límites naturales y añadiendo **solapamiento**:

1. **Aplanado** IR a unidades `{kind, text, page_idx, block_idx}`:

   * `heading` → prefijos `#` para conservar jerarquía (opción para **pegar** con siguiente bloque).
   * `paragraph` → texto limpio (normalizado desde Parser).
   * `table` → **serialización CSV-like** por filas a texto plano.
2. **Empaquetado codicioso**:

   * acumula hasta `target_chars`; si cabe, empuja 1 unidad más sin exceder `max_chars`.
   * si excede, **corta por oración** con *spaCy* (si disponible) o **regex**.
3. **Solapamiento** (`overlap_chars`) para continuidad.
4. **Clasificación** por contenido: `table`, `text`, `mixed`.

**Flags clave:** `target_chars`, `max_chars`, `min_chars`, `overlap`, `sentence_splitter` (`auto|spacy|regex`), `table_policy` (`isolate|merge`).

---

### Sentence/Filter (Chunks → Sentences)

Divide **chunks** en **oraciones** y filtra ruido:

1. **Normaliza** (espacios, guiones partidos, bullets).
2. **Divide en oraciones** con *spaCy* (si está) o **regex** (robusta).
3. **Filtra**:

   * `min_chars` (descarta oraciones muy cortas),
   * `drop_stopword_only` (solo stopwords),
   * `drop_numeric_only` (sin letras),
   * `dedupe` (`none|exact|fuzzy`) con umbral `fuzzy_threshold`.
4. **Devuelve trazabilidad** (`chunk_id`, `page_span`, offsets en chunk).

---

## ⚙️ Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# OCR (si planeas usar fallback OCR en PDF y/o IMG)
# macOS:  brew install tesseract tesseract-lang
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa

# (Opcional) spaCy para mejores cortes por oración
pip install spacy
python -m spacy download es_core_news_sm
# (inglés opcional)
python -m spacy download en_core_web_sm
```

---

## 🚀 Uso rápido (CLI)

### Parse → IR

```bash
python t2g_cli.py parse docs/Resume\ Martin\ Jurado_CDAO_24.pdf --outdir outputs_ir
```

### IR → Chunks

```bash
python t2g_cli.py chunk outputs_ir/DOC-XXXX.json --outdir outputs_chunks \
  --sentence-splitter spacy --target-chars 1400 --max-chars 2048 --overlap 120
```

### Chunks → Sentences

```bash
python t2g_cli.py sentences outputs_chunks/DOC-XXXX_chunks.json \
  --outdir outputs_sentences \
  --sentence-splitter auto --min-chars 25 --dedupe fuzzy --fuzzy-threshold 0.92
```

---

## 🧾 Pipeline declarativo (YAML)

Archivo por defecto: `pipelines/pipeline.yaml`

```yaml
pipeline:
  dry_run: false
  continue_on_error: true

stages:
  - name: parse
    args:
      clean_outdir: true
      inputs_glob:
        - "docs/*.pdf"
        - "docs/*.png"
        - "docs/*.jpg"
        - "docs/*.docx"
      outdir: "outputs_ir"

  - name: chunk
    args:
      ir_glob: "outputs_ir/*.json"
      outdir: "outputs_chunks"
      target_chars: 1400
      max_chars: 2048
      min_chars: 400
      overlap: 120
      table_policy: "isolate"    # isolate | merge
      sentence_splitter: "auto"  # auto | spacy | regex

  - name: sentences
    args:
      chunks_glob: "outputs_chunks/*_chunks.json"
      outdir: "outputs_sentences"
      sentence_splitter: "auto"  # o "spacy"
      min_chars: 25
      dedupe: "fuzzy"            # none | exact | fuzzy
      fuzzy_threshold: 0.92
      # toggles
      no_normalize_whitespace: false
      no_dehyphenate: false
      no_strip_bullets: false
      keep_stopword_only: false
      keep_numeric_only: false
```

Ejecutar:

```bash
python t2g_cli.py pipeline-yaml
# o explícito:
python t2g_cli.py pipeline-yaml --file pipelines/pipeline.yaml
```

---

## 📊 Métricas por subsistema

### Parser

* **`percent_docs_ok`**: % de documentos parseados sin error (éxito de lote).
* **`layout_loss`**: proporción `unknown/total` (proxy de pérdida de estructura).
* **`table_consistency`**: tablas encontradas vs. valor *golden* (si existe).

**Umbrales sugeridos:** `%docs_ok ≥ 95%`, `layout_loss ≤ 0.15`, `table_consistency.ratio ≥ 0.9`.

```python
import json, glob
from parser.schemas import DocumentIR
from parser.metrics import percent_docs_ok, layout_loss, table_consistency

irs = []
for path in sorted(glob.glob("outputs_ir/*.json")):
    try:
        irs.append(DocumentIR(**json.load(open(path, "r", encoding="utf-8"))))
    except Exception:
        continue

ok_pct = percent_docs_ok(irs)
losses = {ir.doc_id: layout_loss(ir) for ir in irs}
cons   = {ir.doc_id: table_consistency(ir) for ir in irs}

print({"percent_docs_ok": round(ok_pct, 2)})
print({"layout_loss_avg": round(sum(losses.values())/max(1,len(losses)), 4)})
print({"table_consistency_sample": list(cons.items())[:2]})
```

---

### HybridChunker

* **`chunk_length_stats`**: distribución de longitudes.
* **`percent_within_threshold(min,max)`**: proporción de chunks dentro del rango (p. ej. 400–2048).
* **`table_mix_ratio`**: proporción `text/mixed/table`.

**Umbrales sugeridos:** `within(400–2048) ≥ 0.95`.

```python
import json, glob
from parser.schemas import DocumentChunks
from chunker.metrics import chunk_length_stats, percent_within_threshold, table_mix_ratio

for path in sorted(glob.glob("outputs_chunks/*_chunks.json"))[:3]:
    dc = DocumentChunks(**json.load(open(path, "r", encoding="utf-8")))
    print(dc.doc_id, {
        "len_stats": chunk_length_stats(dc),
        "within_400_2048": percent_within_threshold(dc, 400, 2048),
        "mix": table_mix_ratio(dc)
    })
```

---

### Sentence/Filter

* **`meta.counters`** en salida: `total_split`, `kept`, `dropped_short`, `dropped_stopword`, `dropped_numeric`, `dropped_dupe`, `keep_rate`.
* **`sentence_length_stats`**: distribución de longitudes de oraciones.
* **`unique_ratio`**: proporción de oraciones únicas (detecta duplicados).

**Umbrales sugeridos:** `keep_rate` saludable ≈ 0.6–0.9 según dominio; `unique_ratio ≥ 0.85`.

```python
import json, glob
from sentence_filter.schemas import DocumentSentences
from sentence_filter.metrics import sentence_length_stats, unique_ratio

for path in sorted(glob.glob("outputs_sentences/*_sentences.json"))[:3]:
    ds = DocumentSentences(**json.load(open(path, "r", encoding="utf-8")))
    counters = ds.meta.get("counters", {})
    print(ds.doc_id, {
        "keep_rate": round(counters.get("keep_rate", 0), 3),
        "kept": counters.get("kept"),
        "len_stats": sentence_length_stats(ds),
        "unique_ratio": round(unique_ratio(ds), 3)
    })
```

---

## 🛠️ Troubleshooting

* **JSONs vacíos/corruptos:** se escribe de forma **atómica** y el runner **salta** JSON inválidos. Puedes limpiar y re-ejecutar:

  ```bash
  rm -rf outputs_ir/* outputs_chunks/* outputs_sentences/*
  python t2g_cli.py pipeline-yaml
  ```
* **spaCy no carga:** instala `es_core_news_sm` (y opcionalmente `en_core_web_sm`).
* **OCR fallback no funciona:** instala Tesseract y configura `tesseract_cmd` (Windows) o asegúrate de que esté en `PATH`.
* **Imports fallan:** ejecuta desde la **raíz** del repo (`python t2g_cli.py …`).

---

## 🧪 Roadmap inmediato

* Triples (dependency-based) y NER/RE.
* Normalización (fechas, montos, IDs) con validadores robustos.
* Publicación a ES/Qdrant/Postgres/Grafo.
* Métricas de evaluación y lazo humano (RAGAS + HITL).
* Tests (`pytest`) y *golden sets* de regresión.

---

## 📚 Referencias

* **OCR:** PaddleOCR, Tesseract
* **PDF:** pdfplumber
* **DOCX:** python-docx
* **NLP:** spaCy
* **Papers:** DocRED (ACL 2019), K-Adapter (ACL 2020), AutoNER (ACL 2018), KG-BERT (AAAI 2020)

---

## ⚖️ Licencia

MIT (o la que definas).
