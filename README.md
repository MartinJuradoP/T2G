# 📚 Proyecto T2G — Knowledge Graph a partir de Documentos

**T2G** es una *pipeline modular y extensible* que convierte documentos heterogéneos (PDF, DOCX, imágenes) en una **Representación Intermedia (IR) homogénea**, los enriquece con **contexto semántico global y local**, y prepara la base para construir **grafos de conocimiento** y sistemas de **búsqueda avanzada (RAG, QA, compliance, etc.)**.

* **Entrada (hoy):** PDF / DOCX / PNG / JPG
* **Salidas (hoy):**

  * `DocumentIR (JSON)`
  * `DocumentIR+Topics (JSON)`
* **Salidas futuras:** Chunks, Mentions, Entities, Triples, Normalización, Grafo.
* **Diseño:** subsistemas **desacoplados**, contratos **Pydantic**, orquestación vía **CLI + YAML**.

---

## ✨ Objetivos

* Unificar la ingesta de documentos en una **IR JSON común** independientemente del formato.
* Enriquecer documentos con **contexto semántico a nivel documento y chunk** usando **embeddings + BERTopic**.
* Mantener una arquitectura **resiliente, escalable y modular**: cada subsistema puede ejecutarse de forma independiente.
* Preparar la base para **grafos de conocimiento**, **QA empresarial**, **compliance regulatorio** y **sistemas RAG**.

---

## 🧩 Subsistemas

| Nº | Subsistema                       | Rol principal                                                               | Entrada             | Salida                   | Estado |
| -: | -------------------------------- | --------------------------------------------------------------------------- | ------------------- | ------------------------ | ------ |
|  1 | **Parser**                       | Genera **IR JSON** homogénea con metadatos y layout                         | Doc (PDF/DOCX/IMG)  | `DocumentIR` JSON        | ✅      |
|  2 | **BERTopic Contextizer (doc)**   | Asigna **tópicos y keywords globales** a nivel documento                    | `DocumentIR`        | `DocumentIR+Topics` JSON | ✅      |
|  3 | **HybridChunker**                | Segmenta documento en **chunks semánticos estables (≤2048 tokens)**         | `DocumentIR+Topics` | `DocumentChunks` JSON    | 🔜     |
|  4 | **BERTopic Contextizer (chunk)** | Asigna tópicos locales a cada chunk (subtemas); enlaza con tópicos globales | `DocumentChunks`    | `Chunks+Topics` JSON     | 🔜     |
|  5 | **Adaptive Schema Selector**     | Define dinámicamente entidades relevantes según contexto                    | `Chunks+Topics`     | `SchemaSelection` JSON   | 🔜     |
|  6 | **Mentions (NER/RE)**            | Detecta menciones condicionadas por tópicos                                 | `Chunks+Topics`     | `Mentions` JSON          | 🔜     |
|  7 | **Clustering de Menciones**      | Agrupa spans en clusters semánticos                                         | `Mentions` JSON     | `Clusters` JSON          | 🔜     |
|  8 | **Weak Supervision / Label**     | Etiqueta clusters de alta confianza (Snorkel-style)                         | `Clusters`          | `LabeledClusters` JSON   | 🔜     |
|  9 | **LLM Intervention**             | Clasifica clusters ambiguos con **few-shot prompting o prototipos**         | `Clusters`          | `RefinedLabels` JSON     | 🔜     |
| 10 | **Normalización (híbrida)**      | Canonicaliza entidades y estandariza representaciones                       | `Mentions/Clusters` | `Entities` JSON          | 🔜     |
| 11 | **Graph Export**                 | Publica entidades y relaciones en grafos (Neo4j, GraphDB, RDF/SHACL)        | `Entities+Triples`  | Grafo / DB               | 🔜     |

---

## 📂 Estructura del proyecto

```bash
project_T2G/
├── docs/                   # Documentos de prueba
├── parser/                 # Subsistema Parser
│   ├── parsers.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── contextizer/            # Subsistema Contextizer
│   ├── contextizer.py
│   ├── metrics.py
│   ├── models.py
│   ├── utils.py
│   ├── schemas.py
│   └── __init__.py
├── pipelines/
│   └── pipeline.yaml
├── outputs_ir/             # Salidas: DocumentIR
├── outputs_doc_topics/     # Salidas: IR+Topics
├── t2g_cli.py              # CLI unificado
├── requirements.txt
└── README.md
```

---

## 🧠 Etapas explicadas

---

### 1) Parser (Doc → IR) ✅

**Entrada:** PDF / DOCX / PNG / JPG
**Salida:** `DocumentIR` (`outputs_ir/{DOC}_*.json`)

**Qué hace:**

* Detecta formato y selecciona parser:

  * **PDF:** texto + tablas (pdfplumber); OCR fallback si escaneado.
  * **DOCX:** párrafos, headings, tablas (python-docx).
  * **IMG:** OCR (pytesseract).
* Normaliza espacios, guiones cortados, saltos de línea.
* Añade metadatos: `sha256`, `mime`, `page_count`, `size_bytes`.

**Ejemplo de salida:**

```json
{
  "doc_id": "DOC-12345",
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {"type": "paragraph", "text": "Los leones viven en África..."},
        {"type": "table", "cells": [{"row":0,"col":0,"text":"Dato"}]}
      ]
    }
  ],
  "meta": {
    "mime":"application/pdf",
    "page_count":1,
    "sha256":"…",
    "source_path":"docs/leones.pdf"
  }
}
```

---

### 2) BERTopic Contextizer (doc-level) ✅

**Entrada:** `DocumentIR` (`outputs_ir/*.json`)
**Salida:** `DocumentIR+Topics` (`outputs_doc_topics/*.json`)

**Qué hace:**

* Calcula embeddings globales con **SentenceTransformers (`all-MiniLM-L6-v2`)**.
* Descubre tópicos con **BERTopic**.
* Si hay pocos chunks:

  * 0–1 → topic único (singleton).
  * 2 → fallback basado en frecuencia de términos.
  * 3+ → BERTopic normal.
* Enriquecimiento agregado a `meta.topics_doc`:

  * `n_topics`, `keywords_global`, `exemplar`, `outlier_ratio`.

**Ejemplo de salida:**

```json
"topics_doc": {
  "reason": "doc-level",
  "n_topics": 2,
  "keywords_global": ["diabetes","insulina","síntomas"],
  "topics": [
    {
      "topic_id": 0,
      "count": 5,
      "exemplar": "Síntomas Clínicos más frecuentes...",
      "keywords": ["síntomas","micción","aumento"]
    },
    {
      "topic_id": 1,
      "count": 4,
      "exemplar": "Diabetes Tipo 2: Un enfoque clínico",
      "keywords": ["diabetes","tratamiento","complicaciones"]
    }
  ]
}
```

---

### 3) HybridChunker 🔜

* Divide documento en **chunks semánticos ≤2048 tokens**.
* Hereda `topics_doc` para mantener coherencia.

### 4) BERTopic Contextizer (chunk-level) 🔜

* Asigna tópicos específicos a cada chunk.
* Permite detectar **subtemas** y transferir contexto.

### 5) Adaptive Schema Selector 🔜

* Define dinámicamente qué entidades extraer según tópicos.

### 6) Mentions (NER/RE) 🔜

* Detecta menciones de entidades **condicionadas por tópicos**.

### 7–11) Clustering → Label → LLM → Normalización → Grafo 🔜

* Agrupan spans, etiquetan con weak supervision, refinan con LLM y normalizan entidades.
* Export final a **grafo de conocimiento**.

---

## 📂 Pipeline declarativo (YAML)

Archivo: `pipelines/pipeline.yaml`

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

  - name: contextize-doc
    args:
      clean_outdir: true
      ir_glob: "outputs_ir/*.json"
      embedding_model: "all-MiniLM-L6-v2"
      nr_topics: null
      seed: 42
      outdir: "outputs_doc_topics"
```

Ejecutar:

```bash
python t2g_cli.py pipeline-yaml
```

---

## 📊 Métricas por subsistema

### Parser ✅

* `percent_docs_ok`: éxito de parseo por lote.
* `layout_loss`: pérdida de estructura.
* `table_consistency`: tablas detectadas vs esperadas.

### Contextizer (doc-level) ✅

* `coverage`: proporción de chunks asignados a algún tópico.
* `outlier_rate`: ratio de outliers vs asignaciones válidas.
* `topic_size_stats`: distribución (min, mediana, p95).
* `keywords_diversity`: diversidad de keywords únicas.

### Próximos subsistemas 🔜

* **HybridChunker**: `chunk_length_stats`.
* **Mentions**: `precision/recall vs golden`.
* **Normalization**: % de entidades deduplicadas.

---

## ⚙️ Instalación

### Requisitos principales

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### OCR

```bash
brew install tesseract tesseract-lang   # macOS
sudo apt install tesseract-ocr-spa      # Ubuntu/Debian
```

### NLP / embeddings

```bash
pip install spacy sentence-transformers bertopic
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### Opcionales

```bash
pip install torch joblib matplotlib wordcloud
```

---

## 🚀 Uso rápido (CLI)

```bash
# Parser → IR
python t2g_cli.py parse docs/ejemplo.pdf --outdir outputs_ir

# Contextizer (doc) → IR+Topics
python t2g_cli.py contextize-doc outputs_ir/*.json --outdir outputs_doc_topics

# Pipeline completo
python t2g_cli.py pipeline-yaml
```

---

## 🛠️ Troubleshooting

* **JSONs vacíos:** borra outputs y reejecuta.
* **OCR falla:** instala Tesseract y revisa `PATH`.
* **spaCy no carga:** descarga modelos `es_core_news_sm`, `en_core_web_sm`.
* **Errores de import:** ejecuta siempre desde raíz del proyecto.

---

## 🧪 Roadmap inmediato

* Añadir **HybridChunker** con herencia de `topics_doc`.
* Implementar **chunk-level contextizer**.
* Integrar **Adaptive Schema Selector**.
* Completar **Mentions + Clustering + Normalización**.
* Exportar entidades y relaciones a **Neo4j / RDF**.

