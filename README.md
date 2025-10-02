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
|  3 | **HybridChunker**                | Segmenta documento en **chunks semánticos estables (≤2048 tokens)**         | `DocumentIR+Topics` | `DocumentChunks` JSON    | ✅     |
|  4 | **BERTopic Contextizer (chunk)** | Asigna tópicos locales a cada chunk (subtemas); enlaza con tópicos globales | `DocumentChunks`    | `Chunks+Topics` JSON     | ✅     |
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

**Qué hace:**

* **1. Extracción de texto**:
  Toma todos los bloques textuales del documento (párrafos, headings, tablas OCR).
  Preprocesa eliminando espacios raros, stopwords multilingües y normalizando tokens.

* **2. Embeddings globales**:
  Cada bloque se convierte en un vector usando **SentenceTransformers** (`all-MiniLM-L6-v2`).
  Estos embeddings capturan similitud semántica más allá de palabras exactas.

* **3. Clustering de tópicos con BERTopic**:
  Los embeddings se reducen con **UMAP** y se agrupan con **HDBSCAN**.

  * **UMAP** → baja dimensión para preservar estructura semántica.
  * **HDBSCAN** → encuentra clusters de tamaño variable sin fijar `k`.
  * **BERTopic** → asigna palabras clave representativas a cada cluster.

* **4. Heurística adaptativa (corpus pequeño)**:
  Como documentos pueden ser muy cortos, aplicamos **fallbacks** para evitar errores o ruido:

  * **0–1 bloques** → se asigna un único topic (`singleton`), con keywords derivadas por frecuencia.
  * **2 bloques** → no hay suficiente masa para clustering; se aplica **TF-based fallback** (frecuencia de términos relevantes).
  * **≥3 bloques** → se ejecuta BERTopic normal con embeddings + clustering.
    Esto asegura que siempre exista al menos un `topic` incluso en documentos mínimos.

* **5. Post-procesamiento y limpieza**:
  Para cada cluster (topic) se selecciona:

  * `exemplar`: bloque de texto más representativo.
  * `keywords`: lista de palabras clave filtradas (removiendo stopwords, duplicados, ruido corto).
  * `count`: número de bloques asignados.
    Además se calculan métricas globales como `outlier_ratio` (proporción de bloques no asignados a ningún cluster válido).

**Enriquecimiento agregado a `meta.topics_doc`:**

* `n_topics`: número total de tópicos encontrados o inferidos.
* `keywords_global`: lista unificada de keywords más frecuentes en todo el documento.
* `topics`: listado detallado de cada topic con `id`, `count`, `exemplar`, `keywords`.
* `reason`: explica el modo usado (`doc-level`, `doc-fallback-small`, `doc-fallback-error`).
* `outlier_ratio`: porcentaje de bloques descartados como outliers (cuando aplica).

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

#### 3) HybridChunker ✅

**Entrada:** `DocumentIR+Topics` (`outputs_doc_topics/*.json`)
**Salida:** `DocumentChunks` (`outputs_chunks/*.json`)

**Qué hace (paso a paso):**

1. **Extracción de bloques base**

   * Toma todos los bloques textuales del `DocumentIR`.
   * Cada bloque conserva trazabilidad (`page_number`, `block_indices`) para poder mapear chunks a posiciones exactas en el documento.

2. **Segmentación semántica híbrida**
   Se combinan varias estrategias para dividir el documento en **chunks coherentes de ≤2048 tokens** (óptimo para LLMs):

   * **Reglas de headings**: patrones típicos (`Introducción`, `Métodos`, `Conclusiones`, etc.) detectados vía regex.
   * **spaCy sentence boundaries**: segmenta párrafos largos en oraciones bien definidas.
   * **Embeddings (SentenceTransformers)**: evalúa cohesión semántica entre bloques y decide si agrupar o dividir.
     → El resultado son **chunks “estables”**: suficientemente largos para contexto, pero sin sobrepasar el límite de tokens.

3. **Herencia de contexto (`topic_hints`)**

   * Cada chunk hereda información del doc-level contextizer (`topic_ids`, `keywords_global`, `topic_affinity`).
   * Esto asegura **consistencia semántica vertical**: lo que el documento sabe a nivel global está presente también en cada chunk.

4. **Cálculo de métricas locales**
   Para evaluar la calidad de los chunks, se añaden scores:

   * `cohesion_vs_doc`: similitud entre el chunk y el embedding global del documento (cercanía semántica).
   * `max_redundancy`: medida de solapamiento con otros chunks (evita duplicados o repetición excesiva).

5. **Serialización robusta**

   * Cada chunk se guarda con `chunk_id` único, trazabilidad al documento (`doc_id`), orden secuencial y `source_spans`.
   * Se asegura compatibilidad JSON (ej. timestamps en ISO 8601).

* **Ejemplo de salida (simplificado):**

```json
{
  "chunk_id": "DOC-123_0001",
  "text": "Complicaciones Asociadas...",
  "topic_hints": {
    "inherited_keywords": ["diabetes","tratamiento"],
    "inherited_topic_ids": [0,1]
  },
  "scores": {
    "cohesion_vs_doc": 0.82,
    "max_redundancy": 0.59
  }
}
```

---

#### 4) BERTopic Contextizer (chunk-level) ✅

* **Entrada:** `DocumentChunks`

* **Salida:** `Chunks+Topics`

* **Qué hace:**

  * Recalcula embeddings para cada chunk.
  * Intenta descubrir **subtemas locales** con BERTopic.
  * **Lógica de fallback**:

    * `n_samples = 0` → no hay texto, se omite.
    * `n_samples < 5` → usa fallback por frecuencia de términos.
    * `n_samples ≥ 5` → corre BERTopic normal.
  * Cada chunk queda enriquecido con:

    * `topic` (id, keywords, prob).
    * `topic_hints` (heredados de doc-level).

* **Ejemplo de salida (fallback):**

```json
"topics_chunks": {
  "reason": "chunk-fallback-small",
  "n_samples": 3,
  "n_topics": 1,
  "keywords_global": ["diabetes","tratamiento","insulina"],
  "topics": [
    {
      "topic_id": 0,
      "count": 3,
      "exemplar": "Diabetes Tipo 2: Un enfoque clínico...",
      "keywords": ["diabetes","complicaciones","tratamiento"]
    }
  ]
}
```

---

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

  - name: chunk
    args:
      clean_outdir: true
      ir_glob: "outputs_doc_topics/*.json"
      embedding_model: "all-MiniLM-L6-v2"
      spacy_model: "es_core_news_sm"
      max_tokens: 2048
      min_chars: 280
      outdir: "outputs_chunks"

  - name: contextize-chunks
    args:
      clean_outdir: false
      ir_glob: "outputs_chunks/*.json"
      embedding_model: "all-MiniLM-L6-v2"
      nr_topics: null
      seed: 42
      outdir: "outputs_chunks"
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

### HybridChunker**

  * `chunk_length_stats`: distribución de tamaño en caracteres/tokens.
  * `cohesion_vs_doc`: similitud coseno chunk ↔ doc.
  * `max_redundancy`: similitud máx entre chunks (para evitar duplicados).

### Contextizer (chunk-level)**

  * `coverage`: % de chunks con topic asignado.
  * `fallback_rate`: % de documentos donde se usó fallback vs BERTopic.
  * `topic_size_stats`: tamaño medio de clusters locales.

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

