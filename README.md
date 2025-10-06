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
├── docs/                      # Documentos de prueba
├── parser/                    # Subsistema Parser
│   ├── parsers.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── contextizer/               # Subsistema Contextizer
│   ├── contextizer.py
│   ├── metrics.py
│   ├── models.py
│   ├── utils.py
│   ├── schemas.py
│   └── __init__.py
├── schema_selector/           # Adaptive Schema Selector 🔥
│   ├── registry.py            # Ontologías y dominios (medical, legal, etc.)
│   ├── schemas.py             # Contratos Pydantic + Config
│   ├── selector.py            # Lógica de scoring y selección
│   ├── utils.py               # Funciones auxiliares (similitud, normalización)
│   └── __init__.py
├── pipelines/
│   └── pipeline.yaml
├── outputs_ir/                # Salidas: DocumentIR
├── outputs_doc_topics/        # Salidas: IR+Topics
├── outputs_chunks/            # Salidas: Chunks
├── outputs_schema/            # Salidas: SchemaSelection
├── t2g_cli.py                 # CLI unificado
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
   * Filtra bloques vacíos o no textuales (imágenes, tablas sin OCR).

2. **Segmentación semántica híbrida**
   Se combinan varias estrategias para dividir el documento en **chunks coherentes de ≤2048 tokens** (óptimo para LLMs):

   * **Reglas de headings:** patrones típicos (`Introducción`, `Métodos`, `Conclusiones`, etc.) detectados vía regex.
     Esto evita cortar secciones temáticas de forma arbitraria.

   * **spaCy sentence boundaries:** segmenta párrafos largos en oraciones completas, preservando la coherencia sintáctica.
     Si spaCy no está disponible, se aplica un fallback mediante puntuación (`.`, `;`, `?`, `!`) o saltos de línea dobles (`\n\n`).

   * **Empaquetado semántico por longitud:** agrupa oraciones hasta alcanzar un límite aproximado de tokens.
     Controla umbrales de tamaño (`min_chars`, `max_chars`, `max_tokens`) para mantener chunks **equilibrados** en densidad y contexto.
     → El resultado son **unidades estables**: suficientemente largas para el contexto, pero sin sobrepasar los límites óptimos de procesamiento.

3. **Herencia de contexto (`topic_hints`)**

   * Cada chunk hereda información del doc-level contextizer (`topic_ids`, `keywords_global`, `topic_affinity`).
   * Esto asegura **consistencia semántica vertical**: lo que el documento conoce a nivel global se transfiere a los fragmentos locales.
   * La afinidad entre el chunk y los tópicos globales se calcula mediante una mezcla semántico-léxica:

     $$\text{topic affinity blend}(c, t) = \alpha \cdot \cos(\vec{c}, \vec{t}) + (1 - \alpha) \cdot J(c, t)$$

     Donde:

     * $\vec{c}$ y $\vec{t}$ son los embeddings del chunk y del topic.
     * $J(c, t)$ es la similitud léxica (*Jaccard*).
     * $\alpha$ controla el peso semántico (por defecto, $\alpha = 0.7$).

     Esta combinación hace al sistema más robusto frente a textos cortos (tweets, cláusulas legales o notas clínicas) donde la semántica sola puede ser insuficiente.

4. **Cálculo de métricas locales**
   
   Para evaluar la calidad y la coherencia de los chunks, se calculan métricas cuantitativas:

   * `cohesion_vs_doc`: mide la similitud coseno entre el embedding del chunk y el embedding promedio del documento.
     Representa **qué tan bien el fragmento conserva el contexto global**.

     $$\text{cohesion vs doc}(c_i) = \cos(\vec{c_i}, \bar{\vec{D}})$$

   * `max_redundancy`: mide la similitud máxima del chunk con cualquier otro chunk dentro del mismo documento.
     Detecta **fragmentos repetitivos o duplicados**.

     $$\text{max redundancy}(c_i) = \max_{j \neq i} \cos(\vec{c_i}, \vec{c_j})$$

   * `redundancy_norm`: versión normalizada de la redundancia, que ajusta el valor según el tamaño relativo del fragmento.

     $$\text{redundancy norm}(c_i) = \text{max redundancy}(c_i) \times \frac{\text{len}(c_i)}{\text{avg len(chunks)}}$$

     Esto penaliza más a los chunks **largos y redundantes**, y reduce el impacto de los fragmentos **cortos pero similares**.
     En textos como contratos, reseñas o tweets, mejora la detección de contenido **repetitivo** versus **informativo**.

   * `novelty`: mide la proporción de información nueva que aporta cada fragmento.
     Es complementaria a la redundancia.

     $$\text{novelty}(c_i) = 1 - \text{max redundancy}(c_i)$$

     Un valor alto de `novelty` indica que el chunk aporta **contexto único o evidencia nueva**.

   * `chunk_health`: métrica compuesta que pondera la cohesión y la novedad penalizando la redundancia.
     Resume la **salud semántica del fragmento**.

     $$\text{chunk health}(c_i) = \text{cohesion vs doc}(c_i) \times (1 - \text{max redundancy}(c_i))$$

     Este score puede usarse en etapas posteriores (por ejemplo, el **Adaptive Schema Selector**) para **ponderar o filtrar chunks** según su calidad semántica.

5. **Serialización robusta**

   * Cada chunk se guarda con un `chunk_id` único, metadatos (`doc_id`, idioma, embeddings opcionales) y trazabilidad (`source_spans`).
   * El formato JSON conserva compatibilidad con las etapas siguientes (`contextize-chunks`, `schema-select`).
   * Si los embeddings o spaCy no están disponibles, aplica **fallbacks automáticos** para mantener la robustez del pipeline.

* **Ejemplo de salida (simplificado):**

```json
{
  "chunk_id": "DOC-123_0001",
  "text": "Complicaciones Asociadas al tratamiento prolongado...",
  "topic_hints": {
    "inherited_keywords": ["diabetes", "tratamiento"],
    "inherited_topic_ids": [0, 1],
    "topic_affinity_blend": {"0": 0.82, "1": 0.71}
  },
  "scores": {
    "cohesion_vs_doc": 0.82,
    "max_redundancy": 0.59,
    "redundancy_norm": 0.73,
    "novelty": 0.41,
    "chunk_health": 0.34
  },
  "meta_local": {
    "embedding_model": "all-MiniLM-L6-v2",
    "lang": "es"
  }
}
```


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

### 5) Adaptive Schema Selector ✅

**Entrada:** `Chunks+Topics` (`outputs_chunks/*.json`)
**Salida:** `SchemaSelection` (`outputs_schema/*.json`)

**Qué hace:**

El **Adaptive Schema Selector** determina dinámicamente qué **dominios de entidades** (ej. médico, legal, financiero, genérico) son relevantes para cada documento y chunk. Esto evita extraer entidades irrelevantes y mejora la **precisión** del grafo.

1. **Registro de dominios (`registry.py`)**
   Cada dominio contiene:

   * Entidades (`EntityTypeDef`) con atributos (ej. `Disease`, `Treatment`).
   * Relaciones (`RelationTypeDef`) entre entidades.
   * Aliases y vocabulario específico (ej. `"enfermedad"`, `"patología"` para `Disease`).

2. **Extracción de señales del documento/chunk**

   * **Keywords**: se buscan overlaps entre tokens y aliases.
   * **Embeddings**: se calcula similitud coseno entre centroides del texto y embeddings predefinidos de etiquetas (`label_vecs`).
   * **Priors**: se aplican pesos de confianza inicial (ej. `generic` siempre se incluye con peso bajo).

3. **Fórmula de scoring (por dominio):**

   Para cada dominio (d):

$$
\text{score}(d) = \alpha \cdot S_{\text{kw}}(d) + \beta \cdot S_{\text{emb}}(d) + \gamma \cdot P(d)
$$

Donde:

* $S_{\text{kw}}$: score normalizado por overlap de keywords.
* $S_{\text{emb}}$: similitud coseno entre embeddings.
* $P(d)$: prior asignado al dominio.
* $\alpha, \beta, \gamma$ → hiperparámetros configurables en `SelectorConfig`.

**Ejemplo default:**

$\alpha = 0.6, \beta = 0.3, \gamma = 0.1$ → más peso a keywords, menos a embeddings y priors.

4. **Selección final**

   * Se ordenan los dominios por score.
   * Se descartan dominios con score < `min_topic_conf`.
   * Se seleccionan los `top-k` dominios (por config).
   * Se marca `ambiguous=True` si la diferencia entre primer y segundo dominio < `ambiguity_threshold`.
   * Siempre se incluye el dominio **genérico** como fallback (`allow_fallback_generic=True`).

---

**Ejemplo de salida (simplificado):**

```json
{
  "doc": {
    "doc_id": "DOC-123",
    "top_domains": ["medical"],
    "ambiguous": false,
    "domain_scores": [
      {"domain":"medical","score":0.18,"evidence":[{"kind":"keyword","detail":{"overlap":2,"kw_score":0.2}},{"kind":"stat","detail":{"alpha":0.6,"beta":0.3}}]},
      {"domain":"legal","score":0.05,"evidence":[{"kind":"keyword","detail":{"overlap":0,"kw_score":0.0}}]}
    ]
  },
  "chunks": [
    {
      "chunk_id": "DOC-123_0001",
      "top_domain": "medical",
      "ambiguous": false,
      "domain_scores": [...]
    }
  ],
  "meta": {
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1,
    "always_include": ["generic"],
    "ambiguity_threshold": 0.1
  }
}
```

---

### 6) Mentions (NER/RE) 🔜

* Detecta menciones de entidades **condicionadas por tópicos**.

### 7–11) Clustering → Label → LLM → Normalización → Grafo 🔜

* Agrupan spans, etiquetan con weak supervision, refinan con LLM y normalizan entidades.
* Export final a **grafo de conocimiento**.

---


# 📋 Flujo de información en T2G (Contrado de datos)


| Etapa / Subsistema                  | **Entrada (qué toma)**                  | **Salida (qué agrega / construye)**                                                                               |
| ----------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **1. Parser** (Doc → IR)            | Documento bruto (PDF, DOCX, IMG)        | `DocumentIR`: texto por bloques, tablas, metadatos (`mime`, `pages`, `sha256`, `source_path`).                    |
| **2. Contextizer (doc-level)**      | `DocumentIR.pages.blocks.text`          | `meta.topics_doc`: tópicos globales, keywords generales, ejemplares, `outlier_ratio`.                             |
| **3. HybridChunker**                | `DocumentIR+Topics`                     | `chunks[*]`: segmentos ≤2048 tokens. Heredan `topic_hints` + métricas (`cohesion`, `redundancy`).                 |
| **4. Contextizer (chunk-level)**    | `chunks.text` + `topic_hints` heredados | `chunks[*].topic`: tópico local con `topic_id`, `keywords`, `prob` + `meta.topics_chunks`.                        |
| **5. Schema Selector (adaptativo)** | `chunks+topics` + `registry`            | `SchemaSelection`: dominios relevantes por doc y chunk + `evidence_trace` (keywords, embeddings, priors, scores). |
                            |

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
      
  - name: schema-select
    args:
      chunks_glob: "outputs_chunks/*.json"
      outdir: "outputs_schema"
      clean_outdir: true
      alpha_kw: 0.6
      beta_emb: 0.3
      gamma_prior: 0.1
      ambig_margin: 0.08
      topk_domains: 2
      topk_entity_types: 5
```

Ejecutar:

```bash
python t2g_cli.py pipeline-yaml
```

---

## 📊 Métricas por subsistema

### Parser 

* `percent_docs_ok`: éxito de parseo por lote.
* `layout_loss`: pérdida de estructura.
* `table_consistency`: tablas detectadas vs esperadas.

### Contextizer (doc-level) 

* `coverage`: proporción de chunks asignados a algún tópico.
* `outlier_rate`: ratio de outliers vs asignaciones válidas.
* `topic_size_stats`: distribución (min, mediana, p95).
* `keywords_diversity`: diversidad de keywords únicas.

### HybridChunker**

* `chunk_length_stats`: distribución de tamaño en caracteres/tokens.
* `cohesion_vs_doc`: similitud coseno chunk ↔ doc (coherencia semántica global).
* `max_redundancy`: similitud máxima entre chunks (redundancia bruta).
* `redundancy_norm`: redundancia normalizada según longitud media del documento  
  *(penaliza fragmentos largos y repetitivos)*.


### Contextizer (chunk-level)**

  * `coverage`: % de chunks con topic asignado.
  * `fallback_rate`: % de documentos donde se usó fallback vs BERTopic.
  * `topic_size_stats`: tamaño medio de clusters locales.

### Adaptive Schema Selector**

  * `domain_score_distribution`: histograma de scores por dominio.
  * `ambiguity_rate`: % de documentos/chunks con `ambiguous=True`.
  * `coverage_domains`: promedio de dominios relevantes por documento.
  * `evidence_trace`: lista de evidencias usadas para cada score (auditoría).

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

