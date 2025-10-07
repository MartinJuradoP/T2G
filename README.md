# 📚 Proyecto T2G — Knowledge Graph a partir de Documentos

**T2G** es una *pipeline modular y extensible* que convierte documentos heterogéneos (PDF, DOCX, imágenes) en una **Representación Intermedia (IR) homogénea**, los enriquece con **contexto semántico global y local**, y prepara la base para construir **grafos de conocimiento** y sistemas de **búsqueda avanzada (RAG, QA, compliance, etc.)**.

* **Entrada :** PDF / DOCX / PNG / JPG
* **Salidas :**

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

---

**Qué hace:**

El **Adaptive Schema Selector (ASS)** determina dinámicamente qué **dominios de entidades** (por ejemplo, médico, legal, financiero o genérico) son relevantes para cada documento y chunk.
Su propósito es **filtrar, priorizar y contextualizar** los tipos de entidades que deben extraerse en las etapas siguientes, mejorando la **precisión semántica** del grafo y reduciendo ruido.

---

1. **Registro de dominios (`registry.py`)**

   Cada dominio está definido en la ontología base (`registry.py`) y contiene:

   * **Entidades (`EntityTypeDef`)** con atributos (por ejemplo: `Disease`, `Treatment`, `Contract`, `Transaction`).
   * **Relaciones (`RelationTypeDef`)** entre entidades (por ejemplo: `treated_with`, `paid_by`, `binds`).
   * **Aliases** y vocabulario específico en español e inglés (por ejemplo: `"enfermedad"`, `"patología"`, `"disease"` para `Disease`).
   * **Descripciones semánticas** utilizadas para generar embeddings de referencia.

   Los dominios incluidos en la versión `v2_bilingual` son:

   | Dominio            | Ejemplo de entidades                                   | Contextos típicos                         |
   | ------------------ | ------------------------------------------------------ | ----------------------------------------- |
   | `medical`          | `Disease`, `Symptom`, `Drug`, `Treatment`, `LabTest`   | artículos clínicos, diagnósticos          |
   | `legal`            | `Contract`, `Party`, `Obligation`, `Penalty`           | contratos, cláusulas, litigios            |
   | `financial`        | `Invoice`, `Transaction`, `StockIndicator`, `Policy`   | facturas, informes financieros            |
   | `reviews_and_news` | `Review`, `NewsArticle`, `MarketEvent`                 | reseñas, noticias económicas              |
   | `ecommerce`        | `Order`, `Product`, `Review`                           | comercio electrónico, reseñas de clientes |
   | `identity`         | `Person`, `Address`, `IDDocument`                      | registros, formularios                    |
   | `generic`          | `Person`, `Organization`, `Date`, `Location`, `Amount` | fallback universal                        |

---

2. **Extracción de señales del documento/chunk**

   El selector combina **tres tipos de señales** para estimar la afinidad de cada texto con los dominios registrados:

   * **Keywords**
     Se detectan coincidencias entre los tokens normalizados del documento y los `aliases` del dominio.
     Las coincidencias se ponderan por frecuencia y relevancia POS (sustantivos, nombres propios, etc.).

     $$
     S_{kw}(d) = \frac{\text{overlaps}(d)}{\text{total aliases}(d)} \times \log(1 + f_{term})
     $$

     Donde:

     * $\text{overlaps}(d)$ → número de alias del dominio encontrados.

     * $f_{term}$ → frecuencia media de los términos coincidentes.

     > Ejemplo: un documento con “contrato”, “firma”, “cláusula” activará el dominio `legal` con alto $S_{kw}$.

   * **Embeddings**
     Calcula la similitud coseno entre los **embeddings promedio del texto** y los **embeddings representativos del dominio** (precalculados a partir de sus descripciones y aliases).

     $$
     S_{emb}(d) = \cos(\vec{v}*{text}, \vec{v}*{domain})
     $$

     * $\vec{v}_{text}$ → embedding medio del chunk o documento.

     * $\vec{v}_{domain}$ → embedding medio del dominio.

     > Ejemplo: “antihipertensivo” activa el dominio `medical` aunque la palabra “enfermedad” no aparezca explícitamente.

   * **Priors**
     Cada dominio tiene un peso base $P(d)$ que refleja su probabilidad a priori de aparecer.

     $$
     P(d) = \text{prior}(d) \in [0, 1]
     $$

     > Ejemplo: `generic = 0.1`, `medical = 0.05`, `legal = 0.05`
     > El dominio `generic` siempre se considera como fallback.

---

3. **Fórmula de scoring (por dominio)**

   Para cada dominio $d$, se calcula un score ponderado combinando las tres señales:

   $$
   \text{score}(d) = \alpha \cdot S_{kw}(d) + \beta \cdot S_{emb}(d) + \gamma \cdot P(d)
   $$

   Donde:

   * $S_{kw}(d)$: score normalizado por coincidencia léxica.
   * $S_{emb}(d)$: similitud coseno entre embeddings.
   * $P(d)$: prior asignado al dominio.
   * $\alpha, \beta, \gamma$: hiperparámetros configurables en `SelectorConfig`.

   **Ejemplo default:**

   $\alpha = 0.6, ; \beta = 0.3, ; \gamma = 0.1$

   → más peso a keywords, menor a embeddings y priors.

   > En textos técnicos (contratos, facturas) domina $\alpha$.
   > En textos conceptuales (reseñas o informes), $\beta$ captura mejor la afinidad semántica.

---

4. **Selección final**

   Una vez calculados los scores, se aplica la fase de decisión:

   * Se **ordenan** los dominios de mayor a menor score.
   * Se **descartan** los dominios con score < `min_topic_conf`.
   * Se **seleccionan** los `top-k` dominios configurados (por defecto `topk_domains = 2`).
   * Se marca `ambiguous=True` si la diferencia entre el primer y segundo dominio es menor al margen definido:

     $$
     ambiguous = |S(d_1) - S(d_2)| < \tau
     $$

     donde $\tau$ es `ambiguity_threshold` (por defecto 0.1).
   * Siempre se incluye el dominio **genérico** como fallback (`allow_fallback_generic=True`).

---

**Ejemplo de salida (simplificado):**

```json
{
  "doc": {
    "doc_id": "DOC-CA28DAF58CC7",
    "top_domains": ["financial", "reviews_and_news"],
    "ambiguous": false,
    "domain_scores": [
      {
        "domain": "financial",
        "score": 0.27,
        "evidence": [
          {"kind": "keyword", "detail": {"match": ["pago","banco","transacción"], "kw_score": 0.23}},
          {"kind": "embedding", "detail": {"cosine": 0.84}},
          {"kind": "prior", "detail": {"value": 0.05}}
        ]
      },
      {
        "domain": "reviews_and_news",
        "score": 0.15,
        "evidence": [
          {"kind": "keyword", "detail": {"match": ["noticia","acciones"], "kw_score": 0.11}},
          {"kind": "embedding", "detail": {"cosine": 0.72}}
        ]
      }
    ]
  },
  "chunks": [
    {
      "chunk_id": "DOC-CA28DAF58CC7_0001",
      "top_domain": "financial",
      "ambiguous": false,
      "domain_scores": [
        {"domain": "financial", "score": 0.31},
        {"domain": "reviews_and_news", "score": 0.12}
      ]
    }
  ],
  "meta": {
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1,
    "always_include": ["generic"],
    "ambiguity_threshold": 0.1,
    "registry_version": "v2_bilingual"
  }
}
```

---

**Notas adicionales:**

* La afinidad entre dominios puede ajustarse con un refuerzo contextual:

  $$
  S_{\text{adj}}(d, c_i)
  = S_{\text{domain}}(d, c_i)\,\times\,
  \Big(1 + \lambda \cdot \text{cohesion\_vs\_doc}(c_i)\Big)
  $$

  donde \\( \lambda = 0.2 \\) pondera la cohesión semántica entre chunk y documento.

* La inferencia final combina contexto global y local:

  $$
  S_{\text{final}}(d)
  = \omega \cdot S_{\text{doc}}(d)
  + (1 - \omega) \cdot \frac{1}{N}\sum_{i=1}^{N} S_{\text{chunk}}(d, c_i)
  $$

  con \\( \omega = 0.5 \\) por defecto.



---

**Beneficios:**

* **Precisión contextual:** sólo se procesan entidades coherentes con el dominio dominante.
* **Escalabilidad:** nuevos dominios pueden añadirse fácilmente al `registry.py`.
* **Interpretabilidad:** cada decisión conserva su `evidence_trace` (palabras clave, similitudes, priors).
* **Auditable:** todos los pesos, fórmulas y umbrales se guardan en el `meta` del JSON.
* **Consistencia vertical:** mantiene coherencia entre documento y chunks.

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

---

### Parser 
Evalúa la calidad y consistencia del parseo de documentos heterogéneos.

- `percent_docs_ok`: proporción de documentos parseados sin errores.
- `layout_loss`: pérdida de estructura visual o de formato.
- `table_consistency`: coherencia entre tablas detectadas y esperadas.
- `ocr_ratio`: porcentaje de páginas procesadas mediante OCR (indicador de calidad visual).
- `avg_parse_time`: tiempo promedio de procesamiento por documento.
- `block_density`: número promedio de bloques válidos por página.

---

### Contextizer (doc-level) 
Mide la calidad del modelado temático global.

- `coverage`: proporción de bloques asignados a algún tópico.
- `outlier_rate`: ratio de bloques descartados por ruido o baja densidad.
- `topic_size_stats`: distribución del tamaño de clusters (`min`, `median`, `p95`).
- `keywords_diversity`: diversidad de palabras clave únicas (riqueza semántica).
- `topic_stability`: correlación promedio entre embeddings de tópicos en runs sucesivos (indicador de consistencia temporal).
- `topic_entropy`: medida de dispersión temática (mayor = más heterogeneidad).

---

### HybridChunker 
Evalúa la **coherencia**, **redundancia** y **salud semántica** de los fragmentos.

#### 🔹 Métricas base
- `chunk_length_stats`: distribución de tamaños (caracteres / tokens).
- `cohesion_vs_doc`: similitud coseno entre embedding de chunk y embedding global del documento.  
  $$\text{cohesion\_vs\_doc}(c_i) = \cos(\vec{c_i}, \bar{\vec{D}})$$
- `max_redundancy`: similitud máxima entre embeddings de chunks.  
  $$\text{max redundancy}(c_i) = \max_{j \neq i} \cos(\vec{c_i}, \vec{c_j})$$
- `redundancy_norm`: redundancia ajustada por longitud.  
  $$\text{redundancy norm}(c_i) = \text{max redundancy}(c_i) \times \frac{\text{len}(c_i)}{\text{avg len(chunks)}}$$
- `novelty`: proporción de información nueva aportada.  
  $$\text{novelty}(c_i) = 1 - \text{max redundancy}(c_i)$$

#### 🔹 Métricas compuestas
- `chunk_health`: salud semántica = cohesión × (1 − redundancia).  
  $$\text{chunk health}(c_i) = \text{cohesion\_vs\_doc}(c_i) \times (1 - \text{max redundancy}(c_i))$$
- `semantic_density`: proporción de tokens relevantes (sin stopwords) sobre el total.
- `lexical_density`: densidad léxica medida por términos significativos / totales.
- `type_token_ratio`: diversidad de vocabulario (variedad léxica).
- `semantic_coverage`: % de chunks con cohesión ≥ 0.7 (bien alineados al documento).
- `redundancy_flag_rate`: % de chunks con redundancia excesiva ≥ 0.6.
- `topic_affinity_blend`: afinidad semántico-léxica con los tópicos globales del documento.  
  $$\text{topic affinity blend}(c,t) = \alpha \cos(\vec{c}, \vec{t}) + (1-\alpha)J(c,t)$$

#### 🔹 Métricas globales
- `global_health_score`: indicador compuesto (`good`, `moderate`, `poor`).
- `avg_chunk_health`: promedio de salud semántica global.
- `coverage_ratio`: proporción de texto total cubierto por chunks válidos.
- `oversegmentation_rate`: % de chunks demasiado pequeños (bajo umbral `min_chars`).
- `undersegmentation_rate`: % de chunks demasiado largos (superan `max_tokens`).

---

### Contextizer (chunk-level) 
Evalúa la coherencia temática local y su relación con los tópicos globales.

- `coverage`: % de chunks con tópico asignado.
- `fallback_rate`: % de documentos donde se usó fallback en lugar de BERTopic.
- `topic_size_stats`: distribución de tamaños de subtemas.
- `keywords_overlap`: solapamiento promedio entre keywords globales y locales.
- `topic_coherence_local`: coherencia intracluster promedio (similitud coseno media entre embeddings de un mismo tema).
- `local_entropy`: dispersión de tópicos locales (mide estabilidad semántica).

---

### Adaptive Schema Selector 
Evalúa la **relevancia y precisión contextual** del mapeo dominio–documento.

#### 🔹 Métricas base
- `domain_score_distribution`: histograma de scores por dominio.  
  $$\text{score}(d) = \alpha S_{\text{kw}}(d) + \beta S_{\text{emb}}(d) + \gamma P(d)$$
- `coverage_domains`: número promedio de dominios relevantes por documento.
- `ambiguity_rate`: % de documentos o chunks marcados como `ambiguous = True`.
- `domain_confidence_gap`: diferencia entre el primer y segundo dominio (medida de separabilidad).
- `prior_influence`: peso efectivo de los priors sobre el score final.
- `always_included_rate`: % de documentos donde el dominio genérico fue incluido por fallback.

#### 🔹 Métricas de refuerzo contextual
- `contextual_boost_effect`: variación media del score tras aplicar refuerzo semántico.  
  $$\Delta S = S_{\text{adj}} - S_{\text{domain}}$$
- `lambda_effectiveness`: sensibilidad del refuerzo de cohesión (variación promedio por unidad de λ).  
  $$\eta_\lambda = \frac{\Delta S}{\lambda}$$

#### 🔹 Métricas de calidad ontológica
- `schema_alignment`: similitud promedio entre entidades detectadas y entidades esperadas del dominio.
- `entity_type_coverage`: % de tipos de entidad del dominio detectados al menos una vez.
- `relation_type_coverage`: % de relaciones del dominio identificadas.
- `ontology_diversity`: número de dominios distintos presentes en el corpus.

#### 🔹 Métricas globales
- `domain_precision`: proporción de dominios correctamente asignados (vs. gold standard si existe).
- `domain_recall`: proporción de dominios relevantes detectados.
- `domain_f1`: media armónica entre precisión y recall (solo si hay ground truth disponible).
- `evidence_trace`: trazabilidad completa de evidencias (keywords, embeddings, priors, scores).

---



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

