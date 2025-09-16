# 📚 Proyecto T2G — Knowledge Graph a partir de Documentos

**T2G** es una *pipeline modular* para convertir documentos heterogéneos (PDF, DOCX, imágenes) en una **Representación Intermedia (IR) homogénea**, segmentarlos en **chunks** semánticos, luego en **oraciones filtradas**, y finalmente extraer **triples (S,R,O)** listos para RAG/IE/grafos.

* **Entrada:** PDF / DOCX / IMG
* **Salidas (hoy):** **IR (JSON)** → **Chunks (JSON)** → **Sentences (JSON)** → **Triples (JSON)**
* **Diseño:** subsistemas desacoplados, contratos claros, ejecución CLI/YAML

---

## ✨ Objetivos

* Convertir documentos heterogéneos en una **IR homogénea JSON** con bloques y tablas.
* Desarrollar, probar y orquestar los **subsistemas**: Parser, Chunker, Sentence/Filter, Triples (dep.), **Mentions (NER/RE)**, Normalización, Publicación, Retriever, Evaluación.
* Sentar base para **grafos de conocimiento**, **QA empresarial** y **compliance**.
* Mantener una arquitectura **escalable y modular**, con **contratos Pydantic** y CLIs consistentes.

---

## 🧩 Subsistemas (vivos y planeados)

| Nº | Subsistema            | Rol                                                      | I/O                                     | Estado |
| -: | --------------------- | -------------------------------------------------------- | --------------------------------------- | ------ |
|  1 | **Parser**            | Unificar formatos a **IR JSON/MD** con layout y tablas   | Doc → **IR**                            | ✅      |
|  2 | **HybridChunker**     | Chunks **cohesivos** con tamaños estables y solapamiento | IR → **Chunks**                         | ✅      |
|  3 | **Sentence/Filter**   | Dividir en **oraciones** y filtrar ruido antes de IE     | Chunks → **Sentences**                  | ✅      |
|  4 | **Triples (dep.)**    | (S,R,O) ligeros ES/EN (spaCy + regex)                    | Sentences → **Triples**                 | ✅      |
|  5 | **Mentions (NER/RE)** | Menciones de entidades/relaciones + consenso con Triples | Sentences → **Mentions**                | ✅      |
|  6 | Normalización         | Fechas, montos, IDs, orgs                                | Mentions → Entidades                    | 🕒     |
|  7 | Publicación           | Índices / grafo 1-hop                                    | Chunks/Ent/Triples → ES/Qdrant/PG/Grafo | 🕒     |
|  8 | Retriever (cascada)   | Recall → Precisión                                       | Query → Contexto                        | 🕒     |
|  9 | Evaluación & HITL     | Calidad / drift / lazo humano                            | Respuestas → Scores                     | 🕒     |
| 10 | **IE (orquestador)**  | Reusa/crea Triples y ejecuta Mentions con boost          | Sentences/(Chunks) → Triples+Mentions   | ✅      |

---

## 📂 Estructura del proyecto

```
project_T2G/
├── docs/                          # Documentos de prueba (PDF, DOCX, PNG, JPG)
├── parser/
│   ├── parsers.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── chunker/
│   ├── hybrid_chunker.py
│   ├── metrics.py
│   └── __init__.py
├── sentence_filter/
│   ├── sentence_filter.py
│   ├── metrics.py
│   ├── schemas.py
│   └── __init__.py
├── triples/
│   ├── dep_triples.py
│   ├── metrics.py
│   └── schemas.py
├── mentions/                      # NER/RE (con boost desde Triples y HF opcional)
│   ├── ner_re.py
│   ├── hf_plugins.py              # Wrapper HF local (opcional, offline)
│   ├── schemas.py
│   └── __init__.py
├── tools/
│   └── validate_ie.py             # Validación (Mentions soportadas por Triples)
├── pipelines/
│   └── pipeline.yaml              # Pipeline declarativo (YAML)
├── outputs_ir/
├── outputs_chunks/
├── outputs_sentences/
├── outputs_triples/
├── outputs_mentions/
├── outputs_metrics/               # Reportes/CSVs/plots (notebooks)
├── t2g_cli.py                     # CLI unificado: parse · chunk · sentences · triples · mentions · ie · pipeline-yaml
├── requirements.txt
└── README.md
```

> **Nota:** coloca tus archivos de prueba en `docs/`:
>
> ```
> docs/
> ├── Resume Martin Jurado_CDAO_24.pdf
> ├── tabla.png
> └── ejemplo.docx
> ```

---

## 🧠 Qué hace cada etapa


> Cada etapa toma una entrada bien definida y devuelve un JSON con **contrato Pydantic**. Abajo verás: **entrada → salida**, cómo decide, campos clave y flags útiles.

---

### 1) Parser (Doc → IR)

**Entrada:** PDF / DOCX / PNG / JPG
**Salida:** `DocumentIR` (`outputs_ir/{DOC}_*.json`)

**Qué hace:**

* Detecta tipo por **extensión/MIME** y enruta a un parser especializado.
* **PDF (pdfplumber):**

  * Extrae **texto por líneas** y **tablas** usando heurísticas de líneas (estrategias vertical/horizontal).
  * Si una página no tiene texto/tabla, aplica **OCR por página** (Tesseract) como *fallback*.
* **DOCX (python-docx):** lee párrafos y estilos (para *headings*), serializa tablas por celdas. Se considera una “página lógica”.
* **IMG (Pillow + Tesseract):** OCR directo; normaliza espacios y guiones rotos.
* **Normaliza**: `normalize_whitespace`, `dehyphenate`.
* **Anota metadatos:** `size_bytes`, `page_count`, `mime`, `source_path`, `created_at`.

**Contrato de salida (resumen):**

```json
{
  "doc_id": "DOC-XXXX",
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {"kind": "text", "text": "…", "bbox": [x0,y0,x1,y1]},
        {"kind": "table", "rows": [["A","B"],["C","D"]], "bbox": [ … ]},
        {"kind": "figure", "caption": "…", "bbox": [ … ]}
      ]
    }
  ],
  "meta": {"mime":"application/pdf","page_count":1,"source_path":"docs/x.pdf"}
}
```

**Flags útiles:** *(se configuran en el parser internamente; no hay flags CLI aquí)*

**Notas:**

* Si planeas OCR, instala Tesseract y su pack de idioma (es/en).
* IR es **la base de verdad** para el resto de etapas; si algo se ve raro aquí, arrastrará ruido después.

---

### 2) HybridChunker (IR → Chunks)

**Entrada:** `DocumentIR`
**Salida:** `DocumentChunks` (`outputs_chunks/{DOC}_chunks.json`)

**Qué hace:**

1. **Aplana** la IR a una secuencia `{kind, text, page_idx, block_idx}`.

   * `heading` se pasa a texto con prefijo `#` para conservar jerarquía (y se puede **pegar** con el siguiente bloque).
   * `table` se **serializa a texto** estilo CSV por filas (conserva contenido).
2. **Empaqueta** en chunks *cohesivos* con un objetivo de longitud (`target_chars`), sin superar `max_chars`.
3. Si un chunk excede, **corta por oración** usando spaCy si hay modelo; si no, **regex robusta**.
4. Aplica **solapamiento** (`overlap`) en caracteres para dar **contexto continuo**.
5. Etiqueta cada chunk como `text` / `table` / `mixed`.

**Contrato de salida (resumen):**

```json
{
  "doc_id": "DOC-XXXX",
  "chunks": [
    {
      "chunk_id": 0,
      "text": "…",
      "kind": "mixed",
      "meta": {
        "chars": 1042, "overlap": 120,
        "page_span": [0,0], "block_span": [3,7]
      }
    }
  ]
}
```

**Flags clave:**

* `--target-chars` (≈1400 recomendado), `--max-chars` (2048), `--min-chars` (400), `--overlap` (120).
* `--sentence-splitter {auto|spacy|regex}` (solo afecta cortes internos).
* `--table-policy {isolate|merge}`: mantener tablas separadas o fusionarlas cuando convenga.

**Consejos:**

* Si “rompe” demasiado los párrafos, baja `target_chars` o usa `spacy` en `sentence_splitter`.
* Si tienes muchos cuadros/tablas, prueba `table_policy=merge` para evitar chunks minúsculos.

---

### 3) Sentence/Filter (Chunks → Sentences)

**Entrada:** `DocumentChunks`
**Salida:** `DocumentSentences` (`outputs_sentences/{DOC}_sentences.json`)

**Qué hace:**

1. **Normaliza**: colapsa espacios, repara guiones de línea, elimina bullets.
2. **Divide en oraciones**: spaCy si está disponible; si no, **regex** (diseñada para no romper abreviaturas comunes).
3. **Filtra ruido**:

   * `min_chars` (descarta oraciones telegráficas),
   * `drop_stopword_only` y `drop_numeric_only`,
   * **dedupe**: `fuzzy` con `fuzzy_threshold` (evita repetir oraciones near-duplicadas).
4. **Traza** cada oración al chunk origen (y por transitividad a la IR).

**Contrato de salida (resumen):**

```json
{
  "doc_id": "DOC-XXXX",
  "sentences": [
    {"text":"…","chunk_id":0,"span":[s,e]}
  ],
  "meta": {
    "counters": {
      "total_split": 42, "kept": 28,
      "dropped_short": 9, "dropped_numeric": 2,
      "dropped_dupe": 3, "keep_rate": 0.67
    }
  }
}
```

**Flags clave:**

* `--sentence-splitter {auto|spacy|regex}`, `--min-chars 25`, `--dedupe fuzzy`, `--fuzzy-threshold 0.92`.
* Toggles: `--no-normalize-whitespace`, `--no-dehyphenate`, `--no-strip-bullets`, `--keep-stopword-only`, `--keep-numeric-only`.

**Consejos:**

* `keep_rate` entre **0.6–0.9** suele ser sano; muy bajo = filtros agresivos, muy alto = ruido.
* Si salen oraciones “cortadas”, fuerza `--sentence-splitter spacy`.

---

### 4) Triples (Sentences → Triples)

**Entrada:** `DocumentSentences`
**Salida:** `DocumentTriples` (`outputs_triples/{DOC}_triples.json`)

**Qué hace:**

* Extrae **(Sujeto, Relación, Objeto)** con un enfoque bilingüe **ES/EN**:

  * **spaCy (si disponible)**: reglas de dependencias (SVO, copulares, preposicionales, nominal+prep, apposición → `alias`, pasiva de adquisición, empleo/cargo).
  * **Regex de respaldo** si no hay spaCy o el árbol no ayuda (copulares, adquisición, empleo, prep genéricas).
* **Anti-ruido**: ignora líneas tipo contacto (URLs, emails, teléfonos), headings `#`, cadenas con casi solo números/puntuación.
* **Canonicaliza** relaciones (opcional): **superficies ES/EN** → **forma común** (p. ej., `trabaja_en` / `works at` → `works_at`).
* Asigna **confianza `conf`** por regla (regex genéricas ≈0.60; dependencias mayores).

**Contrato de salida (resumen):**

```json
{
  "doc_id": "DOC-XXXX",
  "triples": [
    {
      "subject":"Alice","relation":"works_at","object":"ACME",
      "meta":{
        "dep_rule":"VERB_prep_pobj_nsubj",
        "rel_surface":"works at",
        "conf":0.74,"lang":"en",
        "sentence_idx":12,"span":[s,e]
      }
    }
  ],
  "meta":{"counters":{"used_sents": 3}}
}
```

**Flags clave:**

* `--spacy {auto|force|off}` (fuerza dependencias o usa solo regex),
* `--min-conf-keep 0.66` (recomendado),
* `--max-triples-per-sentence 4`,
* `--lang {auto|es|en}`,
* `--no-canonicalize-relations` (si quieres conservar la superficie textual).

**Consejos:**

* Si ves demasiadas relaciones genéricas de preposición, **sube** `--min-conf-keep` a 0.70–0.72 o usa `--spacy force`.
* Para auditoría, une por `sentence_idx` con `DocumentSentences` y revisa `meta.rel_surface`.

---

### 5) Mentions (Sentences → Mentions) — **NER/RE + consenso con Triples**

**Entrada:** `DocumentSentences` (+ opcionalmente `DocumentTriples` para boost)
**Salida:** `DocumentMentions` (`outputs_mentions/{DOC}_mentions.json`)

**Qué hace:**

* Detecta **entidades** (regex/phrase/spaCy) y **relaciones** (DependencyMatcher/regex).
* **Consenso/boost** contra Triples existentes:

  * Si pasas `--boost-from-triples` (o hay `outputs_triples/*_triples.json`), cualquier relación mencionada en la **misma oración** y con la **misma relación** recibe un **incremento de confianza** (`boost_conf`), escalado por `triples_boost_weight`.
* **HF (opcional, offline):** re-ranker local si pones `--use-transformers` y `--hf-rel-model-path` a una carpeta con un modelo de clasificación de relaciones. Esto **no descarga** nada.
* Aplica filtros: `min_conf_keep`, `max_relations_per_sentence`, `canonicalize_labels`.

**Contrato de salida (resumen):**

```json
{
  "doc_id":"DOC-XXXX",
  "entities":[
    {"id":"E0","text":"Alice","label":"PERSON","canonical_label":"person","span":[s,e],"conf":0.91}
  ],
  "relations":[
    {
      "subj_entity_id":"E0","obj_entity_id":"E1",
      "label":"works_at","canonical_label":"works_at",
      "sentence_idx":12,"conf":0.78
    }
  ]
}
```

**Flags clave:**

* `--min-conf-keep 0.66`, `--max-relations-per-sentence 6`, `--canonicalize-labels`.
* `--boost-from-triples "outputs_triples/*_triples.json"`, `--boost-conf 0.05–0.12`, `--triples-boost-weight 1.0`.
* `--use-transformers`, `--hf-rel-model-path`, `--hf-device`, `--hf-batch-size`, `--hf-min-prob`, `--transformer-weight`.

**Consejos:**

* Si activas HF, **asegúrate** de que `hf_rel_model_path` apunte a un directorio local válido. Si no, déjalo apagado.
* La métrica **Support\@Triples** (en `tools/validate_ie.py`) te dice qué % de relaciones en Mentions están soportadas por Triples en la misma oración.

---

### 6) IE (Orquestador por documento) — **recomendado**

**Entrada:** `DocumentSentences` (o `chunks` si quieres que genere oraciones)
**Salida:** `DocumentTriples` + `DocumentMentions` (carpetas de salida)

**Qué hace:**

1. **Resuelve oraciones:**

   * Si das `--sents-glob`, usa esas oraciones.
   * Si no y pasas `--chunks-glob`, **genera** `DocumentSentences`.
   * Si no das nada, busca `outputs_sentences/*_sentences.json`.
2. **Triples**: si **no existen** (o limpiaste), los **genera**; si existen, los **reusa**.
3. **Mentions**: ejecuta NER/RE con **boost** desde los Triples producidos/encontrados.
4. **Validación opcional**: `--validate` corre `tools/validate_ie.py` y reporta **Support\@Triples**.

**Flags clave (las que realmente usarás):**

* Fuentes/destinos: `--sents-glob`, `--chunks-glob`, `--outdir-sentences`, `--outdir-triples`, `--outdir-mentions`.
* Calidad: `--min-conf-keep-triples`, `--min-conf-keep-mentions`, `--max-relations-per-sentence`, `--canonicalize-labels`, `--boost-conf`.
* HF (opcional): `--use-transformers`, `--hf-rel-model-path`, `--hf-device`.
* Ejecución: `--workers`, `--validate`, `--clean-outdir-mentions`, `--clean-outdir-triples`.

**Ejemplo típico (sin HF):**

```bash
python t2g_cli.py ie \
  --sents-glob "outputs_sentences/*_sentences.json" \
  --outdir-triples outputs_triples \
  --outdir-mentions outputs_mentions \
  --boost-conf 0.08 \
  --validate
```

**Consejos:**

* Si solo quieres **ajustar thresholds de Mentions** sin recalcular Triples, corre `ie` con `--clean-outdir-mentions` y deja Triples intactos.
* Si tus documentos son muy telegráficos, baja `--min-conf-keep-mentions` a 0.55–0.60 y usa un `boost_conf` un poco más alto (0.10–0.12).

---

**Checklist mental rápido**

* ¿IR se ve bien? → Sí: sigue. No: reintenta Parser u OCR.
* ¿Chunks dentro de \[400, 2048] con overlap ≈120? → Sí.
* ¿Sentences con `keep_rate` razonable (0.6–0.9)? → Sí.
* ¿Triples con `min-conf-keep ≥ 0.66` y relaciones canónicas útiles? → Sí.
* ¿Mentions con buen **Support\@Triples** (p. ej. ≥60–80% según dominio)? → Sí.

---


## ⚙️ Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# OCR (si planeas usar fallback OCR en PDF y/o IMG)
# macOS:  brew install tesseract tesseract-lang
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa

# (Opcional) spaCy para mejores cortes y dependencias (Triples)
pip install spacy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

---

## ⚙️ Instalación

> Recomendado: **Python 3.11**, entorno virtual, y (si vas a usar HF/torch) fijar `numpy<2` para evitar choques ABI.

```bash
# 1) Crear y activar venv
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2) Pip moderno
python -m pip install --upgrade pip wheel setuptools

# 3) Instalar dependencias del proyecto
pip install -r requirements.txt
```

### OCR (opcional pero útil en PDF/IMG sin texto)

```bash
# macOS (Homebrew)
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-spa

# Windows (choco)
choco install tesseract
```

### spaCy (opcional; mejora cortes y dependencias para Triples/Mentions)

```bash
pip install spacy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### (Opcional) PyTorch + Transformers para re-rank local en Mentions

> Si **no** vas a usar re-rank HF, puedes saltarte esta sección y dejar `--use-transformers` desactivado.

**CPU / macOS (MPS):**

```bash
pip install "torch>=2.1,<2.3" transformers>=4.38,<4.43
```

**GPU NVIDIA (CUDA 12.1, ejemplo):**

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install "transformers>=4.38,<4.43"
```

**Nota de compatibilidad (NumPy):**
Si ves el error *“A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x”*, fija:

```bash
pip install "numpy<2" --upgrade
```

**Modelo HF offline (si decides activarlo):**

```bash
# Descarga un modelo de clasificación de relaciones a una carpeta local
# (ejemplo ilustrativo — sustituye <repo_id>)
huggingface-cli snapshot-download <repo_id> \
  --local-dir hf_plugins/re_models/relation_mini

# Luego lo habilitas en CLI con:
# --use-transformers --hf-rel-model-path "hf_plugins/re_models/relation_mini"
```

---

## 🚀 Uso rápido (CLI)

> Puedes correr etapas sueltas o usar el **orquestador `ie`** (recomendado) o el **pipeline declarativo YAML**.

### 1) Parse → IR

Convierte documentos a **IR JSON**.

```bash
python t2g_cli.py parse docs/Resume\ Martin\ Jurado_CDAO_24.pdf --outdir outputs_ir
# Resultado: outputs_ir/DOC-XXXX.json
```

### 2) IR → Chunks

Crea chunks semánticos con solapamiento.

```bash
python t2g_cli.py chunk outputs_ir/DOC-XXXX.json --outdir outputs_chunks \
  --sentence-splitter auto --target-chars 1400 --max-chars 2048 --overlap 120
# Resultado: outputs_chunks/DOC-XXXX_chunks.json
```

### 3) Chunks → Sentences

Split por oraciones + filtros/normalización.

```bash
python t2g_cli.py sentences outputs_chunks/DOC-XXXX_chunks.json \
  --outdir outputs_sentences \
  --sentence-splitter auto \
  --min-chars 25 \
  --dedupe fuzzy \
  --fuzzy-threshold 0.92
# Resultado: outputs_sentences/DOC-XXXX_sentences.json
```

### 4) Sentences → Triples

Triples (S,R,O) bilingües con reglas de dependencias/regex.

```bash
python t2g_cli.py triples outputs_sentences/DOC-XXXX_sentences.json \
  --outdir outputs_triples \
  --lang auto --ruleset default-bilingual \
  --spacy auto --max-triples-per-sentence 4 \
  --min-conf-keep 0.66
# Tip: añade --no-canonicalize-relations si quieres conservar superficie cruda.
# Resultado: outputs_triples/DOC-XXXX_triples.json
```

### 5) Sentences → Mentions (NER/RE + consenso con Triples)

Si ya tienes triples y quieres iterar rápido en NER/RE:

```bash
python t2g_cli.py mentions outputs_sentences/DOC-XXXX_sentences.json \
  --outdir outputs_mentions \
  --lang auto --spacy auto \
  --min-conf-keep 0.66 \
  --max-relations-per-sentence 6 \
  --boost-from-triples "outputs_triples/*_triples.json" \
  --boost-conf 0.08
# Opcional HF:
# --use-transformers --hf-rel-model-path "hf_plugins/re_models/relation_mini" --hf-device cpu
# Resultado: outputs_mentions/DOC-XXXX_mentions.json
```

### 6) IE — Orquestador por documento (recomendado)

Hace: **(reusa o crea) Triples → Mentions con boost**.
Si ya corriste `sentences`, esta es la forma más práctica:

```bash
python t2g_cli.py ie \
  --sents-glob "outputs_sentences/*_sentences.json" \
  --outdir-triples outputs_triples \
  --outdir-mentions outputs_mentions \
  --boost-conf 0.08 \
  --validate
# Limpia menciones para iterar thresholds sin recalcular triples:
#   añade: --clean-outdir-mentions
# Apaga spaCy (regex-only): --spacy off
# Activa HF offline:
#   --use-transformers --hf-rel-model-path "hf_plugins/re_models/relation_mini" --hf-device cpu
```

### 7) Pipeline declarativo (YAML)

Corre todo el flujo desde `pipelines/pipeline.yaml`:

```bash
python t2g_cli.py pipeline-yaml
# o
python t2g_cli.py pipeline-yaml --file pipelines/pipeline.yaml
```

---

### Verificación rápida de salidas

```bash
# Revisa contadores
jq '.meta.counters // {}' outputs_sentences/*_sentences.json | head
jq '.meta.counters // {}' outputs_triples/*_triples.json | head
jq '.meta.counters // {}' outputs_mentions/*_mentions.json | head

# Valida soporte Triples→Mentions
python tools/validate_ie.py --mentions outputs_mentions --triples outputs_triples
```

**Consejos finales:**

* Si ves demasiadas relaciones “genéricas”, sube `--min-conf-keep` o usa `--spacy force`.
* Si los documentos son telegráficos, baja `--min-conf-keep` en Mentions (0.55–0.60) y usa `--boost-conf 0.10–0.12`.
* Si activas HF y no hay modelo local válido, el sistema sigue corriendo **sin** HF (log de advertencia).
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
      clean_outdir: true
      ir_glob: "outputs_ir/*.json"
      outdir: "outputs_chunks"
      target_chars: 1400
      max_chars: 2048
      min_chars: 400
      overlap: 120
      table_policy: "isolate"
      sentence_splitter: "auto"

  - name: sentences
    args:
      clean_outdir: true
      chunks_glob: "outputs_chunks/*_chunks.json"
      outdir: "outputs_sentences"
      sentence_splitter: "auto"
      min_chars: 25
      dedupe: "fuzzy"
      fuzzy_threshold: 0.92
      no_normalize_whitespace: false
      no_dehyphenate: false
      no_strip_bullets: false
      keep_stopword_only: false
      keep_numeric_only: false

  # IE orquestador (recomendado)
  - name: ie
    args:
      sents_glob: "outputs_sentences/*_sentences.json"
      outdir_triples: "outputs_triples"
      outdir_mentions: "outputs_mentions"
      lang: "auto"
      spacy: "auto"
      min_conf_keep_triples: 0.66
      min_conf_keep_mentions: 0.66
      max_relations_per_sentence: 6
      canonicalize_labels: true
      boost_conf: 0.08
      use_transformers: false         # true si tienes modelo HF local
      # hf_rel_model_path: "hf_plugins/re_models/relation_mini"
      # hf_device: "cpu"
      workers: 4
      validate: true
      clean_outdir_triples: false
      clean_outdir_mentions: true
```

> **Alternativas avanzadas:**
> • Ejecutar **Triples** y **Mentions** como etapas independientes (útil para auditar o ajustar thresholds sin recalcular).
> • Si omites `sentences`, puedes pasar `chunks_glob` al `ie` para que genere oraciones internamente.

---
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
from chunker.schemas import DocumentChunks
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

### Triples

* **Agregados:** nº de documentos, **nº de triples**, **unique ratio** (`drop_duplicates` por S,R,O).
* **Distribución de relaciones** y **reglas** (qué patrones producen qué).
* **`conf` stats** y efecto de `--min-conf-keep`.
* **Auditoría con contexto:** join contra oraciones por `sentence_idx`.

```python
import json, glob, pandas as pd
from triples.schemas import DocumentTriples

rows = []
for p in sorted(glob.glob("outputs_triples/*_triples.json")):
    dt = DocumentTriples(**json.load(open(p)))
    for t in dt.triples:
        rows.append({
          "doc_id": dt.doc_id,
          "subject": t.subject, "relation": t.relation, "object": t.object,
          "dep_rule": t.meta.get("dep_rule"), "conf": t.meta.get("conf"),
          "lang": t.meta.get("lang"), "sentence_idx": t.meta.get("sentence_idx"),
          "file": p
        })

df = pd.DataFrame(rows)
print("Docs:", df["doc_id"].nunique(), "| Triples:", len(df))
print("Unique ratio:", df.drop_duplicates(["subject","relation","object"]).shape[0]/max(1,len(df)))
print("Top relaciones:\n", df["relation"].value_counts().head())
print("Top reglas:\n", df["dep_rule"].value_counts().head())
print("Conf stats:\n", df["conf"].describe())
```

### **Mentions (NER/RE)**

**Agregados:**

* Nº de entidades y relaciones por doc.
* Distribución por etiquetas (`entity.label`, `relation.canonical_label`).
* **Soporte por Triples**: % de relaciones de Mentions con pareja `(doc_id, sentence_idx, relation)` presente en Triples.

**Snippet (soporte vs Triples):**

```python
import json, glob, pandas as pd

# Carga Mentions
mrs = []
for p in sorted(glob.glob("outputs_mentions/*_mentions.json")):
    dm = json.load(open(p, "r", encoding="utf-8"))
    for r in dm.get("relations", []):
        mrs.append({
          "doc_id": dm.get("doc_id"),
          "sentence_idx": r.get("sentence_idx"),
          "label": r.get("canonical_label") or r.get("label") or "",
          "subj": r.get("subj_entity_id"),
          "obj": r.get("obj_entity_id"),
          "conf": r.get("conf", 0.0),
          "file": p
        })
rel_df = pd.DataFrame(mrs)

# Carga Triples
trs = []
for p in sorted(glob.glob("outputs_triples/*_triples.json")):
    dt = json.load(open(p, "r", encoding="utf-8"))
    for t in dt.get("triples", []):
        trs.append({
          "doc_id": dt.get("doc_id"),
          "sentence_idx": (t.get("meta", {}) or {}).get("sentence_idx"),
          "relation": t.get("relation") or "",
          "file": p
        })
tri_df = pd.DataFrame(trs)

# Keys y soporte
if not rel_df.empty and not tri_df.empty:
    rel_df["key"] = rel_df.apply(lambda r: f"{r['doc_id']}::{int(r['sentence_idx'])}::{str(r['label']).lower()}", axis=1)
    tri_df["key"] = tri_df.apply(lambda r: f"{r['doc_id']}::{int(r['sentence_idx'])}::{str(r['relation']).lower()}", axis=1)
    tri_keys = set(tri_df["key"].tolist())
    rel_df["supported"] = rel_df["key"].isin(tri_keys)
    print("Support@Triples:", f"{rel_df['supported'].mean()*100:.2f}%", f"({rel_df['supported'].sum()}/{len(rel_df)})")
    print(rel_df.groupby(["label","supported"]).size().unstack(fill_value=0).head())
else:
    print("No hay datos suficientes para calcular soporte.")
```

**Export a grafo (opcional):**

```python
import json, glob, networkx as nx

G = nx.MultiDiGraph()
for p in sorted(glob.glob("outputs_mentions/*_mentions.json")):
    dm = json.load(open(p, "r", encoding="utf-8"))
    ents = {e["id"]: e for e in dm.get("entities", [])}
    for r in dm.get("relations", []):
        s = ents.get(r["subj_entity_id"]); o = ents.get(r["obj_entity_id"])
        if not s or not o: continue
        G.add_node(s["id"], label=s.get("canonical_label") or s.get("label"), text=s["text"])
        G.add_node(o["id"], label=o.get("canonical_label") or o.get("label"), text=o["text"])
        G.add_edge(s["id"], o["id"], r=r.get("canonical_label") or r.get("label"), conf=r.get("conf", 0.0))
nx.write_gexf(G, "outputs_metrics/mentions_graph.gexf")
print("Grafo exportado a outputs_metrics/mentions_graph.gexf")
```

**Auditoría con oración original:**

```python
import json, pandas as pd

doc = "DOC-XXXX"
dt = json.load(open(f"outputs_triples/{doc}_triples.json"))
ds = json.load(open(f"outputs_sentences/{doc}_sentences.json"))

trip = pd.json_normalize(dt["triples"])
trip["sentence_idx"] = trip["meta.sentence_idx"]
sents = pd.DataFrame(ds["sentences"]).assign(sentence_idx=lambda d: range(len(d)))
audit = trip.merge(sents[["sentence_idx","text"]], on="sentence_idx", how="left")

cols = ["subject","relation","object","meta.conf","meta.dep_rule","text"]
print(audit[cols].head(12))
```

**Sugerencias de calidad:**

* Usa `--min-conf-keep 0.66` para recortar `regex_prep` genéricas.
* Si hay ruido de preposiciones, **sube** a 0.70–0.72 o activa spaCy (`--spacy force`) para más precisión estructural.
* Desactiva `--no-canonicalize-relations` si necesitas la superficie cruda (útil para depuración).

---

## 🛠️ Troubleshooting

* **JSONs vacíos/corruptos:** escritura **atómica**, el runner **salta** JSON inválidos. Limpia y re-ejecuta:

  ```bash
  rm -rf outputs_ir/* outputs_chunks/* outputs_sentences/* outputs_triples/*
  python t2g_cli.py pipeline-yaml
  ```
* **spaCy no carga / modelos faltantes:** instala `es_core_news_sm` y `en_core_web_sm` o usa `--spacy off` (regex-only).
* **OCR fallback no funciona:** instala Tesseract y verifica `PATH` (o `tesseract_cmd` en Windows).
* **Imports fallan:** ejecuta desde la **raíz** (`python t2g_cli.py …`).
* **Triples con “of/in/with” dominantes:** eleva `--min-conf-keep` o fuerza spaCy.
* **Pandas `InvalidIndexError` en auditorías:** usa `reset_index(drop=True)` antes de `concat/merge`.
* **Carpetas con “ruido” de corridas previas:** en YAML pon `clean_outdir: true` en cada etapa.

---

## 🧪 Roadmap inmediato

* NER/RE y normalización (fechas, montos, IDs).
* Publicación a ES/Qdrant/Postgres/Grafo (con esquemas para triples).
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
