# 📚 T2G Project — Documents to Knowledge Graph (Updated Feb 2026)

> Status: **parse**, **contextize-doc**, **chunk**, **contextize-chunks**, **schema-select**, **mentions (LLM)** and **graph-builder (Neo4j)** are all implemented and callable from the CLI.

T2G is a **modular, extensible pipeline** that converts heterogeneous documents (PDF/DOCX/IMG) into a unified **Intermediate Representation (IR)**, enriches them with **global and local semantic context**, and prepares them for **schema-aware entity extraction** and **knowledge graph export**.

- **Inputs:** PDF / DOCX / PNG / JPG / (CSV in YAML example)
- **Outputs (current):**
  - `DocumentIR` JSON
  - `DocumentIR+Topics` JSON (doc-level)
  - `DocumentChunks` JSON (chunked + topics)
  - `SchemaSelection` JSON
  - `Mentions` JSON (LLM extraction)
  - `Graph` ingestion for Neo4j

---

## ✨ Goals

- Normalize ingestion into a **common IR JSON** regardless of format.
- Enrich with **semantic topics** at document and chunk level.
- Keep the architecture **decoupled and auditable** via Pydantic contracts.
- Provide the foundation for **knowledge graphs**, **enterprise QA**, **compliance**, and **RAG**.

---

## 🧩 Subsystems

| # | Subsystem | Role | Input | Output | Status |
|--:|-----------|------|-------|--------|--------|
| 1 | **Parser** | IR JSON with layout + metadata | Doc (PDF/DOCX/IMG) | `DocumentIR` JSON | ✅ |
| 2 | **Hybrid Contextizer (doc)** | Global topics & keywords | `DocumentIR` | `DocumentIR+Topics` JSON | ✅ |
| 3 | **HybridChunker** | Semantic chunks (≤ tokens cap) + hints | `DocumentIR+Topics` | `DocumentChunks` JSON | ✅ |
| 4 | **Hybrid Contextizer (chunk)** | Local topics per chunk (hybrid by default) | `DocumentChunks` | `Chunks+Topics` JSON | ✅ |
| 5 | **Adaptive Schema Selector** | Domain/schema selection with explainability | `Chunks+Topics` | `SchemaSelection` JSON | ✅ |
| 6 | **Mentions (LLM)** | Schema-aware entity mentions | `Chunks+Topics` + schema | `Mentions` JSON | ✅ |
| 7 | **Graph Builder (Neo4j)** | Export entities/relations to graph | IR + Mentions | Neo4j + `outputs_graph/` | ✅ |

---

## 📂 Project Structure (updated)

```bash
project_T2G/
├── parser/                  # Parsing (PDF/DOCX/IMG → IR)
│   ├── parsers.py           # Core logic (pdfplumber, python-docx, OCR)
│   ├── metrics.py           # Parse metrics
│   ├── schemas.py           # Pydantic contracts (DocumentIR)
│   └── helpers.py
├── contextizer/             # Doc + chunk topic modeling (hybrid/light)
│   ├── contextizer.py       # Adaptive router
│   ├── hybrid/              # Hybrid engines (TF-IDF + KeyBERT + embeddings + DBSCAN + MMR)
│   ├── metrics.py, metrics_ext.py, models.py, schemas.py, utils.py
├── chunker/                 # HybridChunker (semantic segmentation + metrics)
│   ├── chunker.py, schemas.py, metrics.py
├── schema_selector/         # Adaptive Schema Selector 2.0
│   ├── selector.py, registry.py, registry_embeddings.py, schemas.py, utils.py
├── mentions/                # LLM-based mentions extraction (batch over chunks)
│   ├── llm_extractor.py, prompt_builder.py, llm_client.py, schemas.py, metrics.py
├── graph_builder/           # Neo4j ingestion
│   ├── graph_ingestor.py, neo4j_client.py, utils.py, metrics.py
├── pipelines/pipeline.yaml  # Declarative pipeline
├── t2g_cli.py               # Unified CLI (all stages enabled)
├── outputs_ir/              # IR JSON
├── outputs_doc_topics/      # IR + doc topics
├── outputs_chunks/          # Chunks + topics
├── outputs_schema/          # SchemaSelection results
├── outputs_mentions/        # LLM mentions
├── outputs_graph/           # Graph export artifacts
├── outputs_prompts/         # Saved prompts (MENTIONS_DEBUG)
├── outputs_metrics/         # Collected metrics
├── analysis_outputs/, t2g_evaluation/ # notebooks & studies
├── docs/                    # Samples / benchmarks
├── requirements.txt
└── README.md
```

---

## 🧠 Pipeline Stages (with real defaults)

### 1) Parser (Doc → IR) ✅
**Input:** PDF / DOCX / PNG / JPG  
**Output:** `DocumentIR` (`outputs_ir/{doc_id}.json`)

- Auto-detects format; pdfplumber + tables, python-docx, OCR fallback (pytesseract) for scanned pages if enabled.
- Normalizes whitespace, dehyphenates, preserves `text_raw`.
- Adds metadata: `sha256`, `mime`, `page_count`, `size_bytes`.
- Language hints per block/page/doc (langdetect + stopword density).
- Headings/list detection heuristics (configurable).
- Metrics: `percent_docs_ok`, `layout_loss`, `table_consistency`, `ocr_ratio`, `avg_parse_time`, `block_density`.

### 2) Hybrid Contextizer (doc-level) ✅
**Input:** `DocumentIR`  
**Output:** `DocumentIR+Topics` (`*_doc_topics.json`)

- Hybrid mode (default): embeddings + TF-IDF + KeyBERT + DBSCAN density clustering + MMR.
- Light mode: TF-IDF (+ optional KeyBERT) without embeddings/clustering.
- Router runs via CLI flags `--use-hybrid/--disable-hybrid`.
- Metrics saved in `meta.topics_doc` (entropy, variance, coherence, redundancy, diversity).
- Quantitative signals:
  - Topic entropy \(H = -\sum p_j \log p_j\).
  - Semantic variance over topic exemplars.
  - Redundancy score (keyword overlap).
  - Keywords diversity (unique/total).

### 3) HybridChunker ✅
**Input:** `DocumentIR+Topics`  
**Output:** `DocumentChunks` (`outputs_chunks/*.json`)

- Sentence/heading-based segmentation + semantic packing (≤ `max_tokens`, default 2048).
- Inherits `topic_hints` from doc-level.
- Metrics per chunk: `cohesion_vs_doc`, `max_redundancy`, `redundancy_norm`, `novelty`, `chunk_health`, lexical density, TTR.
- Uses embeddings and spaCy when available; regex fallbacks otherwise.
- Global metrics (see `chunker/metrics.py`):
  - `semantic_coverage` = % chunks with `cohesion_vs_doc ≥ 0.7`
  - `redundancy_flag_rate` = % chunks with `redundancy_norm ≥ 0.6`
  - `global_health_score` combines chunk_health & cohesion.

### 4) Hybrid Contextizer (chunk-level) ✅
**Input:** `DocumentChunks`  
**Output:** Enriched `DocumentChunks` (same file) with `meta.topics_chunks`

- **Default = hybrid ON** (embeddings + clustering). Disable with `--disable-hybrid` for light mode.
- Reuses chunk texts; assigns local topics and keywords; applies MMR if enabled.
- Computes extended metrics (`context_alignment`, `redundancy_penalty`).
- Topic affinity per chunk blends cosine + Jaccard between chunk keywords and domain aliases.

### 5) Adaptive Schema Selector 2.0 ✅
**Input:** `DocumentChunks` with topics  
**Output:** `SchemaSelection` (`outputs_schema/{doc}_schema.json`)

- Signal fusion: `S_d = α·K + β·E + γ·C + δ·T + ε·P`
  - `K`: keyword F1 vs domain aliases
  - `E`: cosine vs domain label vectors (centroids from `registry_vectors.json`)
  - `C`: context quality (cohesion, health, 1−redundancy, novelty, richness)
  - `T`: topic affinity (doc/chunk keywords ↔ domain aliases)
  - `P`: priors
- **CLI defaults:** α=0.30, β=0.25, γ=0.25, δ=0.15, ε=0.05. (Pipeline YAML example uses α=0.20, β=0.40, γ=0.20, δ=0.15, ε=0.05)
- Produces explainable `DecisionTrace` + `Evidence`, ambiguity flag, softmax confidence, per-chunk selection limited to evaluated domains.
- Embeddings for domains are loaded from cache (`schema_selector/registry_vectors.json`) or rebuilt on demand.
- Context score \(C = 0.30\,\text{cohesion} + 0.30\,\text{chunk\_health} + 0.20(1-\text{redundancy\_norm}) + 0.10\,\text{novelty} + 0.10\,\text{richness}\).
- Ambiguity rule: `ambiguous = |S1 − S2| < τ` (τ = `ambig_threshold`).

### 6) Mentions (LLM, schema-aware) ✅
**Input:** `DocumentChunks` + `SchemaSelection`  
**Output:** `outputs_mentions/*.json`

- Batch over chunks (`MENTIONS_BATCH_SIZE`, default 3) to control context length.
- Prompts saved to `outputs_prompts/` when `MENTIONS_DEBUG=1`.
- Re-tags generic mentions using registry hints; deduplicates and averages confidence.
- Requires OpenAI (or configured provider) credentials via environment (`.env`).
- Metrics: confidence averages, duplicate reduction count, batch stats.

### 7) Graph Builder (Neo4j) ✅
**Input:** `outputs_ir` + `outputs_mentions`  
**Output:** `outputs_graph/` + writes to Neo4j

- Uses `graph_builder/neo4j_client.py` with retries and idempotent constraints.
- Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (or pass via CLI).
- Stats reported: new/existing documents, entities, relations, labels created, reused edges.

---

## 📋 Data Flow Contract

| Stage | Input | Output |
|-------|-------|--------|
| Parser | Raw doc | `DocumentIR` (pages, blocks, tables, meta) |
| Contextizer (doc) | `DocumentIR.pages.blocks.text` | `meta.topics_doc` (topics, keywords, metrics) |
| Chunker | `DocumentIR+Topics` | `chunks[*]` + `topic_hints` + chunk metrics |
| Contextizer (chunk) | `chunks.text` | `chunks[*].topic` + `meta.topics_chunks` |
| Schema Selector | `chunks+topics` + registry | `SchemaSelection` with domain scores & evidence |
| Mentions | chunks + schema | Mention spans with domain/type/confidence |
| Graph Builder | IR + mentions | Nodes/edges in Neo4j + graph metrics |

---

## 📂 Pipeline YAML (current)

File: `pipelines/pipeline.yaml`

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
        - "docs/*.docx"
        - "docs/*.png"
        - "docs/*.jpg"
        - "docs/*.csv"
      outdir: "outputs_ir"

  - name: contextize-doc
    args:
      clean_outdir: true
      ir_glob: "outputs_ir/*.json"
      embedding_model: "all-MiniLM-L6-v2"
      nr_topics: null
      seed: 42
      outdir: "outputs_doc_topics"
      use_hybrid: true
      use_keybert: true
      enable_mmr: true
      hybrid_eps: 0.25
      hybrid_min_samples: 2
      cache_dir: "cache/"

  - name: chunk
    args:
      clean_outdir: true
      ir_glob: "outputs_doc_topics/*.json"
      outdir: "outputs_chunks"
      max_tokens: 1024
      min_chars: 280
      use_embeddings: true
      embedding_model: "all-MiniLM-L6-v2"
      spacy_model: "es_core_news_sm"
      seed: 42

  - name: contextize-chunks
    args:
      chunks_glob: "outputs_chunks/*.json"
      embedding_model: "all-MiniLM-L6-v2"
      nr_topics: null
      seed: 42
      outdir: "outputs_chunks"
      use_keybert: true
      enable_mmr: true
      fusion_weights: [0.5, 0.3, 0.2]
      use_hybrid: true   # set false to force light mode

  - name: schema-select
    args:
      clean_outdir: true
      chunks_glob: "outputs_chunks/*.json"
      outdir: "outputs_schema"
      alpha_kw: 0.20
      beta_emb: 0.40
      gamma_ctx: 0.20
      delta_top: 0.15
      epsilon_prior: 0.05
      ambig_threshold: 0.10
      fallback_threshold: 0.10
      softmax_temp: 0.86
      disable_generic: false
      topk_domains: 5
      max_domains: 5

  - name: mentions
    args:
      chunks_glob: "outputs_chunks/*.json"
      schema_dir: "outputs_schema"
      outdir: "outputs_mentions"
      clean_outdir: true
      llm_model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 1024
      confidence_threshold: 0.10

  - name: graph-builder
    args:
      clean_outdir: true
      ir_glob: "outputs_ir"
      mentions_glob: "outputs_mentions"
      outdir: "outputs_graph"
      continue_on_error: true
```

Run: `python t2g_cli.py pipeline-yaml`

---

## 📊 Key Metrics

### Parser
- `percent_docs_ok`, `layout_loss`, `table_consistency`, `ocr_ratio`, `avg_parse_time`, `block_density`.

### Contextizer (doc)
- `coverage`, `outlier_rate`, `topic_size_stats`, `keywords_diversity`, `topic_entropy`.
- Extended: `entropy_topics`, `redundancy_score`, `keywords_diversity_ext`, `semantic_variance`, `coherence_semantic`, `context_quality`.

### HybridChunker
- Length stats, `coverage_rate`, `boundary_alignment`.
- `cohesion_vs_doc`, `max_redundancy`, `redundancy_norm`, `novelty`.
- Composite: `chunk_health`, `semantic_coverage`, `redundancy_flag_rate`, `global_health_score`.

### Contextizer (chunk)
- `coverage`, `fallback_rate`, `topic_size_stats`, `keywords_overlap`, `topic_coherence_local`, `local_entropy`.
- Extended: `redundancy_penalty`, `context_alignment`.

### Adaptive Schema Selector
- `domain_score_distribution`, `ambiguity_rate`, `domain_confidence_gap`, `prior_influence`, `always_included_rate`.
- Context boost effects (`contextual_boost_effect`, `lambda_effectiveness`), ontology coverage metrics.

---

## 🧮 Key Metric Formulas

- Selector score per domain:  
  \(S_d = \alpha K + \beta E + \gamma C + \delta T + \varepsilon P\)  
  where \(K\)=keyword F1 vs aliases, \(E\)=cosine(centroid, domain vector),  
  \(C = 0.30\,\text{cohesion} + 0.30\,\text{chunk\_health} + 0.20(1-\text{redundancy\_norm}) + 0.10\,\text{novelty} + 0.10\,\text{richness}\);  
  \(T\)=topic affinity (Jaccard keywords ↔ aliases), \(P\)=prior.

- Chunker health:  
  \(\text{chunk\_health} = \text{cohesion\_vs\_doc} \times (1 - \text{max\_redundancy})\)

- Redundancy norm:  
  \(\text{redundancy\_norm} = \text{max\_redundancy} \times \frac{\text{len(chunk)}}{\text{avg len(chunks)}}\)

- Topic entropy (doc-level):  
  \(H = - \sum_j p_j \log p_j\) where \(p_j\) is topic proportion.

- MMR (keyword diversification):  
  \(\text{MMR}(w_i)=\lambda \cos(w_i, t) - (1-\lambda)\max_{w_j\in S}\cos(w_i, w_j)\)

---

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# spaCy models
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
# OCR (macOS example)
brew install tesseract tesseract-lang
```

### Optional / performance
- `torch` (CPU/MPS), `transformers`, `umap-learn`, `joblib`, `matplotlib`.

### Domain embeddings cache
```bash
python -m schema_selector.registry_embeddings   # regenerates registry_vectors.json after ontology changes
```

### Credentials
- OpenAI / provider for `mentions`: set in `.env` or environment variables (e.g., `OPENAI_API_KEY`).
- Neo4j: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

---

## 🚀 Quick CLI Usage

```bash
# Parser → IR
python t2g_cli.py parse docs/sample.pdf --outdir outputs_ir

# Doc-level context
python t2g_cli.py contextize-doc outputs_ir/*.json --outdir outputs_doc_topics

# Chunking
python t2g_cli.py chunk outputs_doc_topics/*.json --outdir outputs_chunks

# Chunk-level context (hybrid by default; add --disable-hybrid for light)
python t2g_cli.py contextize-chunks outputs_chunks/*.json

# Schema selection
python t2g_cli.py schema-select outputs_chunks/*.json --outdir outputs_schema

# Mentions (LLM)
python t2g_cli.py mentions --chunks-glob \"outputs_chunks/*.json\" --schema-dir outputs_schema

# Graph export to Neo4j
python t2g_cli.py graph-builder --ir-glob outputs_ir --mentions-glob outputs_mentions
```

---

## 🧭 Design Notes

- **Decoupled contracts:** every stage uses Pydantic models; JSON artifacts are stable across stages.
- **Hybrid semantics:** TF-IDF + KeyBERT + embeddings + DBSCAN + MMR; switchable to light mode for speed/cost.
- **Vertical consistency:** global topics → `topic_hints` in chunks → domain selection → mentions prompts.
- **Explainability:** selector returns decision trace; chunker/contextizer keep metrics; graph builder logs counters.
- **Robust fallbacks:** regex sentence split if spaCy missing; embeddings optional; OCR only if enabled.

---

## 🔍 Known Caveats

- Imports may load heavy models (SentenceTransformer) unless cached; consider warming cache.
- `contextize-chunks` overwrites chunk files in place; run on copies if you need baselines.
- Parser uses permissive try/except in some fallbacks; enable logging to monitor degradations.
- Provide API keys and Neo4j creds via environment to avoid runtime failures.

---

## 📅 Changelog (high level)
- Feb 2026: README translated to English; aligned defaults with CLI; marked contextize-chunks, mentions, and graph-builder as implemented; documented selector weights and hybrid defaults.
