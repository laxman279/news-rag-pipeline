# News RAG Pipeline

A production-grade, modular Retrieval-Augmented Generation (RAG) pipeline designed for crawling, extracting, indexing, and querying news articles from **The Economic Times Tech**. Built with Python, LangChain, ChromaDB, Sentence-Transformers, GLiNER, and Google Gemini.

The system is partitioned into four independent, checkpointed stages:
1. **Scraping (`scrape.py`)**: Ethical RSS and category discovery with `robots.txt` compliance, rate limiting, and raw HTML archival.
2. **Extraction (`extract.py`)**: Neural zero-shot Named Entity Recognition (GLiNER), Trafilatura article body extraction, JSON-LD metadata parsing, and Pydantic v2 validation.
3. **Indexing (`index.py`)**: Semantic recursive character chunking, local sentence transformer embeddings, and persistent vector upsert with stale-chunk pruning in ChromaDB.
4. **Querying (`query.py`)**: Structured query planning, hybrid vector/metadata retrieval, grounded Gemini answer synthesis with LangChain LCEL, and exact citation attribution.

---

## 1. Setup in Under 10 Commands

You can set up and evaluate the pipeline from scratch in **under 10 commands**:

### Linux / macOS
```bash
git clone <repo-url> && cd news-rag-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python index.py
python query.py "What AI stories are covered?"
pytest
```

### Windows (PowerShell)
```powershell
git clone <repo-url>; cd news-rag-pipeline
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python index.py
python query.py "What AI stories are covered?"
pytest
```

> **API Key Note:** `GEMINI_API_KEY` is completely optional. Without an API key, the pipeline operates in a deterministic, fully source-cited extractive mode. If a key is provided in `.env` or passed via `--api-key`, the pipeline uses Gemini (`gemini-3.8-flash`) for structured query parsing and grounded answer synthesis.

---

## 2. Running Each Stage Separately

Every pipeline stage runs as an independent CLI script checkpointed via the filesystem (`data/raw/`, `data/records/`, and `data/chroma/`):

```bash
# Stage 1: Crawl ET Tech articles into data/raw/
python scrape.py

# Stage 2: Extract clean records into data/records/
python extract.py

# Stage 2 (Optional): Extract with LLM refinement
python extract.py --use-gemini --api-key YOUR_API_KEY

# Stage 3: Chunk and index records into ChromaDB
python index.py

# Stage 4: Query the RAG corpus
python query.py "List all articles mentioning Bengaluru"

# Stage 4: Run the full automated sample evaluation suite
python query.py --run-samples
```

---

## 3. Design Choices and Trade-Offs Per Stage

### Stage 1: Web Scraping (`scrape.py`)
- **Design Choices**:
  - **Discovery**: Discovers article URLs using the official Economic Times Tech RSS feed combined with targeted category index pages (`/tech/technology`, `/tech/startups`, `/tech/artificial-intelligence`, etc.).
  - **Polite Crawler Architecture**: Strictly parses and obeys `robots.txt` rules using Python's `urllib.robotparser`. If `robots.txt` cannot be fetched, the crawler fails closed and aborts rather than risking unpermitted crawling.
  - **Throttling & Resilience**: Implements polite request delays (`REQUEST_DELAY_SECONDS=1.5`) with jitter, standard descriptive `User-Agent` headers, request timeouts (20s), and exponential backoff retries via `tenacity`.
  - **Idempotent Archival**: Hashes canonical URLs with SHA-256 (`<hash[:16]>.html`) and maintains `data/raw/manifest.json`. Rerunning `scrape.py` skips already-downloaded articles.
- **Trade-Offs**:
  - *Requests + BeautifulSoup vs. Headless Browser (Playwright/Selenium)*: We chose static HTTP (`requests`) for speed, minimal resource usage, and zero browser daemon dependencies. The trade-off is the inability to scrape dynamic JavaScript-hydrated single-page applications (SPAs); however, Economic Times Tech serves full server-side rendered (SSR) HTML for article bodies and metadata.

### Stage 2: Extraction & Cleaning (`extract.py`)
- **Design Choices**:
  - **Multi-Pass Body Extraction**: Uses `trafilatura` as the primary body extractor due to its benchmark-proven boilerplate removal, falling back to targeted BeautifulSoup CSS selectors (`.artText`, `.Normal`).
  - **Structured Metadata & Timestamps**: Extracts canonical headlines, author lists, and publication timestamps directly from publisher JSON-LD (`@graph` NewsArticle schema) and meta tags. All dates are normalized to UTC RFC3339 format (`YYYY-MM-DDTHH:MM:SSZ`).
  - **Zero-Shot Neural NER**: Utilizes `GLiNER` (`urchade/gliner_small-v2.1`) for high-precision entity extraction across companies, people, and locations.
  - **Multi-Category Classification**: Combines URL path analysis (e.g. `/tech/funding/` -> `funding`), metadata keywords, and semantic classification to assign multiple category tags per article (`categories: [...]`) and identify out-of-taxonomy domains (e.g., `quantum computing`).
  - **Structured Funding Parsing**: Parses round name, amount, currency, startup name, and investor lists into a validated `Funding` model.
  - **Strict Validation**: Rejects and logs records lacking publisher-supplied dates or clean text via Pydantic v2 `ArticleRecord`.
- **Trade-Offs**:
  - *Neural Zero-Shot NER (GLiNER) vs. Pure Regex vs. Pure LLM*: Pure regex is brittle and prone to false positives on multi-word company names (e.g., "Tata Sons", "Scale AI", "Superintelligence Labs"). LLM-only extraction is slow, expensive, and subject to rate limits. GLiNER provides local, sub-second zero-shot entity extraction with high accuracy and zero API cost.

### Stage 3: Chunking & Indexing (`index.py`)
- **Design Choices**:
  - **Recursive Character Splitting**: Employs LangChain's `RecursiveCharacterTextSplitter` with chunk size of 1,000 characters and 200 characters overlap.
  - **Local Dense Embeddings**: Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`.
  - **Persistent Vector Store**: Uses ChromaDB (`chromadb.PersistentClient`) persisted under `data/chroma/`.
  - **Incremental Upsert & Stale Chunk Pruning**: Re-indexing replaces chunks for existing articles and deletes orphaned chunk IDs, ensuring removed or filtered records cannot linger in the search index.
- **Trade-Offs**:
  - *Local Sentence-Transformers vs. Commercial API Embeddings (e.g. Google text-embedding-004)*: Local embeddings run completely offline, have zero latency overhead from network calls, incur zero cost, and guarantee reproducible embeddings across all environments.

### Stage 4: Query Planning, Retrieval & Synthesis (`query.py` & `src/news_rag/langchain_rag.py`)
- **Design Choices**:
  - **Structured Query Planning**: A deterministic query planner (`plan_query`) detects structured filtering criteria: specific categories, date ranges, cities/locations, company names, quoted title lookups, and aggregation intents.
  - **Dual-Path Hybrid Retrieval**:
    - *Direct Record Matching*: Exact structured queries (e.g., "List all articles mentioning Bengaluru", quoted title matches, repeated investor counts) operate directly over validated record metadata for 100% precision.
    - *Dense Vector Retrieval with Native Grouping*: Open-ended queries perform semantic search on ChromaDB, grouping chunks by article ID to return diverse, non-redundant contexts.
    - *Lexical Fallback*: Automatically triggers keyword retrieval if ChromaDB is unavailable or yields zero matches.
  - **Strict Grounding & Answer Synthesis**: Uses LangChain LCEL with Gemini (`ChatGoogleGenerativeAI`) instructed to answer strictly based on provided contexts and respond with `"The answer cannot be found in the corpus."` when information is missing.
  - **Precise Citation Attribution**: Scans synthesized answers to cite *only* the specific article URLs whose facts or titles were actually used.
- **Trade-Offs**:
  - *Pure Semantic Search vs. Hybrid Planning*: Standard vector search performs poorly on aggregations (e.g. "Which investors appear in >1 round?") and exact phrase matching. Hybrid routing combines the speed and flexibility of vector search with the precision of structured relational queries.

---

## 4. Extraction Method and Why

Accurate extraction from raw news HTML is challenging due to publisher ads, syndication sidebars, inline video players, and inconsistent metadata formatting. We selected a multi-layered hybrid extraction pipeline:

1. **Trafilatura for Clean Text**:
   - *Why*: In standard NLP benchmarks, Trafilatura consistently outperforms readability algorithms and raw CSS scraping by stripping ads, comments, navigation bars, and image captions while preserving paragraph structure.
2. **JSON-LD Schema Parsing**:
   - *Why*: Rather than scraping fragile DOM headers, `parse_json_ld` extracts publisher-validated headlines, authors, and RFC3339 timestamps from embedded `<script type="application/ld+json">` blocks (supporting nested `@graph` structures). Articles lacking an authentic publisher date are rejected rather than assigned an inaccurate crawl timestamp.
3. **GLiNER Neural Token Classification (`urchade/gliner_small-v2.1`)**:
   - *Why*: Traditional spaCy models struggle on Indian tech entities and novel AI startup names. GLiNER uses bidirectional transformer representations to extract arbitrary zero-shot entity types (`company`, `person`, `location`) in single-batch chunked passes.
4. **Boundary Heuristics & Boilerplate Pruning**:
   - *Why*: Post-processing filters strip common news syndication boilerplate (e.g., "Latest News Updates on The Economic Times", "Subscribe Now", relationship modifiers like "backed by") from entity strings to eliminate merged or corrupted names.
5. **Pydantic v2 Schema Enforcement**:
   - *Why*: Guarantees data integrity, strict type validation, and immutable data modeling across the entire ingest pipeline.

---

## 5. Chunking and Embedding Choices

| Parameter / Choice | Selection | Rationale |
| :--- | :--- | :--- |
| **Splitter** | `RecursiveCharacterTextSplitter` | Splits hierarchically on `["\n\n", "\n", ". ", " ", ""]`, respecting natural paragraph and sentence boundaries. |
| **Chunk Size** | 1,000 characters (~150-200 words) | Compact enough to focus on single facts or funding deals, large enough to retain full contextual meaning. |
| **Chunk Overlap** | 200 characters (~30-40 words) | Prevents loss of continuity or fractured entities across chunk boundaries. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Fast, lightweight 384-dimensional model optimized for sentence and paragraph similarity. Operates locally on CPU without rate limits. |
| **Vector Database** | ChromaDB (Persistent) | Local, embedded, serverless vector store stored under `data/chroma/`. |

---

## 6. Observability & Tracing with LangSmith

LangSmith tracing is integrated across all four pipeline stages to monitor latency, inspect inputs/outputs, and debug retrieval quality:

- **Scraping**: Traces URL discovery, robots compliance, and HTTP download latency.
- **Extraction**: Traces HTML cleaning, GLiNER inference, and Pydantic validation.
- **Indexing**: Traces chunk splitting and ChromaDB upsert operations.
- **Querying**: Traces structured planning, vector retrieval, and Gemini LCEL chain synthesis.

To enable tracing:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="news-rag-pipeline"
```
*(If no LangSmith key is provided, tracing is disabled gracefully without runtime overhead).*

---

## 7. Data Sample and Evaluation Results

The repository includes a curated data sample and a full evaluation report:
- **`data/raw/`**: 5 representative raw HTML files and `manifest.json`.
- **`data/records/`**: 5 corresponding extracted JSON records validating titles, authors, categories, dates, entities, and summaries.
- **`results/sample_queries.md`**: Evaluation report containing verified answers and exact source citations across all sample queries:
  1. *What funding rounds are mentioned in the corpus?*
  2. *List all articles mentioning Bengaluru.*
  3. *What does the corpus say about AI?*
  4. *Find the article titled "Meta says its new AI models on par with OpenAI, Anthropic"*
  5. *Which investors are appearing in >1 funding round?*
  6. *What AI stories are covered?*
  7. *Summarize layoffs in the corpus.* (Negative control — correctly reports unmentioned)
  8. *What is the weather today?* (Unanswerable control — correctly rejected)

---

## 8. Known Limitations

1. **Static HTML vs. Dynamic JavaScript**:
   - The scraper relies on `requests` and server-rendered HTML. If The Economic Times shifts to client-side rendered Single Page Applications (SPAs) or enables dynamic paywalls, articles will require a headless browser.
2. **Entity Ambiguity in Short Acronyms**:
   - Short two-letter tokens (e.g. "AI", "US") require regex contextual guards to avoid misclassification as locations or organizations.
3. **Single-Node Local ChromaDB**:
   - ChromaDB is embedded locally via SQLite and DuckDB/hnswlib. While ideal for local development, production multi-tenant environments would require a managed, distributed vector database (e.g., Qdrant, Pinecone).
4. **Publisher DOM Drift**:
   - While JSON-LD extraction is resilient, articles published without JSON-LD metadata rely on CSS selector fallbacks that require maintenance when the publisher updates their frontend layouts, which could cause extraction failures or fewer than `MIN_ARTICLES` discovered.
5. **Extraction Runtime at Scale**:
   - `extract.py` utilizes GLiNER zero-shot neural token classification on CPU, which takes ~1-2s per article. Processing hundreds of articles without GPU acceleration or batching can be time-consuming.
6. **Entity Alias Variation Across Articles**:
   - A company or person may be referenced in varied surface forms across articles (e.g., "Tata Sons", "Tata Group", "Tata Electronics"). Full cross-document coreference and entity deduplication require canonical knowledge-base entity linking.

---

## 9. What I Would Do With Another Week

If given an additional week, I would implement:
1. **Headless Browser Crawler Fallback**: Integrate Playwright with stealth mode to bypass anti-scraping challenges and crawl JavaScript-rendered archives.
2. **Cross-Encoder Reranking**: Add a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top 25 retrieved vector candidates to rerank chunks by semantic relevance before passing context to the LLM.
3. **Graph RAG / Knowledge Graph Integration**: Build an entity relation graph in NetworkX or Neo4j linking Startups $\leftrightarrow$ Funding Rounds $\leftrightarrow$ Investors $\leftrightarrow$ Founders, enabling multi-hop queries (e.g., "Which co-investors participated in rounds alongside Accel?").
4. **Automated Evaluation Benchmark**: Implement automated RAG evaluation metrics using `ragas` or `TruLens` to measure Answer Faithfulness, Context Relevance, and Hallucination Frequency on synthetic test sets.
5. **Interactive UI**: Develop a lightweight, responsive web interface (FastAPI + Streamlit or Next.js) with streaming generation, interactive source highlighting, and one-click trace inspection.
