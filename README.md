# News RAG Pipeline

This project builds a local Python RAG pipeline over Economic Times Tech articles. It is split into four checkpoint scripts so each stage can be run and debugged independently.
 
## Setup And Run

The repository includes pre-extracted article records in `data/records/`, so you can evaluate indexing and querying immediately without waiting for web scraping.

### 1. Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> **Note on API Keys:** `GEMINI_API_KEY` is completely optional. Without a key, the pipeline runs deterministically with full source citations. With a key added to `.env` or passed via `--api-key`, Gemini provides an LLM-synthesized answer.

---

### 2. Quick Evaluation (Instant Testing)

Index the pre-extracted articles and test queries:

```bash
# Index records into ChromaDB
python index.py

# Query the corpus
python query.py "List all articles mentioning Bengaluru"

# Run the automated sample evaluation suite
python query.py --run-samples

# Run the complete unit test suite
pytest
```

---

### 3. Full Pipeline from Scratch (Scraping & Extraction)

If you wish to test live scraping and parsing end-to-end:

```bash
python scrape.py   # Discovers and downloads latest ET Tech articles into data/raw/
python extract.py  # Cleans & extracts records (fast deterministic ~5s; use --use-gemini for LLM refinement)
python index.py    # Chunks and indexes records into ChromaDB
```

### Running on Other Devices & API Key Configuration

To run this pipeline seamlessly on different machines or with different API keys, you can provide your Google Gemini API key in any of the following ways:

1. **`.env` File (Recommended for Local Dev)**:
   Add your key to `.env`:
   ```env
   GEMINI_API_KEY="your-gemini-api-key-here"
   GEMINI_MODEL="gemini-3.8-flash"  # Optional: defaults to gemini-3.8-flash
   ```

2. **Environment Variable (Recommended for CI / Remote Machines)**:
   ```powershell
   # PowerShell
   $env:GEMINI_API_KEY = "your-gemini-api-key-here"
   
   # Bash / macOS / Linux
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```
   *(Note: `GOOGLE_API_KEY` is also supported as an alias).*

3. **Command-Line Argument (Ad-hoc overrides)**:
   ```powershell
   python query.py "What funding rounds happened?" --api-key "your-key"
   python query.py "Explain AI models" --api-key "your-key" --model "gemini-3.8-flash"
   python extract.py --api-key "your-key"
   ```

Never add a real key to `.env.example` or commit `.env`. If a key is exposed, revoke it in Google AI Studio and create a replacement before using the pipeline again.

## Corpus

The chosen news site is Economic Times Tech. `scrape.py` discovers article URLs from the configured RSS feed and a small set of ET Tech category pages, then fetches up to `MAX_ARTICLES` articles while respecting `robots.txt`, request delays, timeouts, retries, and idempotent raw-file reuse. Newsletter URLs are deliberately excluded. If `robots.txt` is unavailable, the crawler fails closed and records the event; if fewer than `MIN_ARTICLES` permitted URLs are discovered, it stops before fetching a partial corpus.

Raw HTML is stored in `data/raw/<sha256-prefix>.html`. `data/raw/manifest.json` records the stable id, canonical URL, fetch time, and HTTP status. Failures go to `data/raw/failed_urls.log`.

## Stage Design

`scrape.py` keeps network crawling separate from downstream parsing. It hashes canonical URLs, skips files already on disk, and maintains a manifest so reruns do not duplicate work.

`extract.py` reads raw HTML and writes Pydantic v2 validated records to `data/records/<id>.json`. It uses trafilatura first for article text, JSON-LD (including `@graph`) and metadata for publication dates, heuristic extraction for entities/category/funding, and optional Google Gemini JSON mode (`gemini-3.8-flash` via LangChain) to refine summaries and structured fields. A record with no publisher-supplied, parseable date is rejected and logged rather than assigned the scrape time.

`index.py` reads validated records, chunks article bodies, and indexes LangChain `Document` objects into a local persistent ChromaDB collection via `langchain-chroma` and `langchain-huggingface` embeddings. Re-indexing replaces existing chunks for each article and removes stale chunks, so filtered-out records cannot remain searchable.

`query.py` uses a rule-based planner to identify metadata filters such as category, date range, location, company, title match, and aggregation intent. Exact list/title/location queries operate on matching records rather than approximate vector rankings; the repeated-investor query is calculated from validated funding metadata. Other queries use Chroma retrieval with a lexical fallback. Gemini synthesis is optional and the no-key path is deterministic and source-backed.

## Extraction Accuracy

Extraction accuracy is measured by spot-checking a small sample of raw HTML files against the corresponding JSON records: title, date, author, body cleanliness, summary faithfulness, entity lists, and category. The extractor is deliberately conservative: if Gemini is unavailable or returns malformed JSON, the deterministic heuristics still produce valid records and log only true validation failures.

## Chunking And Embeddings

The pipeline implements **recursive character chunking** using LangChain's `RecursiveCharacterTextSplitter` with hierarchical separators (`["\n\n", "\n", ". ", " ", ""]`). The default chunk size is 1200 characters with 200 characters of overlap. This ensures paragraphs and sentences are preserved intact wherever possible, avoiding unnatural cuts across semantic boundaries while maintaining precise retrieval context. The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` because it is small, local, and fast enough for a coursework-sized corpus. ChromaDB is persisted under `data/chroma`.

## Observability & Debugging with LangSmith

LangSmith tracing is integrated across all four stages of the pipeline to monitor execution, track latency, and pinpoint failures:

- **Scraping (`scrape.py`)**: Traces discovery (`scrape_discover_urls`), raw downloads (`scrape_fetch_url`), HTTP status codes, and robots.txt checks.
- **Extracting (`extract.py`)**: Traces HTML cleaning/parsing (`extract_article`), schema validation, and Gemini JSON enrichment (`ask_gemini_enrichment`).
- **Indexing (`index.py`)**: Traces batch chunking, document creation, and vector store upserts into ChromaDB (`index_pipeline`).
- **Querying (`query.py` & `src/news_rag/langchain_rag.py`)**: Traces query planning (`rag_query_pipeline`), vector retrieval (`chroma_retrieval`), lexical fallback (`lexical_retrieval`), and Gemini answer generation (`gemini_answer_generation`).

### Enabling LangSmith Tracing

1. Add your LangSmith API key to `.env`:
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your-langsmith-api-key-here
   LANGSMITH_PROJECT=news-rag-pipeline
   ```
2. Run any stage (`python scrape.py`, `python extract.py`, `python index.py`, or `python query.py ...`).
3. View the trace trees, run durations, input/output payloads, and error stack traces in real time on [smith.langchain.com](https://smith.langchain.com).
4. If `LANGSMITH_API_KEY` is not provided, tracing is disabled gracefully without any errors or overhead.

## Known Limitations

The entity and funding heuristics are intentionally lightweight and can confuse names, companies, and branded phrases when running without LLM refinement. Economic Times page structure can change; a changed structure may cause date extraction failures or fewer than `MIN_ARTICLES` permitted URLs, which the pipeline now reports instead of silently fabricating data or indexing a partial corpus. Local embeddings (`sentence-transformers/all-MiniLM-L6-v2` via HuggingFace) keep indexing private and offline, while Gemini can optionally improve extraction and answer synthesis.

## Next Steps

With another week, I would add a manual evaluation sheet for 25 records, expand date parsing for more publisher formats, add named-entity recognition with spaCy or GLiNER, expose a richer CLI for source inspection, and add an automated retrieval-quality test set.
