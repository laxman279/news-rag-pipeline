# News RAG Pipeline
demo video link : https://drive.google.com/file/d/19qm0IWlTLDP6NMdXIt35xvvgZ_fmJHeq/view?usp=sharing

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

### 2. Full Pipeline from Scratch (Scraping & Extraction)

If you wish to test live scraping and parsing end-to-end:

```bash
python scrape.py   # Discovers and downloads latest ET Tech articles into data/raw/
python extract.py  # Cleans & extracts records (fast deterministic ~5s; use --use-gemini for LLM refinement)
python index.py    # Chunks and indexes records into ChromaDB
```

### Running on Other Devices & API Key Configuration

To run this pipeline seamlessly on different machines or with different API keys, you can provide your Google Gemini API key in any of the following ways:

1. **Environment Variable**:
   ```powershell
   # PowerShell
   $env:GEMINI_API_KEY = "your-gemini-api-key-here"
   
   # Bash / macOS / Linux
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```
   *(Note: `GOOGLE_API_KEY` is also supported as an alias).*

2. **Command-Line Argument**:
   ```powershell
   python query.py "What funding rounds happened?" --api-key "your-key"
   python query.py "Explain AI models" --api-key "your-key" --model "gemini-3.8-flash"
   python extract.py --api-key "your-key"
   ```

## Corpus

The chosen news site is Economic Times Tech. `scrape.py` discovers article URLs from the configured RSS feed and a small set of ET Tech category pages, then fetches up to `MAX_ARTICLES` articles while respecting `robots.txt`, request delays, timeouts, retries, and idempotent raw-file reuse. Newsletter URLs are deliberately excluded. If `robots.txt` is unavailable, the crawler fails closed and records the event; if fewer than `MIN_ARTICLES` permitted URLs are discovered, it stops before fetching a partial corpus.

Raw HTML is stored in `data/raw/<sha256-prefix>.html`. `data/raw/manifest.json` records the stable id, canonical URL, fetch time, and HTTP status. Failures go to `data/raw/failed_urls.log`.

## Stage Design

`scrape.py` keeps network crawling separate from downstream parsing. It hashes canonical URLs, skips files already on disk, and maintains a manifest so reruns do not duplicate work.

`extract.py` reads raw HTML and writes Pydantic v2 validated records to `data/records/<id>.json`. It uses trafilatura first for article text, JSON-LD (including `@graph`) and metadata for publication dates, GLiNER (`urchade/gliner_small-v2.1`) zero-shot Named Entity Recognition for high-precision extraction of companies, people, and locations, deterministic sentence summarization, and rule-based category/funding parsing. A record with no publisher-supplied, parseable date is rejected and logged rather than assigned the scrape time.

`index.py` reads validated records, chunks article bodies, and indexes LangChain `Document` objects into a local persistent ChromaDB collection via `langchain-chroma` and `langchain-huggingface` embeddings. Re-indexing replaces existing chunks for each article and removes stale chunks, so filtered-out records cannot remain searchable.

`query.py` uses a rule-based planner to identify metadata filters such as category, date range, location, company, title match, and aggregation intent. Exact list/title/location queries operate on matching records rather than approximate vector rankings; the repeated-investor query is calculated from validated funding metadata. Other queries use Chroma retrieval with a lexical fallback. Gemini synthesis is optional and the no-key path is deterministic and source-backed.

## Extraction Accuracy

Extraction accuracy is measured by spot-checking a small sample of raw HTML files against the corresponding JSON records: title, date, author, body cleanliness, summary faithfulness, entity lists, and category. The extractor uses GLiNER zero-shot token classification with chunked inference to prevent false positives from generic nouns, industry sectors, or financial metrics, raising a hard error (`GLiNER not working`) if the neural model fails.

## Chunking And Embeddings

The pipeline implements **recursive character chunking** using LangChain's `RecursiveCharacterTextSplitter` with hierarchical separators (`["\n\n", "\n", ". ", " ", ""]`). The default chunk size is 1000 characters with 200 characters of overlap. This ensures paragraphs and sentences are preserved intact wherever possible, avoiding unnatural cuts across semantic boundaries while maintaining precise retrieval context. The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` because it is small, local, and fast enough for a coursework-sized corpus. ChromaDB is persisted under `data/chroma`.

## Observability & Debugging with LangSmith

LangSmith tracing is integrated across all four stages of the pipeline to monitor execution, track latency, and pinpoint failures:

- **Scraping (`scrape.py`)**: Traces discovery (`scrape_discover_urls`), raw downloads (`scrape_fetch_url`), HTTP status codes, and robots.txt checks.
- **Extracting (`extract.py`)**: Traces HTML cleaning/parsing (`extract_article`), schema validation, and pipeline execution (`extract_pipeline`).
- **Indexing (`index.py`)**: Traces batch chunking, document creation, and vector store upserts into ChromaDB (`index_pipeline`).
- **Querying (`query.py` & `src/news_rag/langchain_rag.py`)**: Traces query planning (`rag_query_pipeline`), vector retrieval (`chroma_retrieval`), lexical fallback (`lexical_retrieval`), and Gemini answer generation (`gemini_answer_generation`).


## Known Limitations

extract.py's runtime can be an issue with a larger number of files.A company or person can be named multiple times in different ways that appear on the site. Economic Times page structure can change; a changed structure may cause multiple failures or fewer than `MIN_ARTICLES` permitted URLs.

