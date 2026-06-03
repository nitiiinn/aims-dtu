# Agentic RAG for Scholarly Q&A — Ablation Study

An agentic Retrieval-Augmented Generation system that answers research questions over a curated arXiv corpus (2024–2026). The system retrieves evidence from a local ChromaDB vector store and generates grounded, cited answers using a local LLM via Ollama.

This project runs a controlled ablation study across five agent configurations to measure the impact of individual components (Planner, Reflector, Verifier) on answer quality and citation accuracy.

## Architecture

```
Question → Planner → Retriever → Reflector → Synthesizer → Verifier → Answer
             (optional)            (optional)                (optional)
```

**Modules** (defined in `src/modules.py`):

| Module | Role |
|---|---|
| **Planner** | Decomposes complex questions into sub-queries |
| **Retriever** | Embeds queries with BGE-Base and searches ChromaDB |
| **Reflector** | Reviews retrieved chunks and decides if more retrieval is needed |
| **Synthesizer** | Drafts the final answer with inline citations |
| **Verifier** | Audits the draft against source chunks for hallucination |

**Two execution modes:**
- `src/loop.py` — Imperative while-loop with configurable max iterations
- `src/agent.py` — LangGraph `StateGraph` implementation

## Ablation Configurations

| Config | Planner | Reflector | Verifier |
|---|---|---|---|
| `baseline` | OFF | OFF | OFF |
| `full_agent` | ON | ON | ON |
| `no_planner` | OFF | ON | ON |
| `no_reflector` | ON | OFF | ON |
| `no_verifier` | ON | ON | OFF |

## Project Structure

```
├── main.py                         # Entry point — runs all 5 ablation configs
├── requirements.txt
├── .env                            # API keys (GROQ_API_KEY, etc.)
│
├── src/                            # Agent core
│   ├── modules.py                  # Planner, Retriever, Reflector, Synthesizer, Verifier
│   ├── loop.py                     # Imperative agent loop
│   └── agent.py                    # LangGraph agent
│
├── scraper/                        # Data pipeline
│   ├── data_collection.py          # Fetch paper metadata from arXiv API
│   ├── data_download.py            # Download PDFs
│   ├── text_parser.py              # Extract text from PDFs (PyMuPDF)
│   └── reranker.py                 # Score and filter corpus by relevance
│
├── index/                          # Indexing pipeline
│   ├── chunker.py                  # Overlapping text chunking
│   └── vector_store.py             # Build ChromaDB index with BGE-Base
│
├── eval/                           # Evaluation
│   ├── questions.jsonl             # 30 research questions (factoid/comparative/survey)
│   ├── local_ground_truth.jsonl    # Reference answers (generated via Groq Llama 3.3 70B)
│   ├── evaluate.py                 # Automated grading (citation P/R/F1 + LLM judge)
│   ├── seed_ground_truth.py        # Generate ground truth from Groq + ChromaDB chunks
│   └── SUBMISSION_FORMAT.md
│
├── predictions/                    # Model outputs (one per config)
│   ├── baseline.jsonl
│   ├── full_agent.jsonl
│   ├── no_planner.jsonl
│   ├── no_reflector.jsonl
│   └── no_verifier.jsonl
│
├── index/chroma_db/                # Persistent vector store
└── corpus_pdfs/                    # Downloaded arXiv PDFs
```

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) with `qwen2.5-coder:7b` pulled locally
- Groq API key (for ground truth generation only)

### Installation

```bash
git clone <repo-url>
cd aims-dtu
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Pull the local model:
```bash
ollama pull qwen2.5-coder:7b
```

## Reproduce from Scratch

After cloning and installing, run these commands in order to reproduce the full pipeline end-to-end.

> **Note:** Ensure Ollama is running (`ollama serve`) before Steps 5-7.

### Step 1: Collect Paper Metadata from arXiv

```bash
python scraper/data_collection.py
```
Fetches ~3000 candidate papers from arXiv API. Saves to `eval/corpus_metadata.json`.

### Step 2: Filter to Top 800 Papers

```bash
python scraper/reranker.py
```
Scores papers by keyword relevance and keeps top 800. Saves to `eval/corpus_metadata_filtered.json`.

### Step 3: Download PDFs

```bash
python scraper/data_download.py
```
Downloads PDFs to `corpus_pdfs/`. Supports resume (skips already downloaded files).

### Step 4: Parse and Index

```bash
python scraper/text_parser.py         # Extract text from PDFs → eval/parsed_texts.jsonl
python index/chunker.py               # Chunk into overlapping segments → eval/corpus_chunks.jsonl
python index/vector_store.py          # Build ChromaDB vector index → index/chroma_db/
```

### Step 5: Generate Ground Truth (requires Groq API key)

```bash
python eval/seed_ground_truth.py
```
Uses Llama 3.3 70B via Groq to create reference answers from retrieved chunks. Saves to `eval/local_ground_truth.jsonl`. Rate-limited to ~30 RPM.

### Step 6: Run Ablation Study

```bash
python main.py
```
Runs all 5 configurations x 30 questions using the local Ollama model. Outputs:
- `predictions/baseline.jsonl`
- `predictions/full_agent.jsonl`
- `predictions/no_planner.jsonl`
- `predictions/no_reflector.jsonl`
- `predictions/no_verifier.jsonl`

### Step 7: Evaluate

```bash
python eval/evaluate.py
```
Scores all predictions against the ground truth using citation metrics + LLM judge. Outputs:
- Markdown results table in terminal
- PDF report at `eval/evaluation_report.pdf`

### Step 8: Generate Project Report (optional)

```bash
python generate_report.py
```
Creates the full project report PDF at `report.pdf`.

## Evaluation Metrics

| Metric | Method | Scale |
|---|---|---|
| **Citation Precision** | Set overlap (predicted vs ground truth) | 0–1 |
| **Citation Recall** | Set overlap (ground truth vs predicted) | 0–1 |
| **Citation F1** | Harmonic mean of P and R | 0–1 |
| **Accuracy** | LLM judge (Ollama) — semantic alignment with reference | 1–5 |
| **Faithfulness** | LLM judge (Ollama) — grounded in retrieved chunks | 0 or 1 |

## Prediction Format

Each file in `predictions/` is newline-delimited JSON:

```json
{
  "id": "q01",
  "answer": "<system's answer>",
  "cited_papers": ["2504.19413", "2502.12110"]
}
```

**Length guidance:**
- `factoid`: 1–3 sentences
- `comparative`: 100–300 words
- `survey`: 250–600 words

## Tech Stack

- **Embeddings**: BAAI/bge-base-en-v1.5 (sentence-transformers)
- **Vector Store**: ChromaDB (persistent, HNSW indexing)
- **LLM**: qwen2.5-coder:7b via Ollama (local inference)
- **Agent Framework**: LangGraph
- **Ground Truth**: Llama 3.3 70B via Groq API
- **PDF Parsing**: PyMuPDF
