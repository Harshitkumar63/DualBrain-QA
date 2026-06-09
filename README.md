# 🧬 DualBrain-QA: Hybrid RAG + LoRA Fine-Tuned LLM System

> A production-ready dual-pipeline AI system that intelligently routes user queries to either a **Retrieval-Augmented Generation (RAG)** pipeline for document-based answers or a **LoRA fine-tuned language model** for reasoning, summarization, and creative tasks — powered by a 6-intent semantic router.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-4A154B?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 🎯 Project Overview

Large Language Models are powerful, but no single inference strategy fits every query type:

| Query Type | Best Strategy | Why |
|---|---|---|
| *"What is our return policy?"* | **RAG** (retrieval) | Answer lives verbatim in company documents |
| *"Write a formal email about a delay"* | **LoRA fine-tuned model** | Requires stylistic reasoning, not document lookup |
| *"Summarize this meeting transcript"* | **LoRA fine-tuned model** | Compression and synthesis task |
| *"What does page 4 of the manual say?"* | **RAG** (retrieval) | Explicit document-based question |

This project builds **two specialized pipelines** with a **6-Intent Semantic Router** in front:

1. **RAG Pipeline** — Ingests documents (PDF, DOCX, TXT, MD), chunks them, embeds into **ChromaDB** (persistent), runs **hybrid retrieval** (dense + BM25 sparse search with Reciprocal Rank Fusion), and generates answers with source citations.
2. **LoRA Pipeline** — Loads a base LLM (Mistral-7B on GPU / TinyLlama on CPU) with **4-bit quantization** and **dynamic PEFT/LoRA adapter** switching for domain-adapted reasoning and generation.
3. **Semantic Router** — Classifies queries into 6 intents using embedding cosine similarity with softmax confidence scoring, ambiguity detection, and configurable fallback routing.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│              SEMANTIC ROUTER (6 Intents)          │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
│  │ factual  │ │reasoning │ │summarization │     │
│  │(→ RAG)   │ │(→ LoRA)  │ │(→ LoRA)      │     │
│  ├──────────┤ ├──────────┤ ├──────────────┤     │
│  │document  │ │  email   │ │conversational│     │
│  │  _qa     │ │generation│ │(→ LoRA)      │     │
│  │(→ RAG)   │ │(→ LoRA)  │ │              │     │
│  └──────────┘ └──────────┘ └──────────────┘     │
│    Softmax confidence + Ambiguity detection       │
└─────────────────┬────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
┌───────────────┐     ┌────────────────┐
│  RAG Pipeline │     │  LoRA Pipeline │
│               │     │                │
│ ChromaDB +    │     │ Mistral-7B /   │
│ BM25 Hybrid   │     │ TinyLlama +    │
│ Search (RRF)  │     │ PEFT Adapters  │
│               │     │ 4-bit Quant    │
│ Query Rewrite │     │                │
│ + Citations   │     │ Dynamic Adapter│
└───────┬───────┘     └───────┬────────┘
        └─────────┬───────────┘
                  ▼
        ┌──────────────────┐
        │   FastAPI /chat  │
        │   + SQLite Memory│
        │   + Rate Limiter │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Streamlit Chat  │
        │  Premium UI      │
        └──────────────────┘
```

---

## 🚀 Key Features

### Core Pipelines
- **Hybrid RAG Retrieval** — Dense (ChromaDB) + Sparse (BM25) search fused with Reciprocal Rank Fusion (RRF) for superior recall
- **Multi-Format Ingestion** — PDF, DOCX, TXT, and Markdown parsing with detailed metadata extraction
- **Query Rewriting** — LLM-powered query optimization before retrieval for better search results
- **Source Citations** — Every RAG answer includes clickable file + page citations
- **LoRA Pipeline** — Lazy-loaded model with 4-bit NF4 quantization and dynamic adapter switching at runtime
- **Auto CPU/GPU Fallback** — Automatically uses TinyLlama-1.1B on CPU if no GPU is detected

### Intelligent Routing
- **6-Intent Classification** — factual, reasoning, summarization, document_qa, email_generation, conversational
- **Softmax Confidence Scoring** — Calibrated probability distribution over all intents
- **Ambiguity Detection** — Warns when top-2 intents are too close in confidence
- **Configurable Fallback** — Defaults to RAG when confidence is below threshold

### Production Features
- **SQLite Conversational Memory** — Persistent multi-session chat history with automatic summarization
- **Token-Bucket Rate Limiter** — In-memory rate limiting per client IP
- **Structured JSON Logging** — Production-grade log format for monitoring
- **Global Exception Handling** — No unhandled crashes in production
- **Metrics Endpoint** — `/metrics` returns query counts, latency, error rates, and document stats
- **Health Checks** — `/health` validates all pipelines and database connectivity
- **36 Unit Tests** — Full test coverage across all modules

### Premium UI
- **Modern Chat Interface** — Route badges, confidence meters, latency chips
- **Interactive Citations** — Card-based citation display for RAG responses
- **Analytics Dashboard** — Live metrics, route distribution charts, intent matrix
- **File Upload** — Drag-and-drop document ingestion directly from the sidebar
- **Session Management** — Load, resume, and manage conversation sessions

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive chat interface with analytics dashboard |
| **Backend** | FastAPI + Uvicorn | Async REST API with rate limiting and health checks |
| **Orchestration** | LangChain | Document loading, splitting, embedding, retrieval |
| **Vector Store** | ChromaDB (persistent) | Dense similarity search with disk persistence |
| **Sparse Search** | BM25Okapi (rank-bm25) | Keyword-based sparse retrieval for hybrid fusion |
| **Embeddings** | `all-MiniLM-L6-v2` | Lightweight sentence embeddings (22M params) |
| **Base LLM** | Mistral-7B-Instruct / TinyLlama | Instruction-tuned causal language model |
| **Fine-Tuning** | HuggingFace PEFT | LoRA adapters with configurable rank/alpha |
| **Quantization** | bitsandbytes | 4-bit NF4 for efficient GPU inference |
| **Memory** | SQLite | Persistent chat history and session management |
| **Containerization** | Docker + Docker Compose | Multi-container deployment |

---

## 📂 Project Structure

```
DualBrain-QA/
├── src/
│   ├── __init__.py            # Package marker
│   ├── main.py                # FastAPI app (endpoints, lifespan, rate limiter)
│   ├── rag_pipeline.py        # Document parsing → chunking → hybrid retrieval → generation
│   ├── lora_pipeline.py       # Base model + LoRA adapter loading, 4-bit quantization
│   ├── router.py              # 6-intent semantic router (cosine similarity + softmax)
│   ├── memory.py              # SQLite conversational memory with auto-summarization
│   └── chroma_store.py        # ChromaDB persistent vector store manager
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # FastAPI endpoint integration tests (11 tests)
│   ├── test_rag.py            # RAG pipeline unit tests (11 tests)
│   ├── test_lora.py           # LoRA pipeline unit tests (5 tests)
│   ├── test_memory.py         # SQLite memory unit tests (5 tests)
│   └── test_router.py         # Semantic router unit tests (4 tests)
├── data/                      # Document storage + ChromaDB + SQLite database
│   ├── chromadb/              # Persistent vector store (auto-created)
│   └── chat_history.db        # SQLite session database (auto-created)
├── models/
│   ├── base_models/           # Cached HuggingFace model weights (auto-created)
│   └── adapters/              # Place LoRA adapter folders here
├── app.py                     # Streamlit premium chat UI
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project config + pytest settings
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # Multi-container orchestration
├── DEPLOYMENT.md              # Cloud deployment guide (Railway, Render, AWS)
├── .env.example               # Environment variable template
└── README.md                  # This file
```

---

## 💻 Installation & Setup

### Prerequisites

- **Python 3.10+** and **pip**
- **(Optional)** NVIDIA GPU with ≥ 6 GB VRAM for Mistral-7B inference
  - If no GPU is detected, the system automatically falls back to **TinyLlama-1.1B** on CPU

### Step 1 — Clone & Create Environment

```bash
git clone https://github.com/Harshitkumar63/DualBrain-QA.git
cd DualBrain-QA

python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\activate

# Windows (Git Bash):
source venv/Scripts/activate

# macOS/Linux:
source venv/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Important:** Make sure you are inside the project directory before running pip install. If you get `"No such file or directory: 'requirements.txt'"`, you're in the wrong folder — run `cd DualBrain-QA` first.

### Step 3 — Configure Environment (Optional)

```bash
# Copy the template
cp .env.example .env

# Edit .env to customize (all have sensible defaults)
```

### Step 4 — (Optional) Add Domain Documents

Place your documents in the `data/` directory. Supported formats: **PDF, DOCX, TXT, Markdown**.

```bash
# Example: create a sample document
echo "Our company was founded in 2020. We specialize in AI solutions." > data/company_info.txt
```

Documents are automatically ingested when the backend starts. You can also upload files via the Streamlit UI or the `/upload` API endpoint at any time.

---

## 🏃‍♂️ How to Run

You need **two terminals** (both from the project root directory):

### Terminal 1 — Start the FastAPI Backend

```bash
# Make sure venv is activated first!
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Hybrid System starting up...
INFO:     SQLite memory database initialized...
INFO:     Loading router embedding model...
INFO:     ChromaDB initialized...
INFO:     LoRA pipeline configured (lazy load setup).
```

Verify it's running:
```bash
curl http://localhost:8000/health
```

### Terminal 2 — Start the Streamlit Frontend

```bash
# Make sure venv is activated first!
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser.

### Quick API Tests (cURL)

```bash
# Factual query → Routes to RAG pipeline
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy?"}'

# Reasoning query → Routes to LoRA pipeline
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain microservices vs monoliths trade-offs"}'

# Force a specific route
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!", "force_route": "LORA"}'

# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@data/company_info.txt"

# Ingest raw text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Our product costs $99 per month."]}'

# Check system metrics
curl http://localhost:8000/metrics

# List all chat sessions
curl http://localhost:8000/sessions
```

---

## 🧪 Running Tests

```bash
# Run all 36 tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_api.py -v
python -m pytest tests/test_rag.py -v
python -m pytest tests/test_router.py -v
```

**Test coverage:**

| Module | Tests | What's Tested |
|---|---|---|
| `test_api.py` | 11 | All FastAPI endpoints, rate limiting, error handling |
| `test_rag.py` | 11 | File parsing, hybrid retrieval, RRF, citations, ingestion |
| `test_lora.py` | 5 | Device fallback, lazy loading, adapter switching, unload |
| `test_memory.py` | 5 | Sessions, messages, context window, auto-summarization |
| `test_router.py` | 4 | Intent classification, fallback, ambiguity, dynamic examples |

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start both backend + frontend containers
docker-compose up --build -d

# View logs
docker logs -f hybrid_llm_backend
docker logs -f hybrid_llm_frontend

# Stop
docker-compose down
```

- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:8501

> For cloud deployment (Railway, Render, AWS EC2), see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Directory for document ingestion on startup |
| `PEFT_MODEL_PATH` | `None` | Path to trained LoRA adapter weights |
| `BASE_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.1` | HuggingFace model ID for the base LLM |
| `CPU_FALLBACK_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Lighter model used when no GPU is available |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for the Streamlit frontend |
| `RATE_LIMIT_CAPACITY` | `10.0` | Maximum burst capacity for rate limiter |
| `RATE_LIMIT_RATE` | `1.0` | Token replenish rate (tokens/second) |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |

> **💡 Tip:** For CPU-only testing, set `BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0` in your `.env` file.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Pipeline status and database connectivity check |
| `GET` | `/metrics` | Query counts, latency, error rates, document stats |
| `POST` | `/chat` | Main query endpoint (routes through semantic router) |
| `POST` | `/ingest` | Ingest raw text strings into vector store |
| `POST` | `/upload` | Upload PDF/DOCX/TXT/MD files for ingestion |
| `GET` | `/sessions` | List all chat sessions from SQLite |
| `GET` | `/history/{session_id}` | Retrieve full message history for a session |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 🔀 Intent Routing Matrix

| Intent | Route | Example Queries |
|---|---|---|
| **factual** | 📚 RAG | "What is the capital of France?", "What are the pricing options?" |
| **document_qa** | 📚 RAG | "What does page 4 say?", "Find the refund policy in the PDF" |
| **reasoning** | ⚙️ LoRA | "Compare microservices vs monoliths", "Solve this step-by-step" |
| **summarization** | ⚙️ LoRA | "Summarize this article", "Give me the key takeaways" |
| **email_generation** | ⚙️ LoRA | "Write a follow-up email", "Draft a proposal email" |
| **conversational** | ⚙️ LoRA | "Hello!", "Tell me a joke", "How are you?" |

---

## 🔮 Future Improvements

- [ ] Fine-tuning script with `SFTTrainer` for custom LoRA adapter training
- [ ] Evaluation suite (MRR, Recall@K, ROUGE, BERTScore)
- [ ] Streaming response support (SSE/WebSocket)
- [ ] Multi-user authentication and API key management
- [ ] GPU-accelerated embeddings with CUDA
- [ ] Admin panel for adapter management and document CRUD

---

## 📄 License

Open-source under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using FastAPI · LangChain · ChromaDB · HuggingFace · Streamlit</sub>
</div>
