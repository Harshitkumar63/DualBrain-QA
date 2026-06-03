"""
main.py — FastAPI Application for the Hybrid RAG + LoRA System
================================================================

This module acts as the production-ready API layer:
  1. Asynchronous endpoints using `asyncio.to_thread` for CPU/GPU operations.
  2. Structured logging (JSON format compatible).
  3. In-memory token-bucket rate limiter.
  4. Global exception handler middleware.
  5. Metrics tracking (/metrics) and DB session histories (/history/{session_id}).
  6. File upload endpoint (/upload) supporting PDFs, DOCX, TXT, and MD.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline
from src.router import Route, SemanticRouter
from src.lora_pipeline import LoRAPipeline
from src.memory import SQLiteMemory

# ------------------------------------------------------------------ #
#  Logging Configuration                                               #
# ------------------------------------------------------------------ #

# Structured logging setup
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------ #
#  Global Pipeline & Storage Instances                                 #
# ------------------------------------------------------------------ #

rag_pipeline: Optional[RAGPipeline] = None
lora_pipeline: Optional[LoRAPipeline] = None
semantic_router: Optional[SemanticRouter] = None
memory_db: Optional[SQLiteMemory] = None

# Metrics tracking
METRICS = {
    "total_queries": 0,
    "rag_queries": 0,
    "lora_queries": 0,
    "total_latency_ms": 0.0,
    "error_queries": 0,
}

# ------------------------------------------------------------------ #
#  Rate Limiter                                                       #
# ------------------------------------------------------------------ #

class TokenBucketLimiter:
    """In-memory rate limiter using the Token Bucket algorithm."""
    
    def __init__(self, rate_per_second: float, capacity: float) -> None:
        self.rate = rate_per_second
        self.capacity = capacity
        self.buckets: Dict[str, Tuple[float, float]] = defaultdict(lambda: (capacity, time.time()))

    def check_rate_limit(self, client_ip: str) -> bool:
        tokens, last_update = self.buckets[client_ip]
        now = time.time()
        # Refill tokens based on elapsed time
        refilled = tokens + (now - last_update) * self.rate
        tokens = min(self.capacity, refilled)
        
        if tokens >= 1.0:
            self.buckets[client_ip] = (tokens - 1.0, now)
            return True
        
        self.buckets[client_ip] = (tokens, now)
        return False

# Limit to 1 request per second with a burst capacity of 10 requests
rate_limiter = TokenBucketLimiter(rate_per_second=1.0, capacity=10.0)

def rate_limit_dependency(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check_rate_limit(client_ip):
        logging.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

# ------------------------------------------------------------------ #
#  FastAPI Lifespan (Startup/Shutdown)                                 #
# ------------------------------------------------------------------ #

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines and databases once at server startup."""
    global rag_pipeline, lora_pipeline, semantic_router, memory_db

    logging.info("Hybrid System starting up...")

    # 1. Memory Store (SQLite)
    memory_db = SQLiteMemory()

    # 2. Semantic Router
    semantic_router = SemanticRouter()

    # 3. RAG Pipeline
    rag_pipeline = RAGPipeline()
    data_dir = os.environ.get("DATA_DIR", "./data")
    if os.path.isdir(data_dir):
        # Scan and ingest files asynchronously to prevent blocking startup
        n_chunks = await asyncio.to_thread(rag_pipeline.ingest_directory, data_dir)
        logging.info("Auto-ingested %d chunks from %s.", n_chunks, data_dir)

    # 4. LoRA Pipeline
    peft_path = os.environ.get("PEFT_MODEL_PATH", None)
    base_model = os.environ.get("BASE_MODEL_NAME", None)
    
    lora_pipeline = LoRAPipeline(
        base_model_name=base_model,
        load_in_4bit=True,
        device="auto",
    )
    logging.info("LoRA pipeline configured (lazy load setup).")

    yield

    # Shutdown
    logging.info("System shutting down...")
    if lora_pipeline:
        lora_pipeline.unload()

# ------------------------------------------------------------------ #
#  FastAPI App Setup                                                   #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Hybrid RAG + LoRA QA System",
    description="Production-ready API for routing user queries dynamically.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Swap for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Global Error Handler Middleware                                     #
# ------------------------------------------------------------------ #

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error("Unhandled exception occurred: %s", exc, exc_info=True)
    METRICS["error_queries"] += 1
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check logs for details."}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logging.warning("HTTP Exception (%d): %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# ------------------------------------------------------------------ #
#  Pydantic Schemas                                                    #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question.")
    session_id: Optional[str] = Field(None, description="SQLite history session ID.")
    force_route: Optional[str] = Field(None, description="Override router: 'RAG' or 'LORA'.")
    adapter_name: Optional[str] = Field(None, description="LoRA adapter target.")


class SourceItem(BaseModel):
    file: str
    page: int


class ChatResponse(BaseModel):
    query: str
    response: str
    routed_path: str
    router_scores: Dict[str, Any]
    latency_ms: float
    session_id: str
    sources: List[SourceItem] = []


class IngestRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    metadatas: Optional[List[Dict[str, Any]]] = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    lora_loaded: bool
    router_ready: bool
    database_connected: bool


class MetricsResponse(BaseModel):
    total_queries: int
    route_breakdown: Dict[str, int]
    average_latency_ms: float
    error_rate: float
    active_sessions: int
    document_count: int


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return status of all pipelines and persistence layers."""
    db_ok = False
    try:
        if memory_db:
            with memory_db._get_connection() as conn:
                conn.execute("SELECT 1;")
                db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "unhealthy",
        rag_ready=rag_pipeline is not None and rag_pipeline.vector_store.get_document_count() > 0,
        lora_loaded=lora_pipeline is not None and lora_pipeline.is_loaded,
        router_ready=semantic_router is not None,
        database_connected=db_ok,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """Observability endpoint returning aggregate system performance data."""
    active_sessions = len(memory_db.get_all_sessions()) if memory_db else 0
    doc_count = rag_pipeline.vector_store.get_document_count() if rag_pipeline else 0
    
    total = METRICS["total_queries"]
    avg_lat = (METRICS["total_latency_ms"] / total) if total > 0 else 0.0
    err_rate = (METRICS["error_queries"] / total) if total > 0 else 0.0

    return MetricsResponse(
        total_queries=total,
        route_breakdown={
            "RAG": METRICS["rag_queries"],
            "LORA": METRICS["lora_queries"]
        },
        average_latency_ms=round(avg_lat, 2),
        error_rate=round(err_rate, 4),
        active_sessions=active_sessions,
        document_count=doc_count
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit_dependency)], tags=["QA"])
async def chat(request: ChatRequest):
    """Main query entry point. Performs intent classification, retrieves contexts, and generates response."""
    t0 = time.perf_counter()
    METRICS["total_queries"] += 1

    if semantic_router is None or memory_db is None or lora_pipeline is None or rag_pipeline is None:
        METRICS["error_queries"] += 1
        raise HTTPException(status_code=503, detail="Pipelines not initialised.")

    # Manage/Retrieve session
    session_id = request.session_id or str(uuid.uuid4())
    memory_db.create_session(session_id)

    # ---- Intent Routing (Async Thread) --------------------------- #
    route: Route
    scores: Dict[str, Any]

    if request.force_route:
        forced = request.force_route.upper()
        if forced not in ("RAG", "LORA"):
            METRICS["error_queries"] += 1
            raise HTTPException(status_code=400, detail="force_route must be 'RAG' or 'LORA'.")
        route = Route(forced)
        scores = {"forced": 1.0, "route": forced, "intent": "forced", "confidence": 1.0}
    else:
        route, scores = await asyncio.to_thread(semantic_router.route, request.query)

    # ---- Dispatch Pipeline (Async Thread) ------------------------ #
    response_text: str
    sources: List[Dict[str, Any]] = []

    # Inject chat history into query context
    history_msgs, _ = memory_db.get_context_window(session_id, max_tokens=1200)
    history_context = ""
    if history_msgs:
        # Convert list of messages to prompt context
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])

    if route == Route.RAG:
        METRICS["rag_queries"] += 1
        # Retrieve context + Synthesize Answer
        # We rewrite & generate inside thread to avoid event loop locks
        rag_payload = await asyncio.to_thread(
            rag_pipeline.retrieve_and_generate,
            query=request.query,
            lora_pipeline=lora_pipeline,
            filter_dict=None,
        )
        response_text = rag_payload["answer"]
        sources = rag_payload["sources"]

    elif route == Route.LORA:
        METRICS["lora_queries"] += 1
        # Dynamic adapter selection
        target_adapter = request.adapter_name or scores.get("intent", "base")
        
        prompt_with_history = request.query
        if history_context:
            prompt_with_history = (
                f"Below is the conversation history:\n{history_context}\n\n"
                f"User: {request.query}"
            )
            
        try:
            response_text = await asyncio.to_thread(
                lora_pipeline.generate,
                prompt=prompt_with_history,
                adapter_name=target_adapter
            )
        except Exception as exc:
            logging.error("LoRA generation failed: %s", exc)
            METRICS["error_queries"] += 1
            response_text = f"LoRA model execution failed: {exc}."

    else:
        # Defensive fallback — should never happen with the current Route enum
        METRICS["error_queries"] += 1
        response_text = "Unknown route encountered. Please contact support."
        logging.error("Chat endpoint reached unknown route: %s", route)

    # Save to history database
    memory_db.add_message(session_id, "user", request.query)
    
    # Store scores & routing metadata inside SQLite logs
    meta_log = {
        "routed_path": str(route),
        "router_scores": scores,
        "sources": sources
    }
    memory_db.add_message(session_id, "assistant", response_text, meta_log)

    # Auto summarize history if it gets too long
    # We do this asynchronously to keep chat response fast
    async def _safe_summarize():
        try:
            await asyncio.to_thread(memory_db.auto_summarize, session_id, lora_pipeline)
        except Exception as e:
            logging.error("Background auto-summarization failed for session %s: %s", session_id, e)

    asyncio.create_task(_safe_summarize())

    elapsed_ms = (time.perf_counter() - t0) * 1000
    METRICS["total_latency_ms"] += elapsed_ms

    return ChatResponse(
        query=request.query,
        response=response_text,
        routed_path=str(route),
        router_scores=scores,
        latency_ms=round(elapsed_ms, 2),
        session_id=session_id,
        sources=[SourceItem(**s) for s in sources],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_texts(request: IngestRequest):
    """Ingest raw texts directly into the vector store."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    n = await asyncio.to_thread(
        rag_pipeline.ingest_texts,
        texts=request.texts,
        metadatas=request.metadatas
    )
    return IngestResponse(
        chunks_indexed=n,
        message=f"Successfully indexed {n} chunk(s).",
    )


@app.post("/upload", response_model=UploadResponse, tags=["RAG"])
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF, DOCX, TXT, or MD documents and parse/index them directly."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    content = await file.read()
    filename = file.filename or "uploaded_file"

    try:
        n_chunks = await asyncio.to_thread(
            rag_pipeline.ingest_file_data,
            file_bytes=content,
            filename=filename
        )
        if n_chunks == 0:
            raise HTTPException(status_code=400, detail=f"File {filename} could not be parsed or contains no text.")
            
        logging.info("Uploaded file %s successfully parsed and indexed into %d chunks.", filename, n_chunks)
        return UploadResponse(
            filename=filename,
            chunks_indexed=n_chunks,
            message=f"Successfully indexed {n_chunks} chunk(s) from {filename}."
        )
    except Exception as e:
        logging.error("Failed to process uploaded file %s: %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file upload: {str(e)}")


@app.get("/history/{session_id}", tags=["History"])
async def get_session_history(session_id: str):
    """Retrieve all messages and metadata for the given session."""
    if memory_db is None:
        raise HTTPException(status_code=503, detail="Database memory is not initialized.")
        
    messages = memory_db.get_messages(session_id)
    return {
        "session_id": session_id,
        "messages": messages
    }


@app.get("/sessions", tags=["History"])
async def list_all_sessions():
    """Retrieve all session records from SQLite."""
    if memory_db is None:
        raise HTTPException(status_code=503, detail="Database memory is not initialized.")
        
    sessions = memory_db.get_all_sessions()
    return {"sessions": sessions}
