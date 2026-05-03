"""
main.py — FastAPI Application for the Hybrid RAG + LoRA System
================================================================

This is the entry-point that wires together:
  • The Semantic Router  (query → RAG or LoRA)
  • The RAG Pipeline     (factual retrieval)
  • The LoRA Pipeline    (style / reasoning generation)

Run locally:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /chat          — Main QA endpoint
    GET  /health        — Health check
    POST /ingest        — Ingest documents into the RAG pipeline
    GET  /router/scores — Debug: score a query without generating
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline
from src.router import Route, SemanticRouter
from src.lora_pipeline import LoRAPipeline

# ------------------------------------------------------------------ #
#  Logging Configuration                                               #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Global Pipeline Instances                                           #
# ------------------------------------------------------------------ #

# These are module-level singletons initialised at startup.
rag_pipeline: Optional[RAGPipeline] = None
lora_pipeline: Optional[LoRAPipeline] = None
semantic_router: Optional[SemanticRouter] = None


# ------------------------------------------------------------------ #
#  Application Lifespan (startup / shutdown)                           #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise pipelines once at server startup."""
    global rag_pipeline, lora_pipeline, semantic_router

    logger.info("=" * 60)
    logger.info("  Hybrid RAG + LoRA System — Starting Up")
    logger.info("=" * 60)

    # 1. Semantic Router (lightweight — always loads)
    semantic_router = SemanticRouter()

    # 2. RAG Pipeline
    rag_pipeline = RAGPipeline()

    # Auto-ingest if a data directory exists and has content.
    data_dir = os.environ.get("DATA_DIR", "./data")
    if os.path.isdir(data_dir):
        n_chunks = rag_pipeline.ingest_directory(data_dir)
        logger.info("Auto-ingested %d chunks from %s.", n_chunks, data_dir)

    # 3. LoRA Pipeline (heavy — load lazily on first request)
    #    Set PEFT_MODEL_PATH env var to point to trained adapter weights.
    peft_path = os.environ.get("PEFT_MODEL_PATH", None)
    base_model = os.environ.get(
        "BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1"
    )

    lora_pipeline = LoRAPipeline(
        base_model_name=base_model,
        peft_model_path=peft_path,
        load_in_4bit=True,
        device="auto",
    )
    # NOTE: We intentionally do NOT call lora_pipeline.load() here.
    # The 7B model takes 30-60 s to load; we defer to the first
    # LoRA-routed request so the server starts fast.
    logger.info(
        "LoRA pipeline configured (lazy load). Base model: %s", base_model
    )

    logger.info("=" * 60)
    logger.info("  System ready — accepting requests.")
    logger.info("=" * 60)

    yield  # ← application runs here

    # Shutdown
    logger.info("Shutting down…")


# ------------------------------------------------------------------ #
#  FastAPI App                                                         #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Hybrid RAG + LoRA QA System",
    description=(
        "An intelligent routing system that directs user queries to "
        "either a RAG pipeline (factual lookups) or a LoRA fine-tuned "
        "model (reasoning / style tasks)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#  Pydantic Schemas                                                    #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    """Payload for the /chat endpoint."""
    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="The user's natural-language question.",
    )
    force_route: Optional[str] = Field(
        None,
        description=(
            "Override the router.  Pass 'RAG' or 'LORA' to force a "
            "specific pipeline.  Useful for testing."
        ),
    )


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    query: str
    response: str
    routed_path: str = Field(
        ..., description="Which pipeline handled the query: 'RAG' or 'LORA'."
    )
    router_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-cluster similarity scores from the router.",
    )
    latency_ms: float = Field(
        ..., description="End-to-end processing time in milliseconds."
    )


class IngestRequest(BaseModel):
    """Payload for the /ingest endpoint."""
    texts: List[str] = Field(
        ..., min_length=1,
        description="Raw text strings to ingest into the RAG vector store.",
    )


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    rag_ready: bool
    lora_loaded: bool
    router_ready: bool


class RouterDebugRequest(BaseModel):
    """Payload for the /router/scores endpoint."""
    query: str


class RouterDebugResponse(BaseModel):
    """Response from the /router/scores endpoint."""
    query: str
    routed_to: str
    scores: Dict[str, float]


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return the readiness state of each subsystem."""
    return HealthResponse(
        status="ok",
        rag_ready=rag_pipeline is not None and rag_pipeline.vector_store is not None,
        lora_loaded=lora_pipeline is not None and lora_pipeline.is_loaded,
        router_ready=semantic_router is not None,
    )


@app.post("/chat", response_model=ChatResponse, tags=["QA"])
async def chat(request: ChatRequest):
    """Main QA endpoint.

    1. Routes the query via the Semantic Router.
    2. Dispatches to RAG or LoRA pipeline.
    3. Returns the answer + routing metadata.
    """
    t0 = time.perf_counter()

    if semantic_router is None:
        raise HTTPException(status_code=503, detail="Router not initialised.")

    # ---- Routing ------------------------------------------------- #
    if request.force_route:
        forced = request.force_route.upper()
        if forced not in ("RAG", "LORA"):
            raise HTTPException(
                status_code=400,
                detail="force_route must be 'RAG' or 'LORA'.",
            )
        route = Route(forced)
        scores: Dict[str, float] = {"forced": 1.0}
    else:
        route, scores = semantic_router.route(request.query)

    # ---- Dispatch ------------------------------------------------ #
    response_text: str

    if route == Route.RAG:
        if rag_pipeline is None or rag_pipeline.vector_store is None:
            response_text = (
                "RAG pipeline has no documents ingested.  Please POST "
                "to /ingest first, or place .txt files in the /data "
                "directory and restart the server."
            )
        else:
            # Retrieve relevant context
            context = rag_pipeline.retrieve_as_context(request.query)
            # In a full system you would feed this context into an
            # LLM to synthesize an answer.  For this scaffold we
            # return the raw retrieved context.
            response_text = (
                f"[RAG Retrieved Context]\n\n{context}\n\n"
                "──────────────────────────────────\n"
                "NOTE: In production, this context would be fed into "
                "an LLM (e.g., via LangChain's RetrievalQA chain) to "
                "generate a synthesized natural-language answer."
            )

    elif route == Route.LORA:
        if lora_pipeline is None:
            raise HTTPException(
                status_code=503, detail="LoRA pipeline not configured."
            )
        try:
            response_text = lora_pipeline.generate(request.query)
        except Exception as exc:
            logger.exception("LoRA generation failed.")
            response_text = (
                f"[LoRA Error] Generation failed: {exc}.  "
                "This is expected if the base model has not been "
                "downloaded yet.  Set BASE_MODEL_NAME to a smaller "
                "model for local testing."
            )

    else:
        raise HTTPException(status_code=500, detail="Unknown route.")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        query=request.query,
        response=response_text,
        routed_path=route.value,
        router_scores=scores,
        latency_ms=round(elapsed_ms, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_texts(request: IngestRequest):
    """Ingest raw text into the RAG vector store at runtime."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    n = rag_pipeline.ingest_texts(request.texts)
    return IngestResponse(
        chunks_indexed=n,
        message=f"Successfully indexed {n} chunk(s).",
    )


@app.post(
    "/router/scores",
    response_model=RouterDebugResponse,
    tags=["Debug"],
)
async def router_debug(request: RouterDebugRequest):
    """Score a query against the intent clusters without generating."""
    if semantic_router is None:
        raise HTTPException(status_code=503, detail="Router not ready.")

    route, scores = semantic_router.route(request.query)
    return RouterDebugResponse(
        query=request.query,
        routed_to=route.value,
        scores=scores,
    )


# ------------------------------------------------------------------ #
#  Entrypoint (for `python -m src.main`)                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
