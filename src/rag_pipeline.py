"""
rag_pipeline.py — RAG (Retrieval-Augmented Generation) Pipeline
================================================================

Responsibilities:
  1. Load raw text documents from disk.
  2. Split them into semantically meaningful chunks.
  3. Embed chunks using a HuggingFace sentence-transformer model.
  4. Store embeddings in an in-memory FAISS vector store.
  5. At query time, retrieve the top-K most relevant chunks.

Design Trade-offs:
  • We use RecursiveCharacterTextSplitter because it respects natural
    text boundaries (paragraphs → sentences → words) instead of
    blindly slicing at a fixed character count.
  • FAISS is chosen over ChromaDB here for zero-config, in-memory
    simplicity.  For production workloads with persistence and
    metadata filtering, swap to ChromaDB or Pinecone.
  • The embedding model (all-MiniLM-L6-v2) is a lightweight 22M-param
    model optimized for speed.  For higher accuracy, consider
    bge-large-en-v1.5 or instructor-xl at the cost of latency.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration defaults                                             #
# ------------------------------------------------------------------ #

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 4


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Usage::

        rag = RAGPipeline()
        rag.ingest_directory("./data")       # load & embed documents
        results = rag.retrieve("What is X?") # retrieve relevant chunks
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.top_k = top_k

        # ----------------------------------------------------------
        # 1. Embedding model — runs locally on CPU by default.
        #    Set `model_kwargs={"device": "cuda"}` for GPU inference.
        # ----------------------------------------------------------
        logger.info("Loading embedding model: %s", embedding_model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ----------------------------------------------------------
        # 2. Text splitter — recursive strategy preserves semantics.
        # ----------------------------------------------------------
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # The FAISS index is lazily initialized on first ingestion.
        self.vector_store: Optional[FAISS] = None

    # -------------------------------------------------------------- #
    #  Document Ingestion                                              #
    # -------------------------------------------------------------- #

    def _load_text_files(self, directory: str) -> List[Document]:
        """Recursively load all .txt files from *directory*."""
        docs: List[Document] = []
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {directory}")

        for fpath in sorted(dir_path.rglob("*.txt")):
            text = fpath.read_text(encoding="utf-8", errors="replace")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(fpath)},
                )
            )
            logger.debug("Loaded %s (%d chars)", fpath.name, len(text))

        if not docs:
            logger.warning("No .txt files found in %s", directory)
        return docs

    def ingest_directory(self, directory: str) -> int:
        """Load, chunk, embed, and index all .txt files in *directory*.

        Returns the total number of chunks indexed.
        """
        raw_docs = self._load_text_files(directory)
        if not raw_docs:
            logger.warning("Nothing to ingest — directory is empty.")
            return 0

        chunks = self.text_splitter.split_documents(raw_docs)
        logger.info(
            "Split %d document(s) into %d chunks.", len(raw_docs), len(chunks)
        )

        # Build (or extend) the FAISS index.
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        logger.info("FAISS index now contains %d vectors.", self.vector_store.index.ntotal)
        return len(chunks)

    def ingest_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> int:
        """Ingest raw text strings directly (useful for testing).

        Returns the total number of chunks indexed.
        """
        docs = [
            Document(page_content=t, metadata=(metadatas[i] if metadatas else {}))
            for i, t in enumerate(texts)
        ]
        chunks = self.text_splitter.split_documents(docs)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        logger.info("Ingested %d text(s) → %d chunks.", len(texts), len(chunks))
        return len(chunks)

    # -------------------------------------------------------------- #
    #  Retrieval                                                       #
    # -------------------------------------------------------------- #

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Return the top-K most relevant chunks for *query*.

        Each result is a dict with keys ``content``, ``score``, and
        ``metadata``.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No documents ingested. Call `ingest_directory()` or "
                "`ingest_texts()` before retrieving."
            )

        k = top_k or self.top_k
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=k
        )

        output: List[dict] = []
        for doc, score in results_with_scores:
            output.append(
                {
                    "content": doc.page_content,
                    # FAISS returns L2 distance by default; lower = better.
                    # With normalized embeddings (cosine), distance ∈ [0, 2].
                    "score": float(score),
                    "metadata": doc.metadata,
                }
            )
        return output

    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Convenience wrapper: returns retrieved chunks as a single
        formatted context string, ready to be injected into an LLM prompt.
        """
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant context found."

        context_parts: List[str] = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            context_parts.append(
                f"[Chunk {i} | source: {source} | score: {r['score']:.4f}]\n"
                f"{r['content']}"
            )
        return "\n\n---\n\n".join(context_parts)
