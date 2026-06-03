"""
chroma_store.py — ChromaDB Vector Store Manager
================================================

This module implements the migration layer from FAISS to ChromaDB, providing:
  1. Persistent storage on disk (re-loads automatically on startup).
  2. Support for metadata filtering.
  3. Collection management (resetting, counting items).
  4. Integration with standard LangChain Embedding models.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages a persistent ChromaDB database for RAG document storage."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: Embeddings,
    ) -> None:
        """Initialize ChromaDB store.

        Parameters
        ----------
        persist_directory : str
            Path on disk where ChromaDB collections will be stored.
        collection_name : str
            Name of the collection to read/write.
        embedding_model : Embeddings
            LangChain-compatible embedding model.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embedding_model

        logger.info(
            "Initializing ChromaDB at %s with collection '%s'",
            persist_directory,
            collection_name,
        )

        # Ensure directory structure exists
        os.makedirs(persist_directory, exist_ok=True)

        self._vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the persistent collection."""
        if not documents:
            return
        self._vector_store.add_documents(documents)
        logger.info("Added %d documents to ChromaDB collection.", len(documents))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search and return documents with their L2/cosine scores."""
        results = self._vector_store.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform similarity search and return documents."""
        return self._vector_store.similarity_search(query, k=k, filter=filter_dict)

    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            collection = self._vector_store._client.get_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            logger.error("Failed to get document count: %s", e)
            return 0

    def reset_store(self) -> None:
        """Delete current collection and re-create it empty."""
        logger.warning("Resetting ChromaDB collection: %s", self.collection_name)
        try:
            # Delete collection using client
            self._vector_store._client.delete_collection(self.collection_name)
        except Exception as e:
            logger.debug("Collection did not exist or failed to delete: %s", e)

        # Re-initialize
        self._vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        logger.info("ChromaDB collection reset successfully.")

    def delete_db_directory(self) -> None:
        """Completely wipe database directory from disk. Use with caution."""
        logger.warning("Deleting database directory: %s", self.persist_directory)
        try:
            self._vector_store = None
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
        except Exception as e:
            logger.error("Failed to delete ChromaDB directory: %s", e)
