"""
test_rag.py — Unit tests for the RAGPipeline
============================================

Tests document loading, splitting, hybrid retriever (RRF),
query rewriter, and source citation formatting.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def rag_pipeline(tmp_path) -> RAGPipeline:
    """Initialize RAG pipeline in a temporary folder to isolate test databases."""
    db_dir = tmp_path / "chromadb"
    pipeline = RAGPipeline(
        persist_dir=str(db_dir),
        collection_name="test_collection",
        chunk_size=100,
        chunk_overlap=10
    )
    return pipeline


def test_txt_and_md_parsing(rag_pipeline: RAGPipeline, tmp_path):
    """Test parsing of plain text and markdown documents from disk."""
    # Create temp files
    txt_file = tmp_path / "test_doc.txt"
    txt_file.write_text("Hello from the text file.", encoding="utf-8")
    
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Markdown Title\nHello from markdown.", encoding="utf-8")

    # Ingest
    rag_pipeline.ingest_directory(str(tmp_path))

    # Retrieve
    docs = rag_pipeline.hybrid_retrieve("Hello text", top_k=2)
    assert len(docs) > 0
    contents = [d.page_content for d in docs]
    assert any("text file" in c for c in contents)


def test_pdf_parsing_mocked(rag_pipeline: RAGPipeline):
    """Mock pypdf reader to test PDF ingestion flow."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is text extracted from a PDF page."
    
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("src.rag_pipeline.PdfReader", return_value=mock_reader):
        parsed = rag_pipeline.parse_file_bytes(b"dummy pdf bytes", "sample.pdf")
        assert len(parsed) == 1
        assert parsed[0].page_content == "This is text extracted from a PDF page."
        assert parsed[0].metadata["filename"] == "sample.pdf"
        assert parsed[0].metadata["page_number"] == 1
        assert parsed[0].metadata["document_type"] == "pdf"


def test_docx_parsing_mocked(rag_pipeline: RAGPipeline):
    """Mock python-docx to test Word document ingestion flow."""
    mock_para = MagicMock()
    mock_para.text = "This is text from a paragraph inside DOCX."
    
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]

    with patch("src.rag_pipeline.DocxDocument", return_value=mock_doc):
        parsed = rag_pipeline.parse_file_bytes(b"dummy docx bytes", "sample.docx")
        assert len(parsed) == 1
        assert parsed[0].page_content == "This is text from a paragraph inside DOCX."
        assert parsed[0].metadata["filename"] == "sample.docx"
        assert parsed[0].metadata["document_type"] == "docx"


def test_hybrid_rrf_retrieval(rag_pipeline: RAGPipeline):
    """Test reciprocal rank fusion ranking of dense and sparse results."""
    # Ingest text strings
    texts = [
        "Python is a popular programming language.",
        "Java is used for corporate backend systems.",
        "ChromaDB stores high-dimensional vector embeddings."
    ]
    rag_pipeline.ingest_texts(texts)

    # Search query
    fused_docs = rag_pipeline.hybrid_retrieve("Python programming", top_k=2)
    assert len(fused_docs) <= 2
    assert "Python" in fused_docs[0].page_content


def test_source_citation_formatting(rag_pipeline: RAGPipeline):
    """Test that source citation structures are generated correctly."""
    texts = ["React is a frontend framework."]
    metadatas = [{"filename": "react_guide.md", "page_number": 3}]
    rag_pipeline.ingest_texts(texts, metadatas=metadatas)

    result = rag_pipeline.retrieve_and_generate("React framework")
    assert "answer" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["file"] == "react_guide.md"
    assert result["sources"][0]["page"] == 3


def test_query_rewriting_mocked(rag_pipeline: RAGPipeline):
    """Test query rewriter calling generator pipeline."""
    mock_generator = MagicMock()
    mock_generator.is_loaded = True
    mock_generator.generate.return_value = "rewritten search query"

    rewritten = rag_pipeline.rewrite_query("original input question", mock_generator)
    assert rewritten == "rewritten search query"


def test_chroma_store_management(rag_pipeline: RAGPipeline):
    """Test direct operations on the ChromaStore layer."""
    store = rag_pipeline.vector_store
    
    # Check count is 0 initially or after reset
    store.reset_store()
    assert store.get_document_count() == 0
    
    # Ingest document
    store.add_documents([Document(page_content="Store text", metadata={"filename": "test.txt", "page_number": 1})])
    assert store.get_document_count() == 1
    
    # Clean up DB directory
    store.delete_db_directory()


def test_unsupported_file_parsing(rag_pipeline: RAGPipeline, tmp_path):
    """Test that parsing files with unsupported extensions returns an empty list."""
    unsupported_file = tmp_path / "test_doc.xyz"
    unsupported_file.write_text("Hello from the raw file.", encoding="utf-8")
    
    docs = rag_pipeline.parse_file(unsupported_file)
    assert len(docs) == 0

    # Also test byte parsing for unsupported extensions
    docs_bytes = rag_pipeline.parse_file_bytes(b"some raw bytes", "sample.xyz")
    assert len(docs_bytes) == 0


def test_ingest_file_data_bytes(rag_pipeline: RAGPipeline):
    """Test directly ingesting raw file bytes."""
    chunks = rag_pipeline.ingest_file_data(b"some sample raw text bytes inside file", "raw.txt")
    assert chunks > 0
    assert rag_pipeline.vector_store.get_document_count() > 0


def test_ingest_empty_directory(rag_pipeline: RAGPipeline, tmp_path):
    """Test ingesting an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    chunks = rag_pipeline.ingest_directory(str(empty_dir))
    assert chunks == 0


def test_ingest_nonexistent_directory(rag_pipeline: RAGPipeline):
    """Test ingesting a directory that does not exist."""
    chunks = rag_pipeline.ingest_directory("./nonexistent_directory_abc_xyz")
    assert chunks == 0
