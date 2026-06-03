"""
test_api.py — API & Endpoint integration tests
==============================================

Tests the health, metrics, chat, upload, and history endpoints of
the FastAPI application using TestClient and mock pipelines.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_health_check_endpoint(client: TestClient):
    """Test health check returns expected status fields."""
    # Patch instances to represent operational system
    with patch("src.main.rag_pipeline") as mock_rag, \
         patch("src.main.lora_pipeline") as mock_lora, \
         patch("src.main.semantic_router") as mock_router, \
         patch("src.main.memory_db") as mock_mem:
        
        mock_rag.vector_store.get_document_count.return_value = 10
        mock_lora.is_loaded = True
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "unhealthy")
        assert "rag_ready" in data
        assert "lora_loaded" in data


def test_metrics_endpoint(client: TestClient):
    """Test metrics endpoint returns system stats."""
    with patch("src.main.memory_db") as mock_mem, \
         patch("src.main.rag_pipeline") as mock_rag:
        mock_mem.get_all_sessions.return_value = []
        mock_rag.vector_store.get_document_count.return_value = 0

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "average_latency_ms" in data
        assert "error_rate" in data


def test_ingest_endpoint(client: TestClient):
    """Test text ingestion endpoint passes documents to pipeline."""
    with patch("src.main.rag_pipeline") as mock_rag:
        mock_rag.ingest_texts.return_value = 2
        
        payload = {
            "texts": ["This is text 1", "This is text 2"],
            "metadatas": [{"src": "test"}, {"src": "test"}]
        }
        response = client.post("/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_indexed"] == 2
        assert "Successfully indexed" in data["message"]


def test_upload_file_endpoint(client: TestClient):
    """Test multipart file uploading endpoint processes document parses."""
    with patch("src.main.rag_pipeline") as mock_rag:
        mock_rag.ingest_file_data.return_value = 3
        
        files = {"file": ("test_file.txt", b"plain file upload content", "text/plain")}
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test_file.txt"
        assert data["chunks_indexed"] == 3


def test_chat_endpoint_rag_route(client: TestClient):
    """Test chat query routed to RAG pipeline."""
    with patch("src.main.semantic_router") as mock_router, \
         patch("src.main.rag_pipeline") as mock_rag, \
         patch("src.main.memory_db") as mock_mem, \
         patch("src.main.lora_pipeline") as mock_lora:
        
        # Router returns RAG route
        mock_router.route.return_value = ("RAG", {"intent": "factual", "confidence": 0.95})
        mock_rag.retrieve_and_generate.return_value = {
            "answer": "France's capital is Paris.",
            "sources": [{"file": "france.txt", "page": 1}]
        }
        mock_mem.get_context_window.return_value = ([], None)

        payload = {"query": "What is the capital of France?", "session_id": "test_session_123"}
        response = client.post("/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["routed_path"] == "RAG"
        assert "Paris" in data["response"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file"] == "france.txt"


def test_chat_endpoint_lora_route(client: TestClient):
    """Test chat query routed to LoRA pipeline."""
    with patch("src.main.semantic_router") as mock_router, \
         patch("src.main.lora_pipeline") as mock_lora, \
         patch("src.main.memory_db") as mock_mem, \
         patch("src.main.rag_pipeline") as mock_rag:
        
        # Router returns LORA route
        mock_router.route.return_value = ("LORA", {"intent": "reasoning", "confidence": 0.88})
        mock_lora.generate.return_value = "Python supports OOP and functional styles."
        mock_mem.get_context_window.return_value = ([], None)

        payload = {"query": "Tell me about Python styles", "session_id": "test_session_123"}
        response = client.post("/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["routed_path"] == "LORA"
        assert "functional" in data["response"]


def test_chat_validation_limits(client: TestClient):
    """Test that empty queries fail validation checks."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422  # Unprocessable Entity


def test_list_all_sessions(client: TestClient):
    """Test retrieving list of all SQLite sessions."""
    with patch("src.main.memory_db") as mock_mem:
        mock_mem.get_all_sessions.return_value = [{"session_id": "s1", "created_at": "now", "summary": None}]
        
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == "s1"


def test_get_session_history(client: TestClient):
    """Test retrieving message history for a specific session."""
    with patch("src.main.memory_db") as mock_mem:
        mock_mem.get_messages.return_value = [{"role": "user", "content": "hello"}]
        
        response = client.get("/history/s1")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "s1"
        assert len(data["messages"]) == 1


def test_rate_limiter_exceeded(client: TestClient):
    """Test that rate limiter triggers Too Many Requests on high frequency."""
    with patch("src.main.rate_limiter.check_rate_limit", return_value=False):
        response = client.post("/chat", json={"query": "test"})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


def test_global_exception_handler(client: TestClient):
    """Test global error handler maps exceptions to 500 status."""
    with patch("src.main.semantic_router") as mock_router:
        # Cause router to raise a raw exception
        mock_router.route.side_effect = RuntimeError("Db connection failed")
        
        # Patch other variables to bypass prior checks
        with patch("src.main.rag_pipeline"), \
             patch("src.main.lora_pipeline"), \
             patch("src.main.memory_db"):
             
            response = client.post("/chat", json={"query": "test"})
            assert response.status_code == 500
            assert "internal server error" in response.json()["detail"].lower()
