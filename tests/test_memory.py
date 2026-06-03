"""
test_memory.py — Unit tests for SQLiteMemory
=============================================

Tests session creation, message insertion/reading, context window limits,
and token estimation in SQLite memory.
"""

import pytest
from src.memory import SQLiteMemory


@pytest.fixture
def memory_db(tmp_path) -> SQLiteMemory:
    """Initialize SQLite memory store with temporary path."""
    db_file = tmp_path / "test_chat_history.db"
    return SQLiteMemory(db_path=str(db_file))


def test_session_lifecycle(memory_db: SQLiteMemory):
    """Test session creation, retrieval, and deletion."""
    session_id = "test-session-xyz"
    
    # Create session
    memory_db.create_session(session_id)
    sessions = memory_db.get_all_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["summary"] is None

    # Delete session
    memory_db.delete_session(session_id)
    assert len(memory_db.get_all_sessions()) == 0


def test_messages_storage(memory_db: SQLiteMemory):
    """Test adding messages and reading them back with JSON metadata."""
    session_id = "test-session-abc"
    memory_db.add_message(session_id, "user", "Hello computer")
    
    meta = {"route": "RAG", "score": 0.9}
    memory_db.add_message(session_id, "assistant", "Hello human", metadata=meta)

    messages = memory_db.get_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello computer"
    
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["route"] == "RAG"


def test_context_window_estimation(memory_db: SQLiteMemory):
    """Test token estimation and context window limits."""
    session_id = "test-session-tokens"
    
    # Verify token approximation
    assert memory_db.estimate_tokens("Hello world") == 2
    
    # Store messages
    memory_db.add_message(session_id, "user", "Message A") # 2 tokens approx
    memory_db.add_message(session_id, "assistant", "Message B") # 2 tokens approx
    
    # Load context with high token limit
    msgs, summary = memory_db.get_context_window(session_id, max_tokens=100)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Message A"

    # Load context with restrictive limit to force truncation
    msgs, summary = memory_db.get_context_window(session_id, max_tokens=3)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Message B"  # only latest fits


def test_session_summaries(memory_db: SQLiteMemory):
    """Test summary updates and retrieval."""
    session_id = "test-session-summary"
    memory_db.create_session(session_id)
    
    memory_db.update_session_summary(session_id, "The user greeted the system.")
    summary = memory_db.get_session_summary(session_id)
    assert summary == "The user greeted the system."


def test_auto_summarize_trigger(memory_db: SQLiteMemory):
    """Test auto summarization when history exceeds token thresholds."""
    from unittest.mock import MagicMock
    session_id = "test-session-autosum"
    memory_db.create_session(session_id)

    # Add messages to exceed threshold
    memory_db.add_message(session_id, "user", "This is a very long message that contains a lot of text.")
    memory_db.add_message(session_id, "assistant", "Sure, I can help you summarize this conversation.")

    mock_lora = MagicMock()
    mock_lora.is_loaded = True
    mock_lora.generate.return_value = "Summary of conversation."

    memory_db.auto_summarize(session_id, mock_lora, max_history_tokens=5)
    
    summary = memory_db.get_session_summary(session_id)
    assert summary == "Summary of conversation."
