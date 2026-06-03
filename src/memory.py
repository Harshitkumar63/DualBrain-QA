"""
memory.py — SQLite-Backed Conversational Memory System
======================================================

Upgrade features:
  1. SQLite persistence for multi-session chat histories.
  2. Context window management (token/character limits).
  3. Automatic conversation summarization.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Default Database path
DEFAULT_DB_PATH = "./data/chat_history.db"


class SQLiteMemory:
    """Manages chat sessions, messages, context windowing, and summaries in SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        
        # Ensure data directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the sessions and messages tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()
        logger.info("SQLite memory database initialized at %s.", self.db_path)

    # -------------------------------------------------------------- #
    #  Session Management                                              #
    # -------------------------------------------------------------- #

    def create_session(self, session_id: str) -> None:
        """Create a new session if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?);",
                (session_id,),
            )
            conn.commit()

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve all sessions with metadata and their summaries."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT session_id, created_at, summary FROM sessions ORDER BY created_at DESC;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
            conn.commit()
        logger.info("Session %s deleted.", session_id)

    # -------------------------------------------------------------- #
    #  Message Log                                                     #
    # -------------------------------------------------------------- #

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the session history."""
        self.create_session(session_id)
        meta_json = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?);
                """,
                (session_id, role, content, meta_json),
            )
            conn.commit()

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a given session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                msg = dict(row)
                if msg["metadata"]:
                    try:
                        msg["metadata"] = json.loads(msg["metadata"])
                    except json.JSONDecodeError:
                        msg["metadata"] = {}
                else:
                    msg["metadata"] = {}
                messages.append(msg)
            return messages

    # -------------------------------------------------------------- #
    #  Context Window Management                                       #
    # -------------------------------------------------------------- #

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get the current running summary for a session."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT summary FROM sessions WHERE session_id = ?;", (session_id,)).fetchone()
            return row["summary"] if row else None

    def update_session_summary(self, session_id: str, summary: str) -> None:
        """Set or update the conversation summary for a session."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE session_id = ?;",
                (summary, session_id),
            )
            conn.commit()

    def estimate_tokens(self, text: str) -> int:
        """Quick rough estimation of token count (chars / 4)."""
        return max(1, len(text) // 4)

    def get_context_window(
        self,
        session_id: str,
        max_tokens: int = 1500,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Retrieve chat messages fitting within max_tokens limits, prepended with any summary.

        Returns
        -------
        messages : list[dict]
            List of role/content message dicts.
        summary : str | None
            The session summary if exists.
        """
        all_msgs = self.get_messages(session_id)
        summary = self.get_session_summary(session_id)

        # Build list backwards to prioritize recent messages
        window_msgs: List[Dict[str, str]] = []
        token_count = 0
        if summary:
            token_count += self.estimate_tokens(summary) + 10 # Buffer

        for msg in reversed(all_msgs):
            msg_tokens = self.estimate_tokens(msg["content"])
            if token_count + msg_tokens > max_tokens:
                break
            
            window_msgs.insert(0, {"role": msg["role"], "content": msg["content"]})
            token_count += msg_tokens

        return window_msgs, summary

    # -------------------------------------------------------------- #
    #  Summarization Trigger                                           #
    # -------------------------------------------------------------- #

    def auto_summarize(self, session_id: str, lora_pipeline: Optional[Any] = None, max_history_tokens: int = 1200) -> None:
        """Trigger summarization if message history exceeds max_history_tokens."""
        if not lora_pipeline or not lora_pipeline.is_loaded:
            return

        all_msgs = self.get_messages(session_id)
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in all_msgs)

        if total_tokens < max_history_tokens:
            return

        logger.info("Conversation history size (%d estimated tokens) triggers auto-summarization...", total_tokens)

        # Build conversation log for the LLM
        convo_log = []
        for msg in all_msgs:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            convo_log.append(f"{role_label}: {msg['content']}")
        
        convo_text = "\n".join(convo_log)

        # Retrieve existing summary to append
        existing_summary = self.get_session_summary(session_id)
        existing_clause = f"Previous Summary: {existing_summary}\n\n" if existing_summary else ""

        prompt = (
            "[INST] Task: Write a concise, bullet-pointed summary of the following conversation. "
            "Focus on key facts, preferences, and action items discussed. Do not use pleasantries.\n\n"
            f"{existing_clause}"
            f"Conversation log:\n{convo_text} [/INST]\n"
            "Summary:"
        )

        try:
            summary = lora_pipeline.generate(prompt, max_new_tokens=150, temperature=0.3)
            clean_summary = summary.strip()
            self.update_session_summary(session_id, clean_summary)
            logger.info("Session %s successfully summarized.", session_id)
        except Exception as e:
            logger.error("Auto-summarization failed: %s", e)
