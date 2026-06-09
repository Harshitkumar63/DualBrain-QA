"""
app.py — Streamlit Frontend for the Hybrid RAG + LoRA System
==============================================================

An advanced, premium-tier user interface:
  1. Modern chat room with custom styled route badges and confidence meters.
  2. Side-by-side or tabbed Analytics Dashboard retrieving live backend metrics.
  3. Interactive, card-based Citation Display.
  4. Sidebar File Upload utility integrating with the backend /upload endpoint.
  5. SQLite-backed Session History sidebar to load/resume previous conversations.
"""

import os
import streamlit as st
import requests
import time
from typing import Optional, List, Dict, Any

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
INGEST_ENDPOINT = f"{BACKEND_URL}/ingest"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload"
HISTORY_ENDPOINT = f"{BACKEND_URL}/history"
SESSIONS_ENDPOINT = f"{BACKEND_URL}/sessions"
METRICS_ENDPOINT = f"{BACKEND_URL}/metrics"

# ------------------------------------------------------------------ #
#  Page Config & Styling                                               #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Hybrid LLM System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global Styles ───────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
    }

    /* ── Route Badges ────────────────────────────────────────── */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-top: 8px;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        color: #ffffff !important;
    }
    .badge-rag {
        background: linear-gradient(135deg, #0284c7, #4f46e5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-lora {
        background: linear-gradient(135deg, #ea580c, #dc2626);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-intent {
        background: rgba(255, 255, 255, 0.15);
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* ── Latency chip ────────────────────────────────────────── */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* ── Citation Cards ──────────────────────────────────────── */
    .citation-header {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 12px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .citation-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .citation-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.78rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .citation-card span {
        font-weight: 600;
        color: #38bdf8;
    }

    /* ── Progress Indicators ─────────────────────────────────── */
    .score-container {
        margin-top: 10px;
        padding: 12px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* ── Sidebar Styles ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* ── Status Dot ──────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .status-offline { background: #ef4444; box-shadow: 0 0 8px #ef4444; }

    /* ── Title Banner ────────────────────────────────────────── */
    .header-banner {
        text-align: center;
        background: radial-gradient(circle at center, rgba(79, 70, 229, 0.15) 0%, transparent 60%);
        padding: 2.5rem 1rem 1.5rem 1rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .header-banner h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Session State Initialization                                        #
# ------------------------------------------------------------------ #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = ""

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat Room"

# ------------------------------------------------------------------ #
#  API Communication Helpers                                           #
# ------------------------------------------------------------------ #

def check_backend_health() -> Optional[Dict[str, Any]]:
    """Get server health status."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=2)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def get_backend_metrics() -> Optional[Dict[str, Any]]:
    """Fetch live system metrics."""
    try:
        r = requests.get(METRICS_ENDPOINT, timeout=2)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def fetch_sessions() -> List[Dict[str, Any]]:
    """Retrieve chat sessions from SQLite database."""
    try:
        r = requests.get(SESSIONS_ENDPOINT, timeout=2)
        if r.status_code == 200:
            return r.json().get("sessions", [])
    except requests.RequestException:
        pass
    return []


def load_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Fetch previous chat log for session_id."""
    try:
        r = requests.get(f"{HISTORY_ENDPOINT}/{session_id}", timeout=3)
        if r.status_code == 200:
            return r.json().get("messages", [])
    except requests.RequestException:
        pass
    return []


def send_chat_query(
    query: str,
    session_id: str,
    force_route: Optional[str] = None,
    adapter_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Send query to the chat API endpoint."""
    payload = {
        "query": query,
        "session_id": session_id
    }
    if force_route and force_route != "Auto (Router decides)":
        payload["force_route"] = "RAG" if "RAG" in force_route else "LORA"
    if adapter_name and adapter_name != "base":
        payload["adapter_name"] = adapter_name

    try:
        r = requests.post(CHAT_ENDPOINT, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Error {r.status_code}: {r.text}")
    except requests.RequestException as e:
        st.error(f"Could not connect to FastAPI backend: {e}")
    return None


def upload_document(file_name: str, file_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Upload PDF/DOCX/TXT/MD document to backend."""
    try:
        files = {"file": (file_name, file_bytes)}
        r = requests.post(UPLOAD_ENDPOINT, files=files, timeout=45)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Upload failed: {r.text}")
    except requests.RequestException as e:
        st.error(f"Upload request error: {e}")
    return None

# ------------------------------------------------------------------ #
#  Sidebar Layout                                                      #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 🧬 Hybrid LLM Control")
    st.markdown("---")

    # Backend Connection Indicator
    health = check_backend_health()
    if health:
        st.markdown(
            '<span class="status-dot status-online"></span> **Backend Connected**',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("📚 RAG: Ready" if health.get("rag_ready") else "📚 RAG: Empty")
        with c2:
            st.caption("⚙️ LoRA: Loaded" if health.get("lora_loaded") else "⚙️ LoRA: Offline")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span> **Backend Disconnected**',
            unsafe_allow_html=True,
        )
        st.error("FastAPI server must be running on localhost:8000")

    st.markdown("---")

    # Dynamic File Uploader
    st.markdown("### 📤 Ingest Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown files directly to persistent database:",
        type=["pdf", "docx", "txt", "md"],
        help="Parsed text will be chunked, embedded and loaded in ChromaDB + BM25 keyword index."
    )
    if uploaded_file is not None:
        with st.spinner("Parsing and indexing document..."):
            res = upload_document(uploaded_file.name, uploaded_file.getvalue())
            if res:
                st.success(f"Ingested! Chunks: {res.get('chunks_indexed')}")
                # Rerun to refresh metrics
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    # Routing Settings
    st.markdown("### ⚙️ Inference Settings")
    force_opt = st.selectbox(
        "Forced Pipeline Routing",
        options=["Auto (Router decides)", "Force RAG", "Force LoRA"],
        index=0
    )
    
    # Adapter Settings
    adapter_opt = st.selectbox(
        "Select Active LoRA Adapter",
        options=["base"],  # Adapters are auto-discovered from models/adapters/
        index=0,
        help="Place LoRA adapter folders in models/adapters/ — they'll be auto-detected on server startup."
    )

    st.markdown("---")

    # SQLite Session History list
    st.markdown("### 💬 Session History")
    sessions_list = fetch_sessions()
    
    if sessions_list:
        selected_session = st.selectbox(
            "Resume conversation session:",
            options=[s["session_id"] for s in sessions_list],
            format_func=lambda x: f"Session: {x[:8]}...",
        )
        
        # Load session button
        if st.button("Load Selected Session", use_container_width=True):
            st.session_state.session_id = selected_session
            msgs = load_session_messages(selected_session)
            # Reformat messages
            formatted = []
            for m in msgs:
                formatted.append({
                    "role": m["role"],
                    "content": m["content"],
                    "metadata": m.get("metadata", {})
                })
            st.session_state.messages = formatted
            st.rerun()
            
    # New session creation
    if st.button("➕ Start New Session", use_container_width=True):
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Current History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ #
#  Main Panel Content                                                 #
# ------------------------------------------------------------------ #

# App Title Banner
st.markdown("""
<div class="header-banner">
    <h1>🧬 Production-Ready Hybrid LLM System</h1>
    <p>A high-performance system combining ChromaDB RAG and fine-tuned LoRA adapters with an Intent Classification Router.</p>
</div>
""", unsafe_allow_html=True)

# Main Tabbed Interface
tab_chat, tab_analytics = st.tabs(["💬 Dynamic Chat Room", "📊 System Analytics & Evaluation"])

# ------------------------------------------------------------------ #
#  Tab 1: Chat Room                                                   #
# ------------------------------------------------------------------ #

with tab_chat:
    # Render chat messages from history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
            # Render route indicators and citations
            if m["role"] == "assistant" and "metadata" in m and m["metadata"]:
                meta = m["metadata"]
                
                # Badges row
                routed = meta.get("routed_path")
                intent = meta.get("router_scores", {}).get("intent")
                conf = meta.get("router_scores", {}).get("confidence", 0.0)
                latency = meta.get("latency_ms")
                
                if routed:
                    badge_class = "badge-rag" if routed == "RAG" else "badge-lora"
                    badge_text = "📚 RAG pipeline" if routed == "RAG" else "⚙️ LoRA Fine-Tuned Model"
                    
                    badge_html = f"""
                    <div class="badge-container">
                        <span class="badge {badge_class}">{badge_text}</span>
                        {f'<span class="badge badge-intent">Intent: {intent} (conf: {conf:.2f})</span>' if intent else ''}
                        {f'<span class="chip">⚡ {latency:.0f} ms</span>' if latency else ''}
                    </div>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                
                # Source Citations
                sources = meta.get("sources", [])
                if sources:
                    st.markdown('<div class="citation-header">Retrieved Citations</div>', unsafe_allow_html=True)
                    cit_html = '<div class="citation-grid">'
                    for s in sources:
                        cit_html += f"""
                        <div class="citation-card">
                            📄 File: <span>{s.get('file')}</span> | Page: <span>{s.get('page')}</span>
                        </div>
                        """
                    cit_html += '</div>'
                    st.markdown(cit_html, unsafe_allow_html=True)

    # Chat User input
    if prompt := st.chat_input("Ask the dual-pipeline system a question..."):
        # Display user question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call Backend
        with st.chat_message("assistant"):
            with st.spinner("Routing query and generating response..."):
                response = send_chat_query(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    force_route=force_opt,
                    adapter_name=adapter_opt
                )
            
            if response:
                ans_text = response.get("response", "")
                routed_path = response.get("routed_path", "")
                scores = response.get("router_scores", {})
                latency = response.get("latency_ms", 0.0)
                session_id = response.get("session_id", "")
                sources = response.get("sources", [])

                # Update session id in state
                st.session_state.session_id = session_id

                st.markdown(ans_text)

                # Route Badge & Latency
                badge_class = "badge-rag" if routed_path == "RAG" else "badge-lora"
                badge_text = "📚 RAG pipeline" if routed_path == "RAG" else "⚙️ LoRA Fine-Tuned Model"
                intent = scores.get("intent")
                conf = scores.get("confidence", 0.0)

                badge_html = f"""
                <div class="badge-container">
                    <span class="badge {badge_class}">{badge_text}</span>
                    {f'<span class="badge badge-intent">Intent: {intent} (conf: {conf:.2f})</span>' if intent else ''}
                    <span class="chip">⚡ {latency:.0f} ms</span>
                </div>
                """
                st.markdown(badge_html, unsafe_allow_html=True)

                # Render Citations
                if sources:
                    st.markdown('<div class="citation-header">Retrieved Citations</div>', unsafe_allow_html=True)
                    cit_html = '<div class="citation-grid">'
                    for s in sources:
                        cit_html += f"""
                        <div class="citation-card">
                            📄 File: <span>{s.get('file')}</span> | Page: <span>{s.get('page')}</span>
                        </div>
                        """
                    cit_html += '</div>'
                    st.markdown(cit_html, unsafe_allow_html=True)

                # Save history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans_text,
                    "metadata": {
                        "routed_path": routed_path,
                        "router_scores": scores,
                        "latency_ms": latency,
                        "sources": sources
                    }
                })
            else:
                st.error("Failed to generate response. Check FastAPI console logs.")

# ------------------------------------------------------------------ #
#  Tab 2: Analytics Dashboard                                         #
# ------------------------------------------------------------------ #

with tab_analytics:
    st.markdown("### 📊 Live Evaluation & Metrics Dashboard")
    st.markdown("Aggregated latency, routing decisions, database contents, and error rates.")

    metrics = get_backend_metrics()
    if metrics:
        # Metrics columns
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric(
                label="Total Queries",
                value=metrics.get("total_queries", 0)
            )
        with c2:
            st.metric(
                label="Average Latency",
                value=f"{metrics.get('average_latency_ms', 0.0)} ms"
            )
        with c3:
            st.metric(
                label="Error Rate",
                value=f"{metrics.get('error_rate', 0.0) * 100:.2f} %"
            )
        with c4:
            st.metric(
                label="ChromaDB Indexed Documents",
                value=metrics.get("document_count", 0)
            )

        st.markdown("---")

        # Routing breakdown charts
        route_data = metrics.get("route_breakdown", {})
        if route_data:
            st.markdown("#### 🔀 Route Distribution Breakdown")
            # Render a horizontal bar chart of decisions
            st.bar_chart(route_data)

        st.markdown("---")
        
        # Details of the 6 Intents
        st.markdown("#### 🎯 Active Intents & Routing Matrix")
        st.markdown("""
        | Intent Name | Subsystem Target | Routing Rationale |
        |---|---|---|
        | **factual** | 📚 RAG Pipeline | Factual data contained verbatim in documents |
        | **document_qa** | 📚 RAG Pipeline | Explicit QA query about context files |
        | **reasoning** | ⚙️ LoRA Fine-Tuned Model | Analytical or logic problem-solving |
        | **summarization** | ⚙️ LoRA Fine-Tuned Model | Document summarization & compression |
        | **email_generation** | ⚙️ LoRA Fine-Tuned Model | Structured formal output writing |
        | **conversational** | ⚙️ LoRA Fine-Tuned Model | Greeting, feedback and banter |
        """)
        
    else:
        st.warning("Could not fetch metrics. Is backend connected?")
