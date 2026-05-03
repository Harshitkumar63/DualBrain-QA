"""
app.py — Streamlit Frontend for the Hybrid RAG + LoRA System
==============================================================

A modern chat interface that communicates with the FastAPI backend,
displays routing decisions visually, and maintains conversation history.

Run:
    streamlit run app.py
"""

import streamlit as st
import requests
import time
from typing import Optional

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

BACKEND_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
INGEST_ENDPOINT = f"{BACKEND_URL}/ingest"
ROUTER_DEBUG_ENDPOINT = f"{BACKEND_URL}/router/scores"

# ------------------------------------------------------------------ #
#  Page Config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Hybrid LLM System — RAG + LoRA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS                                                          #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ────────────────────────────────────── */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* ── Route badges ──────────────────────────────────────── */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-top: 8px;
    }
    .route-rag {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: #ffffff;
    }
    .route-lora {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: #ffffff;
    }

    /* ── Latency chip ──────────────────────────────────────── */
    .latency-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(100, 100, 100, 0.15);
        color: #888;
        margin-left: 8px;
    }

    /* ── Score bar ─────────────────────────────────────────── */
    .score-bar-container {
        margin-top: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(100, 100, 100, 0.06);
        font-size: 0.75rem;
    }
    .score-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
        font-weight: 500;
        color: #666;
    }
    .score-bar-track {
        width: 100%;
        height: 6px;
        border-radius: 3px;
        background: rgba(100, 100, 100, 0.12);
        margin-bottom: 8px;
        overflow: hidden;
    }
    .score-bar-fill-rag {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
    }
    .score-bar-fill-lora {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }

    /* ── Sidebar styling ───────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown code {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #a5b4fc !important;
    }

    /* ── Status dot ────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .status-offline { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* ── Header ────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #0ea5e9, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .app-header p {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session State Initialization                                        #
# ------------------------------------------------------------------ #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_scores" not in st.session_state:
    st.session_state.show_scores = True


# ------------------------------------------------------------------ #
#  Helper Functions                                                    #
# ------------------------------------------------------------------ #

def check_backend_health() -> Optional[dict]:
    """Ping the backend health endpoint."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    return None


def send_query(query: str, force_route: Optional[str] = None) -> Optional[dict]:
    """Send a query to the /chat endpoint and return the response."""
    payload = {"query": query}
    if force_route:
        payload["force_route"] = force_route
    try:
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Backend returned status {resp.status_code}: {resp.text}")
            return None
    except requests.ConnectionError:
        st.error("❌ Cannot connect to the backend. Is the FastAPI server running?")
        return None
    except requests.Timeout:
        st.error("⏱️ Request timed out. The model may be loading for the first time.")
        return None


def render_route_badge(routed_path: str, latency_ms: float) -> str:
    """Return HTML for the route badge and latency chip."""
    if routed_path == "RAG":
        badge = '<span class="route-badge route-rag">📚 Answered via RAG Pipeline</span>'
    else:
        badge = '<span class="route-badge route-lora">⚙️ Answered via LoRA Fine-Tuned Model</span>'

    latency = f'<span class="latency-chip">⚡ {latency_ms:.0f} ms</span>'
    return badge + latency


def render_score_bars(scores: dict) -> str:
    """Return HTML for the router similarity score visualization."""
    if not scores or "forced" in scores:
        return ""

    html = '<div class="score-bar-container"><strong style="font-size:0.72rem;color:#888;">Router Confidence Scores</strong>'
    for cluster_name, score in scores.items():
        pct = max(0, min(100, score * 100))
        fill_class = "score-bar-fill-rag" if "Factual" in cluster_name else "score-bar-fill-lora"
        html += f"""
        <div class="score-bar-label">
            <span>{cluster_name}</span>
            <span>{score:.4f}</span>
        </div>
        <div class="score-bar-track">
            <div class="{fill_class}" style="width:{pct}%"></div>
        </div>
        """
    html += "</div>"
    return html


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 🧬 Hybrid LLM System")
    st.markdown("---")

    # Backend status
    health = check_backend_health()
    if health:
        st.markdown(
            '<span class="status-dot status-online"></span> **Backend Online**',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            rag_status = "✅" if health.get("rag_ready") else "⬜"
            st.caption(f"{rag_status} RAG Ready")
        with col2:
            lora_status = "✅" if health.get("lora_loaded") else "⏳"
            st.caption(f"{lora_status} LoRA Loaded")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span> **Backend Offline**',
            unsafe_allow_html=True,
        )
        st.caption("Start the FastAPI server first.")

    st.markdown("---")

    # Architecture explanation
    st.markdown("### 🏗️ How It Works")
    st.markdown("""
    Every query goes through a **3-stage pipeline**:

    **① Semantic Router**
    Your query is embedded and compared against two intent clusters 
    using cosine similarity.

    **② Pipeline Dispatch**
    - **📚 RAG Pipeline** — Factual lookups from your documents 
      via FAISS vector search.
    - **⚙️ LoRA Pipeline** — Reasoning and style tasks via a 
      fine-tuned language model.

    **③ Response + Metadata**
    You see the answer along with *which pipeline* handled it 
    and the confidence scores.
    """)

    st.markdown("---")

    # Controls
    st.markdown("### ⚙️ Settings")
    st.session_state.show_scores = st.toggle(
        "Show router scores", value=st.session_state.show_scores
    )

    force_option = st.selectbox(
        "Force route (override router)",
        options=["Auto (Router decides)", "Force RAG", "Force LoRA"],
        index=0,
    )

    st.markdown("---")

    # Quick ingest
    st.markdown("### 📥 Quick Ingest")
    ingest_text = st.text_area(
        "Paste text to add to the RAG knowledge base:",
        height=100,
        placeholder="e.g. Our company was founded in 2020...",
    )
    if st.button("Ingest into RAG", use_container_width=True):
        if ingest_text.strip():
            try:
                resp = requests.post(
                    INGEST_ENDPOINT,
                    json={"texts": [ingest_text.strip()]},
                    timeout=30,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ Indexed {result['chunks_indexed']} chunk(s)")
                else:
                    st.error(f"Error: {resp.text}")
            except requests.ConnectionError:
                st.error("Backend not reachable.")
        else:
            st.warning("Enter some text first.")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div style='text-align:center;margin-top:1rem;'>"
        "<span style='font-size:0.7rem;color:#64748b;'>"
        "Built with FastAPI + LangChain + PEFT"
        "</span></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
#  Main Chat Area                                                      #
# ------------------------------------------------------------------ #

# Header
st.markdown("""
<div class="app-header">
    <h1>🧬 Hybrid RAG + LoRA Chat</h1>
    <p>Ask anything — the semantic router will pick the best pipeline for your query.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show route badge and scores for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            badge_html = render_route_badge(
                meta.get("routed_path", ""),
                meta.get("latency_ms", 0),
            )
            st.markdown(badge_html, unsafe_allow_html=True)

            if st.session_state.show_scores and meta.get("router_scores"):
                score_html = render_score_bars(meta["router_scores"])
                if score_html:
                    st.markdown(score_html, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine force_route
    force_route = None
    if force_option == "Force RAG":
        force_route = "RAG"
    elif force_option == "Force LoRA":
        force_route = "LORA"

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Routing & generating…"):
            result = send_query(prompt, force_route=force_route)

        if result:
            response_text = result.get("response", "No response received.")
            routed_path = result.get("routed_path", "UNKNOWN")
            latency_ms = result.get("latency_ms", 0)
            router_scores = result.get("router_scores", {})

            st.markdown(response_text)

            # Route badge
            badge_html = render_route_badge(routed_path, latency_ms)
            st.markdown(badge_html, unsafe_allow_html=True)

            # Score bars
            if st.session_state.show_scores and router_scores:
                score_html = render_score_bars(router_scores)
                if score_html:
                    st.markdown(score_html, unsafe_allow_html=True)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metadata": {
                    "routed_path": routed_path,
                    "latency_ms": latency_ms,
                    "router_scores": router_scores,
                },
            })
        else:
            fallback = "⚠️ Could not get a response. Please check the backend."
            st.markdown(fallback)
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback,
            })
