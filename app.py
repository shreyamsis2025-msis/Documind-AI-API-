import streamlit as st
import os
import time

from ingestion.loader import load_file
from ingestion.splitter import split_docs
from vectordb.faiss_store import save_db, load_db
from rag.pipeline import RAGChat
from config import DATA_PATH
from streamlit_mic_recorder import mic_recorder
from voice.whisper_local import speech_to_text
from streamlit_pdf_viewer import pdf_viewer

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Documind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a2332;
}

/* Source cards */
.source-card {
    background: #1e2535;
    border: 1px solid #2e3a50;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
}
.source-card strong { color: #4f8ef7; }
.source-snippet {
    color: #8892a4;
    font-size: 0.8rem;
    margin-top: 4px;
    line-height: 1.4;
}

/* Transcription badge */
.voice-badge {
    background: #1a3a2a;
    border: 1px solid #2d6e4e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.83rem;
    color: #4ecca3;
    display: inline-block;
    margin-bottom: 8px;
}

/* File list items */
.file-chip {
    background: #1e2535;
    border: 1px solid #2e3a50;
    border-radius: 6px;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.83rem;
    color: #c9d1e0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Metric cards */
.stat-row {
    display: flex;
    gap: 10px;
    margin: 8px 0;
}
.stat-box {
    flex: 1;
    background: #1e2535;
    border: 1px solid #2e3a50;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
}
.stat-num { font-size: 1.3rem; font-weight: 700; color: #4f8ef7; }
.stat-label { font-size: 0.72rem; color: #8892a4; }

/* Thinking animation */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.thinking { animation: pulse 1.2s ease-in-out infinite; color: #4f8ef7; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, default in {
    "messages": [],
    "open_pdf": None,
    "last_sources": [],
    "last_voice_text": "",
    "processed_files": set(),
    "total_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs("static_docs", exist_ok=True)

# ─────────────────────────────────────────────
# Groq API key — load from env or let user enter it in sidebar
# ─────────────────────────────────────────────
from config import GROQ_API_KEY as _cfg_key
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = _cfg_key

# ─────────────────────────────────────────────
# Load chatbot (stored in session_state so reload_db works correctly)
# ─────────────────────────────────────────────
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChat()

chatbot = st.session_state.chatbot

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.jpg", use_container_width=True)
    # st.markdown("## 🧠 Documind AI")
    # st.caption("Chat with your documents · Powered by Groq")
    st.divider()

    # # ── Groq API Key ──
    # st.markdown("### 🔑 Groq API Key")
    # key_input = st.text_input(
    #     "Enter your free Groq API key",
    #     value=st.session_state.groq_api_key,
    #     type="password",
    #     placeholder="gsk_...",
    #     label_visibility="collapsed"
    # )
    # if key_input != st.session_state.groq_api_key:
    #     st.session_state.groq_api_key = key_input
    #     # Reinitialise chatbot with new key
    #     os.environ["GROQ_API_KEY"] = key_input
    #     st.session_state.chatbot = RAGChat()
    #     chatbot = st.session_state.chatbot
    #     st.success("✅ API key updated!")

    # if not st.session_state.groq_api_key:
    #     st.warning("⚠️ Get a **free** key at [console.groq.com](https://console.groq.com)")

    # st.divider()
    
    # ── Upload ──
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, CSV, XLSX, TXT",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            progress = st.progress(0, text="Processing…")
            all_docs = []

            for i, file in enumerate(new_files):
                progress.progress((i) / len(new_files), text=f"Loading {file.name}…")
                path = os.path.join(DATA_PATH, file.name)
                static_path = os.path.join("static_docs", file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                with open(static_path, "wb") as f:
                    file.seek(0)
                    f.write(file.getbuffer())

                docs = load_file(path)
                all_docs.extend(docs)
                st.session_state.processed_files.add(file.name)

            if all_docs:
                progress.progress(0.85, text="Building vector index…")
                chunks = split_docs(all_docs)
                save_db(chunks)
                st.session_state.total_chunks += len(chunks)
                chatbot.reload_db()
                progress.progress(1.0, text="Done!")
                time.sleep(0.4)
                progress.empty()
                st.success(f"✅ {len(new_files)} file(s) indexed ({len(chunks)} chunks)")

    # ── Indexed files ──
    st.divider()
    st.markdown("### 📋 Indexed Files")
    existing = sorted(os.listdir(DATA_PATH)) if os.path.exists(DATA_PATH) else []
    if existing:
        ext_icon = {"pdf": "📄", "docx": "📝", "csv": "📊", "xlsx": "📊", "txt": "📃"}
        for fname in existing:
            ext = fname.rsplit(".", 1)[-1].lower()
            icon = ext_icon.get(ext, "📁")
            st.markdown(
                f'<div class="file-chip">{icon} {fname}</div>',
                unsafe_allow_html=True
            )

        # Stats
        if st.session_state.total_chunks > 0:
            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-box"><div class="stat-num">{len(existing)}</div>'
                f'<div class="stat-label">Files</div></div>'
                f'<div class="stat-box"><div class="stat-num">{st.session_state.total_chunks}</div>'
                f'<div class="stat-label">Chunks</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No files yet. Upload some documents above.")

    # ── Clear chat ──
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA — split into chat + viewer
# ─────────────────────────────────────────────
chat_col, viewer_col = st.columns([3, 2]) if st.session_state.open_pdf else (st.columns([1])[0], None)

with chat_col:
    # ── Chat header image ──
    col1, col2 = st.columns([0.08, 1])
    with col1:
        st.image("assets/robo.jpg", width=90)
    with col2:
        st.markdown("<h2 style='margin-top: 10px;'>DocumindAI</h2>", unsafe_allow_html=True)

    # ── Message history ──
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📚 Sources used", expanded=False):
                        for src in msg["sources"]:
                            pg_str = f" · Page {src['page']}" if src.get("page") else ""
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>📄 {src["file"]}{pg_str}</strong>'
                                f'<div class="source-snippet">{src["snippet"][:250]}…</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

    # ── Input row ──
    st.divider()
    input_col, mic_col = st.columns([9, 1])

    with input_col:
        text_query = st.chat_input("Ask anything about your documents…")

    with mic_col:
        st.markdown("<br>", unsafe_allow_html=True)
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            key="voice_input"
        )

    # ── Voice transcription ──
    query = None

    if audio and audio.get("bytes"):
        with st.spinner("🎙️ Transcribing…"):
            try:
                transcribed = speech_to_text(audio["bytes"])
                if transcribed:
                    st.session_state.last_voice_text = transcribed
                    query = transcribed
            except Exception as e:
                st.error(f"Voice error: {e}")

    if st.session_state.last_voice_text and not query:
        # Show last transcription so user can confirm/edit
        st.markdown(
            f'<div class="voice-badge">🎤 Heard: "{st.session_state.last_voice_text}"</div>',
            unsafe_allow_html=True
        )

    if text_query:
        query = text_query
        st.session_state.last_voice_text = ""  # clear voice badge on text input

    # ── Process query ──
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            start = time.time()

            # Get streaming generator + sources (sources ready immediately from retrieval)
            token_stream, sources = chatbot.ask_stream(
                query,
                chat_history=st.session_state.messages[:-1]
            )

            # Stream tokens directly into the chat bubble — no waiting!
            full_answer = st.write_stream(token_stream)
            elapsed = time.time() - start

            st.caption(f"⚡ {elapsed:.1f}s")

            if sources:
                st.session_state.last_sources = sources
                with st.expander("📚 Sources used", expanded=True):
                    for src in sources:
                        pg_str = f" · Page {src['page']}" if src.get("page") else ""
                        col_a, col_b = st.columns([5, 1])
                        with col_a:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>📄 {src["file"]}{pg_str}</strong>'
                                f'<div class="source-snippet">{src["snippet"][:250]}…</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        with col_b:
                            if src["file"].endswith(".pdf"):
                                if st.button("👁️", key=f"view_{src['file']}_{src['page']}", help="View PDF"):
                                    st.session_state.open_pdf = (src["file"], src.get("page", 1) or 1)
                                    st.rerun()

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources
        })

# ─────────────────────────────────────────────
# PDF VIEWER PANEL
# ─────────────────────────────────────────────
if st.session_state.open_pdf and viewer_col:
    with viewer_col:
        file, page = st.session_state.open_pdf
        file_path = os.path.join(DATA_PATH, file)

        header_col, close_col = st.columns([5, 1])
        with header_col:
            st.markdown(f"### 📄 {file} — Page {page}")
        with close_col:
            if st.button("✖ Close"):
                st.session_state.open_pdf = None
                st.rerun()

        if os.path.exists(file_path):
            pdf_viewer(file_path, width=680, height=850, page=page)
        else:
            st.error("File not found.")
