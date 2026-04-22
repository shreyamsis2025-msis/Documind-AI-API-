from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from vectordb.faiss_store import load_db
from config import GROQ_MODEL, RETRIEVER_K, GROQ_API_KEY
import os


class RAGChat:
    def __init__(self):
        self.db = load_db()
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1024,
            streaming=True,
        )

    def reload_db(self):
        """Call this after new documents are uploaded."""
        self.db = load_db()

    def _build_messages(self, query: str, context: str, history_str: str):
        system = """You are a document assistant. Your ONLY job is to answer questions using the document excerpts provided.

STRICT RULES:
1. Answer ONLY from the document excerpts. Do NOT use outside knowledge.
2. If the answer is not found in the excerpts, respond ONLY with: "I could not find information about this in the uploaded documents."
3. Never fabricate, infer, or add information not explicitly stated in the excerpts.
4. If answering about a resume or personal document, extract and present the actual details exactly as they appear in the text.
5. Cite the source file at the end of your answer, e.g. [Source: filename.pdf, Page 2]."""

        user_content = f"""{f"--- Conversation so far ---{chr(10)}{history_str}{chr(10)}" if history_str else ""}\
--- Document Excerpts ---
{context}

--- Question ---
{query}"""

        return [SystemMessage(content=system), HumanMessage(content=user_content)]

    def _retrieve(self, query: str):
        if self.db is None:
            return None, None
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3}
        )
        docs = retriever.invoke(query)
        context_parts = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "Unknown")
            pg = d.metadata.get("page", "")
            label = f"[{i}] Source: {src}" + (f", Page {pg}" if pg else "")
            context_parts.append(f"{label}\n{d.page_content}")
        return docs, "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, docs) -> list[dict]:
        sources, seen = [], set()
        for d in docs:
            src = os.path.basename(d.metadata.get("source", "Unknown"))
            pg = d.metadata.get("page", "")
            key = f"{src}||{pg}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": src,
                    "page": pg,
                    "label": f"{src}" + (f" (Page {pg})" if pg else ""),
                    "snippet": d.page_content[:300]
                })
        return sources

    def _history_str(self, chat_history: list[dict]) -> str:
        pairs = []
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            pairs.append(f"{role}: {msg['content']}")
        return "\n".join(pairs)

    def ask_stream(self, query: str, chat_history: list[dict] | None = None):
        """
        Returns (token_generator, sources_list).
        Tokens stream in from Groq in real time.
        """
        chat_history = chat_history or []

        if self.db is None:
            def _no_docs():
                yield "⚠️ No documents have been uploaded yet. Please upload a PDF, DOCX, CSV, or other file using the sidebar, then ask your question."
            return _no_docs(), []

        docs, context = self._retrieve(query)
        history_str = self._history_str(chat_history)
        messages = self._build_messages(query, context, history_str)
        sources = self._extract_sources(docs)

        def _stream():
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content

        return _stream(), sources

    def ask(self, query: str, chat_history: list[dict] | None = None) -> dict:
        """Non-streaming version (used as fallback)."""
        chat_history = chat_history or []

        if self.db is None:
            return {
                "answer": "⚠️ No documents uploaded yet.",
                "sources": [], "snippets": []
            }

        docs, context = self._retrieve(query)
        history_str = self._history_str(chat_history)
        messages = self._build_messages(query, context, history_str)
        response = self.llm.invoke(messages)
        sources = self._extract_sources(docs)

        return {
            "answer": response.content.strip(),
            "sources": sources,
            "snippets": [d.page_content[:300] for d in docs]
        }
