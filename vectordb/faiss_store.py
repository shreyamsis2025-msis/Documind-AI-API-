import os
from langchain_community.vectorstores import FAISS
from ingestion.embedder import get_embeddings
from config import FAISS_PATH


def save_db(chunks):
    embeddings = get_embeddings()

    # If FAISS index already exists, merge new chunks instead of overwriting
    if os.path.exists(FAISS_PATH):
        db = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(FAISS_PATH)


def load_db():
    """
    Load the FAISS index from disk.
    NOT cached with @st.cache_resource — must reload fresh every time
    chatbot.reload_db() is called after new documents are indexed.
    The embeddings model itself is separately cached via get_embeddings().
    """
    if not os.path.exists(FAISS_PATH):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
