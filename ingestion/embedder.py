from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Free local embeddings via sentence-transformers.
    No API key needed. Downloaded once and cached.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
