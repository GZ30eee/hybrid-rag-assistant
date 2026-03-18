import streamlit as st

def initialize_session():
    """Initializes all necessary session state variables."""
    if "corpus" not in st.session_state:
        st.session_state.corpus = []
    if "bm25_retriever" not in st.session_state:
        st.session_state.bm25_retriever = None
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "doc_retriever" not in st.session_state:
        st.session_state.doc_retriever = None
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.5
    if "query" not in st.session_state:
        st.session_state.query = ""
    # New Settings variables
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 500
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "all-MiniLM-L6-v2"
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7
    if "llm_max_tokens" not in st.session_state:
        st.session_state.llm_max_tokens = 1024
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful and expert Q&A assistant."
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = [] # Metadata filter

def clear_session():
    """Clears all state variables except basic config."""
    persistent_keys = [
        "alpha", "top_k_doc", "top_k_web", "mode", "chunk_size", 
        "chunk_overlap", "embedding_model", "llm_temperature", 
        "llm_max_tokens", "system_prompt", "selected_docs"
    ]
    for key in list(st.session_state.keys()):
        if key not in persistent_keys:
            del st.session_state[key]
    initialize_session()