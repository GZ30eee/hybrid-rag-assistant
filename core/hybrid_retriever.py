import streamlit as st
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import itertools
from collections import namedtuple

# Named tuple for consistent result object
SearchResult = namedtuple("SearchResult", ["text", "metadata", "score"])

# --- Model & Index Initialization ---
@st.cache_resource
def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """Load the Sentence-Transformer model."""
    return SentenceTransformer(model_name)

def get_current_model():
    model_name = st.session_state.get("embedding_model", "all-MiniLM-L6-v2")
    return get_sentence_transformer(model_name)

def chunk_text(text, filename, chunk_size=500, chunk_overlap=100):
    """Chunks text into smaller pieces with metadata."""
    chunks = []
    tokens = text.split()
    if not tokens:
        return chunks

    for i in range(0, len(tokens), max(1, chunk_size - chunk_overlap)):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = " ".join(chunk_tokens)
        chunks.append({
            "text": chunk,
            "metadata": {
                "source": filename,
                "type": "doc",
                "char_offset": i,
            },
        })
    return chunks

def create_document_index(corpus):
    """
    Chunks documents and creates BM25 and FAISS indexes.
    Stores them in Streamlit's session state.
    """
    st.session_state.doc_chunks = []
    
    chunk_size = st.session_state.get("chunk_size", 500)
    chunk_overlap = st.session_state.get("chunk_overlap", 100)
    
    # Chunking and preparing data for indexing
    for doc in corpus:
        st.session_state.doc_chunks.extend(
            chunk_text(doc["content"], doc["filename"], chunk_size, chunk_overlap)
        )

    if not st.session_state.doc_chunks:
        return

    # BM25 Index
    tokenized_corpus = [chunk["text"].split(" ") for chunk in st.session_state.doc_chunks]
    st.session_state.bm25_retriever = BM25Okapi(tokenized_corpus)

    # FAISS Index
    model = get_current_model()
    corpus_texts = [chunk["text"] for chunk in st.session_state.doc_chunks]
    corpus_embeddings = model.encode(corpus_texts)
    d = corpus_embeddings.shape[1]
    
    st.session_state.faiss_index = faiss.IndexFlatIP(d)
    st.session_state.faiss_index.add(np.array(corpus_embeddings))

def retrieve_documents(query, k):
    """Performs hybrid retrieval on document corpus with metadata filtering."""
    if not st.session_state.doc_chunks:
        return []
        
    # Metadata filtering setup
    selected_docs = st.session_state.get("selected_docs", [])
    valid_indices = set(range(len(st.session_state.doc_chunks)))
    if selected_docs:
        valid_indices = {i for i, chunk in enumerate(st.session_state.doc_chunks) if chunk["metadata"]["source"] in selected_docs}

    if not valid_indices:
        return []

    # Dense Retrieval (FAISS)
    model = get_current_model()
    query_embedding = model.encode(query)
    # Search deeper to allow for filtering dropping results
    D, I = st.session_state.faiss_index.search(
        np.array([query_embedding]), max(k * 5, len(st.session_state.doc_chunks))
    )
    dense_results = [
        {"chunk_idx": int(i), "score": d} for d, i in zip(D[0], I[0]) if int(i) in valid_indices
    ]

    # Sparse Retrieval (BM25)
    tokenized_query = query.split(" ")
    sparse_scores = st.session_state.bm25_retriever.get_scores(tokenized_query)
    sparse_results = [
        {"chunk_idx": i, "score": score}
        for i, score in enumerate(sparse_scores) if i in valid_indices
    ]
    
    # Normalize scores
    max_dense = max(r["score"] for r in dense_results) if dense_results else 1
    min_dense = min(r["score"] for r in dense_results) if dense_results else 0
    max_sparse = max(r["score"] for r in sparse_results) if sparse_results else 1
    min_sparse = min(r["score"] for r in sparse_results) if sparse_results else 0

    for r in dense_results:
        r["norm_score"] = (r["score"] - min_dense) / (max_dense - min_dense) if max_dense != min_dense else 0
    for r in sparse_results:
        r["norm_score"] = (r["score"] - min_sparse) / (max_sparse - min_sparse) if max_sparse != min_sparse else 0

    # Combine & re-rank (default alpha=0.5)
    combined_scores = {}
    for r in dense_results:
        combined_scores[r["chunk_idx"]] = combined_scores.get(r["chunk_idx"], 0) + (1 - st.session_state.alpha) * r["norm_score"]
    for r in sparse_results:
        combined_scores[r["chunk_idx"]] = combined_scores.get(r["chunk_idx"], 0) + st.session_state.alpha * r["norm_score"]

    sorted_indices = sorted(
        combined_scores, key=combined_scores.get, reverse=True
    )[:k]
    
    final_results = []
    for idx in sorted_indices:
        chunk = st.session_state.doc_chunks[idx]
        final_results.append(
            SearchResult(
                text=chunk["text"],
                metadata=chunk["metadata"],
                score=combined_scores[idx] / 2, # Normalize combined score to 0-1 range
            )
        )
    return final_results

def combine_results(doc_results, web_results, alpha):
    """Combines and re-ranks document and web results."""
    all_results = doc_results + web_results

    if not all_results:
        return []

    # Normalize scores from 0 to 1
    max_score = max(r.score for r in all_results) if all_results else 1
    normalized_results = [
        SearchResult(
            text=r.text,
            metadata=r.metadata,
            score=r.score / max_score if max_score > 0 else 0,
        )
        for r in all_results
    ]

    # Sort by normalized score and deduplicate
    normalized_results.sort(key=lambda x: x.score, reverse=True)
    
    # Simple deduplication based on text content
    seen_texts = set()
    final_results = []
    for res in normalized_results:
        if res.text not in seen_texts:
            final_results.append(res)
            seen_texts.add(res.text)
    
    return final_results