import streamlit as st
import requests
import httpx
import numpy as np
from bs4 import BeautifulSoup
from readability import Document
from core.hybrid_retriever import (
    get_sentence_transformer,
    SearchResult,
    chunk_text,
)

# --- Web Search Providers ---
def search_serpapi(query, num_results=5):
    """Performs a web search using SerpAPI."""
    api_key = st.secrets.get("WEB_SEARCH_API_KEY")
    if not api_key:
        return []
        
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for result in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet"),
            })
        return results
    except requests.RequestException as e:
        st.error(f"SerpAPI search failed: {e}")
        return []

# --- Content Extraction & Retrieval ---
# In core/web_search.py

def fetch_and_parse_url(url):
    """Fetches a URL and extracts the main content."""
    # Add a headers dictionary with a user-agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        with httpx.Client(
            timeout=30, follow_redirects=True
        ) as client:
            # Pass the headers to the get request
            res = client.get(url, headers=headers)
            res.raise_for_status()
            
            doc = Document(res.text)
            title = doc.title()
            text = BeautifulSoup(
                doc.summary(), "html.parser"
            ).get_text(separator="\n", strip=True)
            
            return title, text
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        st.warning(f"Failed to fetch or parse {url}: {e}")
        return None, None

def perform_web_search(query, k):
    """
    Performs a web search, fetches and processes the top results,
    then retrieves the most relevant snippets.
    """
    web_provider = st.secrets.get("WEB_SEARCH_PROVIDER", "serpapi")
    
    if web_provider == "serpapi":
        search_results = search_serpapi(query, num_results=10)
    else:
        st.error(f"Unsupported web search provider: {web_provider}")
        return []

    if not search_results:
        return []

    web_chunks = []
    
    with st.spinner("Fetching and processing web content..."):
        for res in search_results:
            title, content = fetch_and_parse_url(res["link"])
            if content:
                # Chunking with metadata
                chunks = chunk_text(content, f"WEB: {title}")
                for chunk in chunks:
                    chunk["metadata"]["type"] = "web"
                    chunk["metadata"]["url"] = res["link"]
                web_chunks.extend(chunks)

    if not web_chunks:
        return []

    # Retrieve top K snippets using dense retrieval
    model = get_sentence_transformer()
    query_embedding = model.encode(query)
    
    chunk_embeddings = model.encode(
        [c["text"] for c in web_chunks]
    )
    
    # Calculate cosine similarity
    similarity_scores = np.dot(
        chunk_embeddings, query_embedding
    ) / (
        np.linalg.norm(chunk_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    # Combine scores with chunks and sort
    scored_chunks = sorted(
        zip(web_chunks, similarity_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    
    top_k_results = scored_chunks[:k]
    
    # Create SearchResult objects for consistent output
    final_results = [
        SearchResult(
            text=chunk["text"],
            metadata=chunk["metadata"],
            score=score,
        )
        for chunk, score in top_k_results
    ]
    
    return final_results