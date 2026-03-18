# 📄 RAG-based Document + Web Q&A Streamlit App

This is a comprehensive Streamlit application that provides a powerful Q&A system over user-uploaded documents and live web search results. It employs a hybrid retrieval strategy combining BM25 and dense vector embeddings to deliver highly relevant answers.

## ✨ Features
- **Document Management**: Upload PDF, DOCX, TXT, and HTML files.
- **Hybrid Retrieval**: Combines BM25 (sparse) and dense embeddings (semantic) search.
- **Q&A Modes**: Toggle between Document, Web, and Hybrid Q&A modes.
- **Web Search Integration**: Fetches and processes content from live web search results.
- **LLM-Powered Answers**: Generates concise, citable answers and "web-ready" paragraphs using Gemini or a fallback LLM.
- **Interactive UI**: Clean Streamlit interface with a sidebar for settings, a main area for Q&A, and a history panel.
- **Robustness**: Graceful error handling, session management, and environment variable-based configuration.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- A Google Gemini API key and/or a Web Search API key (e.g., SerpAPI, Bing, Google CSE).

### Step 1: Clone the repository
```bash
git clone https://github.com/GZ30eee/hybrid-rag-assistant.git
cd hybrid-rag-assistant
```