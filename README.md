# 🤖 Hybrid RAG Assistant: Document + Web Q&A

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/GZ30eee/hybrid-rag-assistant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A powerful, production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit. This assistant enables users to perform intelligent Q&A over their own document corpus AND live web search results using a state-of-the-art hybrid retrieval strategy.

---

## ✨ Key Features

- 📄 **Multi-Format Document Support**: Seamlessly upload and index PDF, DOCX, TXT, HTML, and CSV files.
- 🔍 **Hybrid Retrieval Engine**: Combines the precision of **BM25 (Sparse/Keyword Search)** with the semantic depth of **FAISS (Dense/Vector Search)**.
- 🌐 **Live Web Search Integration**: Real-time integration with web search APIs to augment local knowledge with the latest information from the internet.
- 🤖 **LLM-Powered Intelligence**: Uses Google Gemini (or fallbacks) to generate concise, citable answers and "web-ready" summary paragraphs.
- ⚙️ **Advanced Configuration**: Fine-tune your RAG pipeline with adjustable parameters:
    - **Hybrid Alpha**: Balance between keyword and semantic search.
    - **Chunking Strategy**: Customize chunk size and overlap for optimal context.
    - **Model Selection**: Choose from various Sentence-Transformer embedding models.
- 💾 **Session & History Management**: Interactive sidebar for document management and a persistent query history with export capabilities.
- 🎨 **Modern UI/UX**: Clean, responsive Streamlit interface with syntax highlighting, citation tracking, and interactive snippet previews.

---

## 🏗️ Technical Architecture

The application follows a modular architecture designed for extensibility:

### 1. Document Processing (`core/document_parser.py`)
Uses robust libraries like `PyMuPDF`, `python-docx`, and `BeautifulSoup4` to extract clean text from various file formats.

### 2. Hybrid Retriever (`core/hybrid_retriever.py`)
- **Indexing**: Documents are chunked and indexed dual-path via `rank-bm25` and `faiss-cpu`.
- **Search**: Implements a weighted scoring system ($Score = \alpha \cdot BM25 + (1-\alpha) \cdot Vector$) to provide the most relevant context snippets.

### 3. LLM Interface (`core/llm_interface.py`)
Handles prompt engineering and communication with the Gemini API, ensuring structured outputs with citations and different formats (bullets vs. paragraphs).

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- (Optional) Web Search API Key (SerpAPI or similar)

### Step 1: Clone the repository
```bash
git clone https://github.com/GZ30eee/hybrid-rag-assistant.git
cd hybrid-rag-assistant
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Required for LLM
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional for Web Search
WEB_SEARCH_API_KEY=your_web_search_api_key_here
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
hybrid-rag-assistant/
├── app.py                # Main Streamlit application entry point
├── core/                 # Core logic modules
│   ├── document_parser.py # File parsing and text extraction
│   ├── hybrid_retriever.py# BM25 + FAISS retrieval logic
│   ├── llm_interface.py   # LLM prompt and API handling
│   ├── session_manager.py # Streamlit session state management
│   └── web_search.py      # Web search API integration
├── assets/               # UI assets (icons, images)
├── tests/                # Unit and integration tests
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🗺️ Roadmap
- [ ] Support for OCR (Optical Character Recognition) in PDFs.
- [ ] Integration with more Vector DBs (Chroma, Pinecone).
- [ ] Multi-user authentication and database persistence.
- [ ] Support for local LLMs via Ollama.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.