import streamlit as st
import os
import time
from core import (
    document_parser,
    hybrid_retriever,
    llm_interface,
    session_manager,
    web_search,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG-Based Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
session_manager.initialize_session()

# --- Utility Functions ---
def display_results(results, mode):
    """Displays the retrieved snippets in the main UI."""
    if not results:
        st.info("No relevant results found. Please try a different query.")
        return

    st.subheader(f"Relevant Snippets ({mode} Mode)")
    for i, res in enumerate(results):
        source = res.metadata.get("source", "Unknown")
        score = res.score * 100 if res.score is not None else 0

        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            # Highlight query terms
            snippet_text = res.text
            for term in st.session_state.query.split():
                snippet_text = snippet_text.replace(
                    term, f"<mark>**{term}**</mark>"
                )

            # Display with markdown
            with st.expander(
                f"**{source}** ({score:.2f}%)",
                expanded=False,
            ):
                st.markdown(snippet_text, unsafe_allow_html=True)
                
                if res.metadata.get("type") == "web":
                    st.markdown(
                        f"**Source URL**: [{res.metadata.get('url')}]({res.metadata.get('url')})"
                    )

        with col2:
            st.button(
                "📋",
                key=f"copy_{mode}_{i}",
                help="Copy to clipboard",
                on_click=lambda text=res.text: st.code(text, language="text"),
            )
            # st.button("📌", key=f"pin_{mode}_{i}", help="Pin this item")
            # Currently just a placeholder, full pin logic would be more complex

def clear_session():
    """Clears all session state variables."""
    session_manager.clear_session()
    st.success("Session cleared. All documents and indexes have been removed.")
    st.rerun()

# --- UI Layout ---
st.title("RAG-Based Document + Web Q&A 🤖")
st.markdown("Ask questions about your documents or the web!")

# --- Sidebar for Settings and Document Management ---
with st.sidebar:
    st.header("⚙️ App Settings & Data")
    
    with st.expander("📄 Document Management", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt", "html", "csv"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Processing files..."):
                parsed_docs = []
                for uploaded_file in uploaded_files:
                    doc_text = document_parser.parse_file(uploaded_file)
                    if doc_text:
                        parsed_docs.append(
                            {"content": doc_text, "filename": uploaded_file.name}
                        )
                
                if parsed_docs:
                    st.session_state.corpus = parsed_docs
                    hybrid_retriever.create_document_index(st.session_state.corpus)
                    st.session_state.doc_retriever = True
                    st.success(f"{len(parsed_docs)} docs processed and indexed!")
                else:
                    st.error("No documents could be parsed.")
                    st.session_state.doc_retriever = None

        if st.session_state.corpus:
            st.write(f"**Total files:** {len(st.session_state.corpus)}")
            for i, doc in enumerate(st.session_state.corpus):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    with st.popover(doc['filename']):
                        st.markdown(f"**Preview:**\n\n{doc['content'][:500]}...")
                with col2:
                    if st.button("❌", key=f"remove_doc_{i}"):
                        st.session_state.corpus.pop(i)
                        st.session_state.doc_retriever = None
                        st.success("Removed! Re-upload to re-index.")
                        st.rerun()

    with st.expander("🔍 Retrieval & Search", expanded=False):
        st.session_state.mode = st.radio(
            "Q&A Mode:", ("Hybrid", "Document", "Web"), index=0, horizontal=True
        )
        st.session_state.alpha = st.slider(
            "Hybrid Score Weight (Alpha)", 0.0, 1.0, st.session_state.alpha, 0.1,
            help="0.0 = Vector Semantic Search only, 1.0 = BM25 Keyword Search only"
        )
        st.session_state.top_k_doc = st.slider("Top K Documents", 1, 10, st.session_state.get("top_k_doc", 5))
        st.session_state.top_k_web = st.slider("Top K Web Results", 1, 10, st.session_state.get("top_k_web", 5))
        
        st.divider()
        st.markdown("**Advanced Indexing**")
        st.session_state.chunk_size = st.number_input("Chunk Size", value=st.session_state.chunk_size, step=50)
        st.session_state.chunk_overlap = st.number_input("Chunk Overlap", value=st.session_state.chunk_overlap, step=10)
        
        embedding_models = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", "paraphrase-multilingual-MiniLM-L12-v2"]
        curr_idx = embedding_models.index(st.session_state.embedding_model) if st.session_state.embedding_model in embedding_models else 0
        st.session_state.embedding_model = st.selectbox("Embedding Model", embedding_models, index=curr_idx)
        
        if st.session_state.corpus:
            options = [doc["filename"] for doc in st.session_state.corpus]
            st.session_state.selected_docs = st.multiselect("Filter by Source File", options, default=st.session_state.selected_docs)

    with st.expander("🤖 LLM Settings", expanded=False):
        st.session_state.llm_temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.llm_temperature, 0.1, help="Higher = more creative/random")
        st.session_state.llm_max_tokens = st.slider("Max Output Tokens", 256, 4096, st.session_state.llm_max_tokens, 256)
        st.session_state.system_prompt = st.text_area("System Prompt Template", value=st.session_state.system_prompt, height=100)

    st.divider()
    if st.button("🗑️ Clear All Data", type="primary", use_container_width=True):
        clear_session()
        
# --- Main Content Area for Q&A ---
query = st.chat_input("Ask a question...")

# FIX: This conditional is now based on doc_retriever
if not st.session_state.get("doc_retriever"):
    st.info("No documents indexed. Please upload files.")

if query:
    st.session_state.query = query
    start_time = time.time()
    st.session_state.history.append({"query": query, "answer": None, "mode": st.session_state.mode})

    # --- Retrieval Logic ---
    with st.spinner("Searching for answers..."):
        retrieved_docs = []
        if st.session_state.mode in ["Document", "Hybrid"]:
            if st.session_state.doc_retriever:
                retrieved_docs = hybrid_retriever.retrieve_documents(
                    query, st.session_state.top_k_doc
                )
            else:
                st.warning("No documents indexed. Please upload files.")

        web_results = []
        if st.session_state.mode in ["Web", "Hybrid"]:
            if os.getenv("WEB_SEARCH_API_KEY"):
                web_results = web_search.perform_web_search(
                    query, st.session_state.top_k_web
                )
            else:
                st.warning(
                    "Web Search API key not configured. Cannot perform web search."
                )

        combined_results = []
        if st.session_state.mode == "Hybrid":
            if retrieved_docs or web_results:
                combined_results = hybrid_retriever.combine_results(
                    retrieved_docs, web_results, st.session_state.alpha
                )
            else:
                st.error("No sources available for Hybrid mode.")
        elif st.session_state.mode == "Document":
            combined_results = retrieved_docs
        elif st.session_state.mode == "Web":
            combined_results = web_results

        # --- LLM Generation ---
        if combined_results:
            context = "\n\n".join([res.text for res in combined_results])
            
            answer_type = st.radio(
                "Answer Format:",
                ("Bulleted List", "Paragraph"),
                horizontal=True,
            )

            llm_response = llm_interface.generate_answer(
                query,
                context,
                answer_type,
                combined_results,
                chat_history=st.session_state.history[:-1] # pass history excluding current incomplete turn
            )
            
            st.session_state.history[-1]["answer"] = llm_response
            st.session_state.history[-1]["context"] = combined_results

            # --- Display LLM Answer ---
            st.header("Answer")
            
            # Short bulleted answer with citations
            st.subheader("Concise Answer")
            st.markdown(llm_response["short"], unsafe_allow_html=True)
            
            st.subheader("Web-Ready Paragraph")
            st.write(llm_response["web_ready"])
            
            # Display Sources
            st.subheader("Sources")
            for source in llm_response["sources"]:
                st.write(source)
                
            # Download Response Button
            download_content = f"Question: {query}\n\nConcise Answer:\n{llm_response['short']}\n\nWeb-Ready Paragraph:\n{llm_response['web_ready']}\n\nSources:\n" + "\n".join(llm_response["sources"])
            st.download_button(
                label="📥 Download Answer",
                data=download_content,
                file_name="rag_response.txt",
                mime="text/plain",
            )

        else:
            st.warning("Could not find relevant information to answer the question.")
    
    end_time = time.time()
    retrieval_time = end_time - start_time
    st.session_state.history[-1]["time"] = retrieval_time
    
    # --- Display Snippets ---
    display_results(combined_results, st.session_state.mode)

# --- Right-side History Panel ---
with st.sidebar:
    st.header("Query History")
    if st.session_state.history:
        history_text = "CHAT HISTORY\n" + "="*20 + "\n\n"
        for item in st.session_state.history:
            history_text += f"Q: {item['query']}\n"
            if item.get("answer"):
                history_text += f"A: {item['answer']['short']}\n\n"
        
        st.download_button(
            label="💾 Download Conversation",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )
        
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"**Q:** {item['query']}"):
            if item["answer"]:
                st.write("**Mode:**", item["mode"])
                st.write(
                    f"**Answer:**",
                    item["answer"]["short"],
                    unsafe_allow_html=True,
                )
                st.write(f"**Time:** {item['time']:.2f}s")
            else:
                st.write("Processing...")