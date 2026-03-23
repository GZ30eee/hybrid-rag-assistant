import streamlit as st
import google.generativeai as genai
import re

def configure_gemini():
    """Configures the Gemini API if the key is present."""
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY is not set in environment variables.")
        return None
    
    genai.configure(api_key=gemini_api_key)
    # Change gemini-pro to a supported model name, e.g., gemini-2.5-flash
    return genai.GenerativeModel("gemini-2.5-flash")

@st.cache_data
def generate_answer(query, context, answer_format, sources, chat_history=None):
    """
    Generates an answer using the LLM based on context and query.
    
    Args:
        query (str): The user's question.
        context (str): The combined text from retrieved documents and web snippets.
        answer_format (str): "Bulleted List" or "Paragraph".
        sources (list): The list of SearchResult objects with metadata.
        chat_history (list): List of previous chat interactions.
        
    Returns:
        dict: A dictionary containing the short answer and web-ready paragraph.
    """
    if chat_history is None:
        chat_history = []
        
    model = configure_gemini()
    if not model:
        return {
            "short": "LLM not configured. Please set GEMINI_API_KEY.",
            "web_ready": "LLM not configured.",
            "sources": [],
        }
    
    source_map = {
        res.metadata.get("source", "Unknown"): i
        for i, res in enumerate(sources)
    }

    # Format chat history
    formatted_history = ""
    if chat_history:
        formatted_history = "PREVIOUS CONVERSATION HISTORY:\n"
        for msg in chat_history[-3:]: # Get the last 3 turns
            formatted_history += f"User: {msg['query']}\n"
            if msg.get('answer') and isinstance(msg['answer'], dict):
                formatted_history += f"Assistant: {msg['answer'].get('short', '')}\n\n"

    system_prompt_text = st.session_state.get("system_prompt", "You are a helpful and expert Q&A assistant.")
    
    # Custom prompt for the LLM
    prompt = f"""
    {system_prompt_text}
    Your task is to answer the user's question using ONLY the provided context and the conversation history.
    If the context does not contain the answer, state that you do not have enough information.
    Do not mention that you're an AI or an LLM.

    {formatted_history}

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Answer strictly in the following format:
    1.  A concise, bulleted list answer with at most 6 bullets. Each bullet must end with a citation tag like [DOC: document_name.pdf] or [WEB: google.com].
    2.  A single, well-written, web-ready paragraph. This paragraph should be a cohesive summary, not a list. Do not include citations in this paragraph.
    3.  A short list of the source titles used.
    
    If the answer is not in the context, your response for both sections should be "I could not find a definitive answer to your question in the provided information."
    """

    try:
        generation_config = genai.types.GenerationConfig(
            temperature=st.session_state.get("llm_temperature", 0.7),
            max_output_tokens=st.session_state.get("llm_max_tokens", 1024),
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        full_text = response.text.strip()
        
        # Split the response into the two required parts
        parts = re.split(
            r"^\s*2\.\s*A\s*single,\s*well-written,\s*web-ready\s*paragraph\.",
            full_text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        
        short_answer_raw = parts[0].strip()
        web_ready_raw = ""
        if len(parts) > 1:
            web_ready_raw = parts[1].split("3. A short list of the source titles used.")[0].strip()
            
        # Extract sources list
        sources_list_raw = re.search(
            r"3\.\s*A\s*short\s*list\s*of\s*the\s*source\s*titles\s*used\.(.*)",
            full_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        sources_list = []
        if sources_list_raw:
            sources_list = [
                s.strip() for s in sources_list_raw.group(1).split("\n") if s.strip()
            ]

        # Final cleaning and formatting
        short_answer = short_answer_raw.replace("1. ", "")
        
        return {
            "short": short_answer,
            "web_ready": web_ready_raw,
            "sources": sources_list,
        }
        
    except Exception as e:
        st.error(f"LLM generation failed: {e}")
        return {
            "short": "An error occurred during answer generation.",
            "web_ready": "",
            "sources": [],
        }