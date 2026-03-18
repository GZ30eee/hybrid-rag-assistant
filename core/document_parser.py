import streamlit as st
import fitz  # PyMuPDF
import docx
import io
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup

def parse_pdf(file_stream):
    """Parses a PDF file, including basic OCR for scanned pages."""
    text = ""
    try:
        pdf_bytes = file_stream.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                # Try OCR if no text was found on the page
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img)
                except Exception as e:
                    st.warning(f"OCR failed on a page: {e}")
        return text
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
        return ""

def parse_docx(file_stream):
    """Parses a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_stream)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Failed to parse DOCX: {e}")
        return ""

def parse_txt(file_stream):
    """Parses a TXT file."""
    try:
        return file_stream.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to parse TXT: {e}")
        return ""

def parse_html(file_stream):
    """Parses an HTML file, extracting main content."""
    try:
        soup = BeautifulSoup(file_stream, "html.parser")
        main_content = soup.find("main") or soup.find("body")
        for script_or_style in main_content(["script", "style"]):
            script_or_style.extract()
        return main_content.get_text(separator="\n", strip=True)
    except Exception as e:
        st.error(f"Failed to parse HTML: {e}")
        return ""

def parse_file(uploaded_file):
    """Parses a single uploaded file based on its type."""
    file_type = uploaded_file.type
    file_stream = io.BytesIO(uploaded_file.getvalue())

    if file_type == "application/pdf":
        return parse_pdf(file_stream)
    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return parse_docx(file_stream)
    elif file_type == "text/plain":
        return parse_txt(file_stream)
    elif file_type == "text/html":
        return parse_html(file_stream)
    elif file_type == "text/csv":
        return parse_txt(file_stream) # CSV is parsed just like TXT
    else:
        st.warning(f"Unsupported file type: {file_type}")
        return ""