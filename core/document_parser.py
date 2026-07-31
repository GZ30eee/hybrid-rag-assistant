import streamlit as st
import fitz  # PyMuPDF
import docx
import io
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def parse_pdf(file_stream):
    """Parses a PDF file, including basic OCR for scanned pages."""
    text = ""
    try:
        pdf_bytes = file_stream.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(pdf):
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                # Try OCR if no text found
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text += ocr_text
                    else:
                        logger.warning(f"No text found on page {page_num+1} even after OCR.")
                except Exception as e:
                    logger.warning(f"OCR failed on page {page_num+1}: {e}")
        return text
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        st.error(f"Failed to parse PDF: {e}")
        return ""

def parse_docx(file_stream):
    """Parses a DOCX file."""
    try:
        doc = docx.Document(file_stream)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Failed to parse DOCX: {e}")
        st.error(f"Failed to parse DOCX: {e}")
        return ""

def parse_txt(file_stream):
    """Parses a TXT file."""
    try:
        return file_stream.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to parse TXT: {e}")
        st.error(f"Failed to parse TXT: {e}")
        return ""

def parse_html(file_stream):
    """Parses an HTML file, extracting main content."""
    try:
        soup = BeautifulSoup(file_stream, "html.parser")
        main_content = soup.find("main") or soup.find("body")
        if main_content:
            for script_or_style in main_content(["script", "style"]):
                script_or_style.extract()
            return main_content.get_text(separator="\n", strip=True)
        return ""
    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        st.error(f"Failed to parse HTML: {e}")
        return ""

def parse_file(uploaded_file):
    """Parses a single uploaded file based on its type."""
    file_type = uploaded_file.type
    file_stream = io.BytesIO(uploaded_file.getvalue())

    if file_type == "application/pdf":
        return parse_pdf(file_stream)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(file_stream)
    elif file_type == "text/plain":
        return parse_txt(file_stream)
    elif file_type == "text/html":
        return parse_html(file_stream)
    elif file_type == "text/csv":
        return parse_txt(file_stream)
    else:
        st.warning(f"Unsupported file type: {file_type}")
        return ""
