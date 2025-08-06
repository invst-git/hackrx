# ingestion_pipeline.py

import os
import requests
from urllib.parse import urlparse, unquote
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def download_file(url: str, save_dir: str = "document_storage"):
    """
    Downloads a file from a URL and saves it locally with a clean filename.
    """
    try:
        parsed_url = urlparse(url)
        file_name = unquote(os.path.basename(parsed_url.path))
        
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        print(f"Downloading file from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded and saved to: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

def extract_text_from_file(file_path: str):
    """
    Extracts text and metadata from a local file (PDF).
    """
    try:
        doc_name = os.path.basename(file_path)
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            print(f"Successfully extracted text from '{doc_name}'.")
            return text, {"document_name": doc_name}
        else:
            print(f"Unsupported file type: {doc_name}. Skipping.")
            return "", {"document_name": doc_name}
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return "", {}

def chunk_and_enrich_text(full_text: str, metadata: dict):
    """
    Chunks the text and enriches each chunk with metadata.
    """
    if not full_text:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(full_text)
    
    enriched_chunks = []
    for i, chunk_text in enumerate(chunks):
        enriched_chunks.append({
            "id": f"chunk_{metadata.get('document_name', 'doc')}_{i}",
            "passage_text": chunk_text,
            "document_name": metadata.get('document_name'),
            "page_range": "N/A",  # Page range info is complex, so we'll omit for now
            "clause_section_heading": "N/A"
        })
    print(f"Created {len(enriched_chunks)} enriched chunks.")
    return enriched_chunks
