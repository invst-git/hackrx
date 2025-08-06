# ingestion_pipeline.py

import os
import requests
import uuid
import fitz  # PyMuPDF
import docx
import email
from PIL import Image
import pytesseract
import io
import tiktoken
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS

# --- 1. FILE DOWNLOADING (For URLs) ---
def download_file(url: str, save_dir: str = "document_storage"):
    """
    Downloads a file from a URL and saves it locally with a clean filename.
    """
    try:
        # --- FIX IS HERE ---
        # Parse the URL to get the path, and then get the base filename
        parsed_url = urlparse(url)
        # Use unquote to handle URL-encoded characters like '%20' for spaces
        file_name = unquote(os.path.basename(parsed_url.path))
        
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        print(f"Downloading file from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded and saved to: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

# --- 2. TEXT EXTRACTION (Works for any local file path) ---
def extract_text_from_file(file_path: str) -> tuple[str, dict]:
    # This function is unchanged as it already works with local paths
    full_text = ""
    doc_metadata = {"source": os.path.basename(file_path), "page_count": 0}
    file_extension = os.path.splitext(file_path)[1].lower()

    print(f"Extracting text from {doc_metadata['source']}...")

    if file_extension == ".pdf":
        doc = fitz.open(file_path)
        doc_metadata["page_count"] = len(doc)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                try:
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    text = pytesseract.image_to_string(img)
                    print(f"  - Used OCR for page {page_num}")
                except Exception as e:
                    print(f"  - OCR failed for page {page_num}: {e}")
                    text = ""
            full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
        doc.close()
    elif file_extension == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            full_text += para.text + "\n"
    elif file_extension == ".eml":
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
            subject = msg.get('Subject', 'No Subject')
            doc_metadata['subject'] = subject
            full_text += f"Subject: {subject}\n\n"
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        full_text += payload.decode('utf-8', errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                full_text += payload.decode('utf-8', errors='ignore')
    else:
        print(f"Unsupported file type: {file_path}. Skipping.")
        return "", {}

    print("Successfully extracted text.")
    return full_text, doc_metadata

# --- 3. TEXT CHUNKING & METADATA ENRICHMENT (Unchanged) ---
# --- 3. TEXT CHUNKING & METADATA ENRICHMENT ---
def chunk_and_enrich_text(full_text: str, doc_metadata: dict) -> list[dict]:
    """
    Chunks text using a token-aware splitter and enriches each chunk with metadata.
    """
    print("Chunking and enriching text...")
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=lambda text: len(encoding.encode(text)),
    )
    
    passages = text_splitter.split_text(full_text)
    enriched_chunks = []
    
    # --- FIX IS HERE ---
    # Changed the main capturing group ( ... ) to a non-capturing group (?: ... )
    # This makes re.findall() return a list of strings instead of a list of tuples.
    heading_pattern = re.compile(r'^(?:Section \d+(\.\d+)*|Clause [IVXLC\d]+|Article \d+|[A-Z][a-zA-Z\s]+:)', re.MULTILINE)
    
    for passage in passages:
        # Find page range for the chunk
        page_numbers = re.findall(r'--- Page (\d+) ---', passage)
        page_range = "N/A"
        if page_numbers:
            pages_int = list(map(int, page_numbers))
            min_page, max_page = min(pages_int), max(pages_int)
            page_range = str(min_page) if min_page == max_page else f"{min_page}-{max_page}"

        # Find the last heading within the chunk
        headings_in_chunk = heading_pattern.findall(passage)
        # This line will now work correctly because headings_in_chunk contains strings
        heading = headings_in_chunk[-1].strip().replace(':', '') if headings_in_chunk else "N/A"

        enriched_chunks.append({
            "id": f"chunk_{uuid.uuid4()}",
            "passage_text": passage,
            "document_name": doc_metadata.get("source"),
            "page_range": page_range,
            "clause_section_heading": heading
        })
        
    print(f"Created {len(enriched_chunks)} enriched chunks.")
    return enriched_chunks


# --- 4. NEW: PIPELINE ORCHESTRATOR FOR LOCAL FILES ---
def run_ingestion_from_local_paths(document_paths: list[str]) -> list[dict]:
    """Runs the full ingestion pipeline for a list of local file paths."""
    all_processed_chunks = []
    for path in document_paths:
        print(f"\n{'='*50}\nProcessing File: {path}\n{'='*50}")
        if not os.path.exists(path):
            print(f"File not found: {path}. Skipping.")
            continue
        full_text, doc_metadata = extract_text_from_file(path)
        if not full_text:
            print(f"Could not extract text from {path}. Skipping.")
            continue
        enriched_chunks = chunk_and_enrich_text(full_text, doc_metadata)
        all_processed_chunks.extend(enriched_chunks)
        print(f"--- Finished processing '{doc_metadata['source']}' ---")
    return all_processed_chunks

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # The script now defaults to processing local files.
    
    # 1. Define the directory where your local documents are stored.
    local_docs_dir = "document_storage"
    os.makedirs(local_docs_dir, exist_ok=True) # Creates the folder if it doesn't exist

    # 2. Automatically find all supported files in that directory.
    supported_extensions = ['.pdf', '.docx', '.eml']
    local_files_to_process = [
        os.path.join(local_docs_dir, f) 
        for f in os.listdir(local_docs_dir) 
        if any(f.lower().endswith(ext) for ext in supported_extensions)
    ]

    # 3. Run the pipeline.
    if not local_files_to_process:
        print(f"\nNo documents found in the '{local_docs_dir}' directory.")
        print("Please add your PDF, DOCX, or EML files to that folder and run again.")
    else:
        print(f"Found {len(local_files_to_process)} documents to process.")
        final_chunks = run_ingestion_from_local_paths(local_files_to_process)
        
        print(f"\n\n{'='*20} LOCAL INGESTION COMPLETE {'='*20}")
        print(f"Total chunks created: {len(final_chunks)}")
        
        if final_chunks:
            print("\n--- SAMPLE CHUNK OUTPUT ---")
            print(json.dumps(final_chunks[0], indent=2))

