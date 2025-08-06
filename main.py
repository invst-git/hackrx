# main.py

import os
import uuid
import requests
from urllib.parse import urlparse, unquote
import fitz  # PyMuPDF
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. INITIALIZATION & SETUP ---

# Load environment variables and initialize clients
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize FastAPI app
app = FastAPI()

# --- 2. HELPER FUNCTIONS (Consolidated from other files) ---

def download_file(url: str, save_dir: str):
    """Downloads a file from a URL and saves it locally with a clean filename."""
    try:
        parsed_url = urlparse(url)
        file_name = unquote(os.path.basename(parsed_url.path))
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        print(f"Downloading file from: {url} to {file_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded file: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

def extract_text_from_file(file_path: str):
    """Extracts text and metadata from a local PDF file."""
    doc_name = os.path.basename(file_path)
    if not file_path.lower().endswith(".pdf"):
        print(f"Unsupported file type: {doc_name}. Only .pdf is supported.")
        return "", {"document_name": doc_name}
    try:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        print(f"Successfully extracted {len(text)} characters from '{doc_name}'.")
        return text, {"document_name": doc_name}
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return "", {"document_name": doc_name}

def chunk_and_enrich_text(full_text: str, metadata: dict):
    """Chunks the text and enriches each chunk with metadata."""
    if not full_text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)
    
    enriched_chunks = []
    for i, chunk_text in enumerate(chunks):
        enriched_chunks.append({
            "id": f"chunk_{metadata.get('document_name', 'doc')}_{i}",
            "passage_text": chunk_text,
            "document_name": metadata.get('document_name'),
            "page_range": "N/A",
            "clause_section_heading": "N/A"
        })
    print(f"Created {len(enriched_chunks)} text chunks.")
    return enriched_chunks

def find_relevant_chunks(question: str, namespace: str):
    """Embeds a question and retrieves relevant chunks from a Pinecone namespace."""
    response = openai_client.embeddings.create(model="text-embedding-3-large", input=[question], dimensions=768)
    query_embedding = response.data[0].embedding
    query_results = pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
    return query_results['matches']

def generate_answer(question: str, retrieved_chunks: list):
    """Uses GPT-4o to generate an answer based on retrieved context."""
    context = "".join(f"--- CONTEXT ---\n{match['metadata']['passage_text']}\n\n" for match in retrieved_chunks)
    system_prompt = "You are an expert assistant. Synthesize a clear answer based *exclusively* on the provided context. If the context does not contain the answer, state that the information is not available in the documents."
    user_prompt = f"User Question: {question}\n\n{context}"
    
    response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1)
    return response.choices[0].message.content

# --- 3. API MODELS & ENDPOINT ---

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, http_request: Request):
    """Processes a document and answers questions about it."""
    if not http_request.headers.get("Authorization"):
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    namespace = str(uuid.uuid4())
    local_file_path = None
    chunks_were_created = False
    
    try:
        # A. Ingest and Process
        local_file_path = download_file(req.documents, save_dir="temp_docs")
        if not local_file_path:
            raise HTTPException(status_code=400, detail="Could not download document.")

        full_text, metadata = extract_text_from_file(local_file_path)
        chunks = chunk_and_enrich_text(full_text, metadata)
        
        # B. Index Chunks
        if chunks:
            chunks_were_created = True
            print(f"Indexing {len(chunks)} chunks into namespace: {namespace}")
            for i in tqdm(range(0, len(chunks), 100), desc="Indexing Batches"):
                batch = chunks[i:i + 100]
                response = openai_client.embeddings.create(model="text-embedding-3-large", input=[c['passage_text'] for c in batch], dimensions=768)
                vectors = [{"id": c['id'], "values": e.embedding, "metadata": c} for c, e in zip(batch, response.data)]
                pinecone_index.upsert(vectors=vectors, namespace=namespace)
        else:
            print("Warning: No chunks created. The document may be empty or unreadable.")

        # C. Answer Questions
        answers = []
        for question in tqdm(req.questions, desc="Answering Questions"):
            if not chunks:
                answers.append("Could not answer question; the document was empty or could not be processed.")
                continue
            
            matches = find_relevant_chunks(question, namespace)
            answers.append(generate_answer(question, matches) if matches else "Relevant information not found in the document.")
        
        return RunResponse(answers=answers)

    finally:
        # D. Cleanup
        print("Cleaning up resources...")
        if chunks_were_created:
            try:
                pinecone_index.delete(delete_all=True, namespace=namespace)
                print(f"Successfully deleted namespace: {namespace}")
            except Exception as e:
                print(f"Cleanup error (namespace may have already been cleared): {e}")
        
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"Successfully deleted temp file: {local_file_path}")

@app.get("/")
def read_root(): return {"status": "API is running."}
