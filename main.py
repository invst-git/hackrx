# main.py

import os
import uuid
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm

# Import the core functions from our previous scripts
from ingestion_pipeline import download_file, extract_text_from_file, chunk_and_enrich_text
from query_handler import find_relevant_chunks, generate_answer

# --- 1. INITIALIZATION & SETUP ---

# Load environment variables and initialize clients
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize FastAPI app
app = FastAPI()

# Define the request and response models to match the API documentation
class RunRequest(BaseModel):
    documents: str  # URL of the document
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

# --- 2. THE CORE API ENDPOINT ---

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, http_request: Request):
    """
    This endpoint processes a document from a URL and answers questions about it.
    """
    auth_header = http_request.headers.get("Authorization")
    if not auth_header:
         raise HTTPException(status_code=401, detail="Authorization header missing")
    print("Received request with valid authorization.")

    # A. Ingest and Process the Document
    print(f"Processing document from URL: {req.documents}")
    temp_dir = "temp_docs"
    local_file_path = download_file(req.documents, save_dir=temp_dir)
    
    if not local_file_path:
        raise HTTPException(status_code=400, detail="Could not download or process the document URL.")

    full_text, doc_metadata = extract_text_from_file(local_file_path)
    final_chunks = chunk_and_enrich_text(full_text, doc_metadata)
    
    # B. Index the Chunks in Pinecone
    print(f"Indexing {len(final_chunks)} chunks into Pinecone...")
    namespace = str(uuid.uuid4())
    
    # FIX: Only perform indexing if chunks were actually created
    if final_chunks:
        batch_size = 100
        for i in tqdm(range(0, len(final_chunks), batch_size), desc="Indexing Batches"):
            batch_chunks = final_chunks[i:i + batch_size]
            passage_texts = [chunk['passage_text'] for chunk in batch_chunks]
            
            response = openai_client.embeddings.create(
                model="text-embedding-3-large", input=passage_texts, dimensions=768
            )
            embeddings = [item.embedding for item in response.data]
            
            pinecone_upserts = []
            for j, chunk in enumerate(batch_chunks):
                pinecone_upserts.append({
                    "id": chunk['id'], "values": embeddings[j],
                    "metadata": {
                        "document_name": chunk['document_name'],
                        "page_range": chunk['page_range'],
                        "passage_text": chunk['passage_text']
                    }
                })
            pinecone_index.upsert(vectors=pinecone_upserts, namespace=namespace)

    print(f"Indexing complete for namespace: {namespace}")

    # C. Answer the Questions
    final_answers = []
    print("Answering questions...")
    for question in tqdm(req.questions, desc="Answering Questions"):
        # FIX: Pass the unique namespace to the search function
        matches = find_relevant_chunks(openai_client, pinecone_index, question, top_k=5, namespace=namespace)
        
        if matches:
            answer, _ = generate_answer(openai_client, question, matches)
            final_answers.append(answer.strip())
        else:
            final_answers.append("Could not find relevant information in the document to answer this question.")

    # D. Cleanup
    print("Cleaning up indexed data...")
    try:
        # FIX: Wrap the delete operation in a try/except block
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"Could not delete namespace '{namespace}', it might have been empty. Error: {e}")
        
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    
    print("Processing complete.")
    return RunResponse(answers=final_answers)

# A simple root endpoint to confirm the server is running
@app.get("/")
def read_root():
    return {"status": "Intelligent Query System API is running."}
