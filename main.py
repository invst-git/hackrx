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

from ingestion_pipeline import download_file, extract_text_from_file, chunk_and_enrich_text
from query_handler import find_relevant_chunks, generate_answer

load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, http_request: Request):
    auth_header = http_request.headers.get("Authorization")
    if not auth_header:
         raise HTTPException(status_code=401, detail="Authorization header missing")
    
    namespace = str(uuid.uuid4())
    local_file_path = None
    
    try:
        # A. Ingest and Process
        temp_dir = "temp_docs"
        local_file_path = download_file(req.documents, save_dir=temp_dir)
        if not local_file_path:
            raise HTTPException(status_code=400, detail="Could not download document from URL.")

        full_text, doc_metadata = extract_text_from_file(local_file_path)
        final_chunks = chunk_and_enrich_text(full_text, doc_metadata)
        
        # B. Index Chunks
        if final_chunks:
            print(f"Indexing {len(final_chunks)} chunks into Pinecone namespace: {namespace}")
            batch_size = 100
            for i in tqdm(range(0, len(final_chunks), batch_size), desc="Indexing Batches"):
                batch = final_chunks[i:i + batch_size]
                texts = [c['passage_text'] for c in batch]
                response = openai_client.embeddings.create(model="text-embedding-3-large", input=texts, dimensions=768)
                embeddings = [item.embedding for item in response.data]
                
                vectors = []
                for j, chunk in enumerate(batch):
                    vectors.append({
                        "id": chunk['id'], "values": embeddings[j],
                        "metadata": {
                            "document_name": chunk['document_name'],
                            "page_range": chunk['page_range'],
                            "passage_text": chunk['passage_text']
                        }
                    })
                pinecone_index.upsert(vectors=vectors, namespace=namespace)
            print("Indexing complete.")
        else:
            print("Warning: No text was extracted from the document. No chunks were indexed.")

        # C. Answer Questions
        final_answers = []
        for question in tqdm(req.questions, desc="Answering Questions"):
            if not final_chunks:
                final_answers.append("Could not answer question because the document could not be processed.")
                continue

            matches = find_relevant_chunks(openai_client, pinecone_index, question, top_k=5, namespace=namespace)
            if matches:
                answer, _ = generate_answer(openai_client, question, matches)
                final_answers.append(answer.strip())
            else:
                final_answers.append("Could not find relevant information in the document to answer this question.")
        
        return RunResponse(answers=final_answers)

    finally:
        # D. Cleanup
        print("Cleaning up resources...")
        try:
            pinecone_index.delete(delete_all=True, namespace=namespace)
            print(f"Successfully deleted namespace: {namespace}")
        except Exception as e:
            print(f"Could not delete namespace '{namespace}', it might have been empty or already deleted. Error: {e}")
        
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"Successfully deleted temporary file: {local_file_path}")

@app.get("/")
def read_root():
    return {"status": "Intelligent Query System API is running."}
