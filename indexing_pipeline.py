# indexing_pipeline.py

import os
import openai
from pinecone import Pinecone
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from tqdm import tqdm
import uuid

# Import the ingestion function from your first script
from ingestion_pipeline import run_ingestion_from_local_paths

# --- 1. INITIALIZATION ---
def initialize_services():
    """Load environment variables and initialize API clients."""
    load_dotenv()
    
    # Load credentials from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    database_url = os.getenv("DATABASE_URL")

    if not all([openai_api_key, pinecone_api_key, pinecone_index_name, pinecone_environment, database_url]):
        raise ValueError("One or more environment variables are missing. Please check your .env file.")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    # Note: Pinecone environment is used when connecting to the index, not during init.
    index = pc.Index(pinecone_index_name)
    print(f"Pinecone index '{pinecone_index_name}' initialized. Stats: {index.describe_index_stats()}")

    # Initialize PostgreSQL connection
    pg_conn = psycopg2.connect(database_url)
    print("PostgreSQL connection established.")
    
    return openai_client, index, pg_conn

# --- 2. POSTGRESQL SETUP ---
def setup_postgres_table(conn):
    """Ensure the audit table exists in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY,
                document_name VARCHAR(255),
                page_range VARCHAR(50),
                clause_section_heading TEXT,
                passage_text TEXT,
                ingested_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()
    print("PostgreSQL 'document_chunks' table verified.")

# --- 3. EMBEDDING AND UPSERTING ---
def create_embeddings_and_upsert(openai_client, chunks: list[dict], index, pg_conn):
    """
    Generates embeddings for chunks and upserts them to Pinecone and PostgreSQL in batches.
    """
    batch_size = 100 # Process 100 chunks at a time to avoid API limits
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and Upserting Batches"):
        batch_chunks = chunks[i:i + batch_size]
        
        passage_texts = [chunk['passage_text'] for chunk in batch_chunks]
        
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=passage_texts,
            dimensions=768
        )
        embeddings = [item.embedding for item in response.data]
        
        pinecone_upserts = []
        postgres_records = []
        
        for j, chunk in enumerate(batch_chunks):
            chunk_uuid = uuid.UUID(chunk['id'].replace('chunk_', ''))
            
            pinecone_upserts.append({
                "id": chunk['id'],
                "values": embeddings[j],
                "metadata": {
                    "document_name": chunk['document_name'],
                    "page_range": chunk['page_range'],
                    "passage_text": chunk['passage_text']
                }
            })
            
            # --- THE FIX IS HERE ---
            # We now convert the UUID object back to a string before sending it to the database.
            postgres_records.append((
                str(chunk_uuid), # <-- Changed chunk_uuid to str(chunk_uuid)
                chunk['document_name'],
                chunk['page_range'],
                chunk['clause_section_heading'],
                chunk['passage_text']
            ))

        # Upsert to Pinecone
        index.upsert(vectors=pinecone_upserts)
        
        # Persist metadata in PostgreSQL
        with pg_conn.cursor() as cur:
            execute_batch(cur, 
                          "INSERT INTO document_chunks (id, document_name, page_range, clause_section_heading, passage_text) VALUES (%s, %s, %s, %s, %s)",
                          postgres_records)
        pg_conn.commit()

    print(f"Successfully upserted {len(chunks)} chunks to Pinecone and PostgreSQL.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    
    # 1. Run Task 1: Ingestion from local files
    print("--- Starting Task 1: Document Ingestion ---")
    local_docs_dir = "document_storage"
    supported_extensions = ['.pdf', '.docx', '.eml']
    
    if not os.path.isdir(local_docs_dir):
        print(f"Error: The directory '{local_docs_dir}' was not found. Please create it and add your documents.")
    else:
        local_files_to_process = [
            os.path.join(local_docs_dir, f) 
            for f in os.listdir(local_docs_dir) 
            if any(f.lower().endswith(ext) for ext in supported_extensions)
        ]
    
        if not local_files_to_process:
            print(f"No documents found in '{local_docs_dir}'. Please add your files and run again.")
        else:
            final_chunks = run_ingestion_from_local_paths(local_files_to_process)
            
            if final_chunks:
                print("\n--- Starting Task 2: Embedding & Indexing ---")
                
                try:
                    # 2. Initialize all services
                    openai_client, pinecone_index, pg_connection = initialize_services()
                    
                    # 3. Ensure Postgres table exists
                    setup_postgres_table(pg_connection)
                    
                    # 4. Run the embedding and upsert process
                    create_embeddings_and_upsert(openai_client, final_chunks, pinecone_index, pg_connection)
                    
                    # 5. Clean up connections
                    if pg_connection:
                        pg_connection.close()
                    print("\n--- INDEXING COMPLETE ---")
                    
                except Exception as e:
                    print(f"\nAn error occurred during the indexing process: {e}")

