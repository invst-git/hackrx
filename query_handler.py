# query_handler.py

import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
def initialize_services():
    """Load environment variables and initialize API clients."""
    load_dotenv()
    
    # Load credentials
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([openai_api_key, pinecone_api_key, pinecone_index_name]):
        raise ValueError("One or more environment variables are missing.")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    print(f"Pinecone index '{pinecone_index_name}' initialized.")
    
    return openai_client, index

# --- 2. QUERY PROCESSING ---
def find_relevant_chunks(openai_client, index, user_question: str, top_k: int = 5):
    """Embeds the user question and retrieves the top_k most relevant chunks from Pinecone."""
    print(f"\nEmbedding user question: '{user_question}'")
    
    # a. Create the query embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[user_question],
        dimensions=768
    )
    query_embedding = response.data[0].embedding
    
    # b. Query Pinecone
    print("Searching for relevant chunks in Pinecone...")
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return query_results['matches']


# --- 3. ANSWER SYNTHESIS (RAG) ---
def generate_answer(openai_client, user_question: str, retrieved_chunks: list):
    """Uses GPT-4o to generate an answer based on the user question and retrieved context."""
    print("Synthesizing the final answer...")
    
    # a. Prepare the context
    context = ""
    sources = []
    for i, match in enumerate(retrieved_chunks):
        metadata = match['metadata']
        passage = metadata.get('passage_text', '')
        source_info = f"Source {i+1} (Document: {metadata.get('document_name', 'N/A')}, Page: {metadata.get('page_range', 'N/A')})"
        context += f"--- CONTEXT BLOCK {i+1} ---\n{passage}\n\n"
        sources.append(source_info)

    # b. Create the prompt for the language model
    system_prompt = """
    You are an expert assistant for answering questions about insurance policies.
    You will be given a user's question and a set of context passages retrieved from relevant documents.
    Your task is to synthesize a clear and concise answer based *exclusively* on the provided context.
    Do not use any external knowledge. If the context does not contain the answer, state that the information is not available in the provided documents.
    After providing the answer, list the sources you used from the context.
    """
    
    user_prompt = f"User Question: {user_question}\n\n--- CONTEXT ---\n{context}"
    
    # c. Call the OpenAI Chat Completions API
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2, # Lower temperature for more factual answers
    )
    
    final_answer = response.choices[0].message.content
    return final_answer, sources


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    try:
        # 1. Initialize services
        openai_client, pinecone_index = initialize_services()
        
        # 2. Define the user's question
        question = "What is the policy on free look period?"
        
        # 3. Find relevant document chunks
        matches = find_relevant_chunks(openai_client, pinecone_index, question)
        
        # 4. Generate the final answer
        if matches:
            answer, sources = generate_answer(openai_client, question, matches)
            print("\n" + "="*50)
            print("                FINAL ANSWER")
            print("="*50)
            print(answer)
        else:
            print("Could not find any relevant information in the documents to answer your question.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

