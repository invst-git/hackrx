# query_handler.py

import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

def initialize_services():
    load_dotenv()
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    return openai_client, index

def find_relevant_chunks(openai_client, index, user_question: str, top_k: int = 5, namespace: str = None):
    """Embeds the user question and retrieves the top_k most relevant chunks from Pinecone."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[user_question],
        dimensions=768
    )
    query_embedding = response.data[0].embedding
    
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return query_results['matches']

def generate_answer(openai_client, user_question: str, retrieved_chunks: list):
    """Uses GPT-4o to generate an answer based on the user question and retrieved context."""
    context = ""
    for i, match in enumerate(retrieved_chunks):
        metadata = match['metadata']
        passage = metadata.get('passage_text', '')
        context += f"--- CONTEXT BLOCK {i+1} ---\n{passage}\n\n"
    
    system_prompt = """
    You are an expert assistant. Synthesize a clear and concise answer based *exclusively* on the provided context.
    Do not use any external knowledge. If the context does not contain the answer, state that the information is not available in the provided documents.
    """
    user_prompt = f"User Question: {user_question}\n\n--- CONTEXT ---\n{context}"
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content, None
