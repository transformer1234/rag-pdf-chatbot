import os
import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from config import LLM_MODEL, EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION, MAX_TOKENS, TOP_K_RESULTS
from dotenv import load_dotenv

load_dotenv()

# Persistent ChromaDB client — survives app restarts
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

# Explicit embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Groq client
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


def add_documents(chunks, source):
    embeddings = embedding_model.encode(chunks).tolist()
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source}_{i}"],
            metadatas=[{"source": source}]
        )


def retrieve_docs(query, k=TOP_K_RESULTS):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results


def generate_llm_response(prompt):
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content