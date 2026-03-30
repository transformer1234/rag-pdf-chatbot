import os
import chromadb
import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient
from config import LLM_MODEL, EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION, MAX_TOKENS, TOP_K_RESULTS
from dotenv import load_dotenv

load_dotenv()

# Persistent ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

# HuggingFace Inference API for embeddings
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

# Groq client
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = hf_client.feature_extraction(
        texts,
        model=EMBEDDING_MODEL
    )
    return embeddings.tolist()


def add_documents(chunks, source):
    embeddings = get_embeddings(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source}_{i}"],
            metadatas=[{"source": source}]
        )


def retrieve_docs(query, k=TOP_K_RESULTS):
    query_embedding = get_embeddings([query])
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
