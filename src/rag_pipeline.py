import chromadb
import os
from dotenv import load_dotenv
#from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from config import *

#embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory=CHROMA_PATH
    )
)

collection = client.get_or_create_collection("rag_docs")

load_dotenv()   # Load environment variables from .env file

llm = InferenceClient(
    model=LLM_MODEL,
    token=os.getenv("HF_TOKEN")
)


def add_documents(chunks, source):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{source}_{i}"],
            metadatas=[{"source": source}]
        )


def retrieve_docs(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents","metadatas"]
    )
    return results


def generate_llm_response(prompt):
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    answer = response.choices[0].message["content"]
    return answer
