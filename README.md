# 📄 RAG-based PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions based on their content.

## 🚀 Features
- PDF ingestion and chunking
- Semantic search using embeddings
- ChromaDB vector database
- Hugging Face hosted LLM (Mistral Instruct)
- Conditional retrieval (context used only when needed)
- Streamlit UI

## 🛠 Tech Stack
- Python
- Hugging Face Inference API
- ChromaDB
- Streamlit

## ▶️ How to Run
```bash
pip install -r requirements.txt
cd src
streamlit run app.py
