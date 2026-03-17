# 📄 RAG-based PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions based on their content.

## 🚀 Features
- PDF ingestion and chunking
- Semantic search using embeddings
- ChromaDB vector database
- Groq hosted LLM (llama-3.1-8b-instant)
- Conditional retrieval (context used only when needed)
- Streamlit UI

## 🛠 Tech Stack
- Python
- Groq Inference API
- ChromaDB
- Streamlit

## 🚀 Live Demo
**[https://mt-rag-pdf-chatbot.streamlit.app/](https://mt-rag-pdf-chatbot.streamlit.app/)**

## ▶️ How to Run
```bash
pip install -r requirements.txt
cd src
streamlit run app.py
