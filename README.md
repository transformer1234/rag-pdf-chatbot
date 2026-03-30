# 📄 RAG-based PDF Question Answering System

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-black?logo=groq)
![RAG](https://img.shields.io/badge/architecture-RAG-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B?logo=streamlit)](https://mt-rag-pdf-chatbot.streamlit.app/)

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
python -m streamlit run src/app.py
