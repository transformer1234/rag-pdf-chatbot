# 📄 RAG-based Question Answering Chatbot

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)
![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-black?logo=groq)
![RAG](https://img.shields.io/badge/architecture-RAG-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B?logo=streamlit)](https://mt-rag-pdf-chatbot.streamlit.app/)

A hybrid RAG (Retrieval-Augmented Generation) chatbot with 3-tier retrieval - ChromaDB vector search, Tavily web search fallback 
and direct LLM response via Groq (llama-3.1-8b-instant) that allows users to upload PDFs/TXTs
and ask questions based on their content.

## 🚀 Features
- PDF/TXT ingestion and chunking
- Huggingface hosted embedding model (all-MiniLM-L6-v2)
- ChromaDB vector database
- Groq hosted LLM (llama-3.1-8b-instant)
- Tavily API for web search
- 3 tier retrieval
- Streamlit UI

## 🛠 Tech Stack
- Python
- Groq Inference API
- Huggingface Inference API
- Tavily API
- ChromaDB
- Streamlit

## 🚀 Live Demo
**[https://mt-rag-pdf-chatbot.streamlit.app/](https://mt-rag-pdf-chatbot.streamlit.app/)**

## ▶️ How to Run
```bash
pip install -r requirements.txt
python -m streamlit run src/app.py
