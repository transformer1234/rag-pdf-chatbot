import os
import re
import streamlit as st
from tavily import TavilyClient

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Strip document-specific phrases before web searching
DOCUMENT_PHRASES = [
    "in the document", "in the pdf", "according to the document",
    "what does the document say", "mentioned in", "in the text",
    "in the file", "uploaded", "from the pdf"
]


def clean_query(query: str) -> str:
    q = query.lower()
    for phrase in DOCUMENT_PHRASES:
        q = q.replace(phrase, "")
    return q.strip()


def search_web(query: str, max_results: int = 3) -> tuple[str, list[str]]:
    try:
        cleaned = clean_query(query)
        response = tavily_client.search(
            query=cleaned,
            search_depth="basic",
            max_results=max_results
        )
        results = response.get("results", [])
        if not results:
            return "", []

        context = "\n\n".join([
            f"{r['title']}:\n{r['content']}"
            for r in results
        ])
        sources = [r["url"] for r in results]
        return context, sources

    except Exception as e:
        return "", []