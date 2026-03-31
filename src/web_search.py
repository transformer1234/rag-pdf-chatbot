import os
import streamlit as st
from tavily import TavilyClient

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def search_web(query: str, max_results: int = 3) -> tuple[str, list[str]]:
    """
    Search the web using Tavily and return context + sources.

    Returns:
        (context string, list of source URLs)
    """
    try:
        response = tavily_client.search(
            query=query,
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