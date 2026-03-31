from tools import search_docs, answer_with_context, direct_answer
from web_search import search_web
from memory import format_chat_history
from rag_pipeline import collection


def agent_decide_and_act(query, chat_history):
    memory = format_chat_history(chat_history)

    # Step 1 — try PDF retrieval if docs are indexed
    if collection.count() > 0:
        retrieval_keywords = [
            "document", "pdf", "according", "mentioned", "says",
            "explain", "what does", "summarize", "chapter", "page",
            "text", "file", "uploaded", "content"
        ]
        needs_retrieval = any(kw in query.lower() for kw in retrieval_keywords)

        if needs_retrieval:
            docs = search_docs(query)
            chunks = docs["documents"][0]
            sources = [m["source"] for m in docs["metadatas"][0]]

            # Check if retrieved chunks are relevant (non-empty)
            if chunks and any(len(c.strip()) > 50 for c in chunks):
                context = "\n\n".join(chunks)
                return answer_with_context(query, context, sources, memory)

    # Step 2 — web search fallback
    web_context, web_sources = search_web(query)
    if web_context:
        return answer_with_context(query, web_context, web_sources, memory)

    # Step 3 — direct LLM answer
    return direct_answer(query, memory)