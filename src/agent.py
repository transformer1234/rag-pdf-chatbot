from tools import search_docs, answer_with_context, direct_answer
from web_search import search_web
from memory import format_chat_history
from rag_pipeline import collection, generate_llm_response

WEB_SEARCH_KEYWORDS = [
    "today", "latest", "current", "news", "recent", "now",
    "price", "weather", "score", "live", "2024", "2025", "2026"
]

LLM_KNOWLEDGE_KEYWORDS = [
    "what is", "define", "explain", "how does", "difference between",
    "tell me about", "describe", "meaning of", "example of"
]

DOCUMENT_KEYWORDS = [
    "document", "pdf", "txt", "uploaded", "file", "text", "chapter",
    "page", "according to", "mentioned", "says", "content"
]


def agent_decide_and_act(query, chat_history):
    memory = format_chat_history(chat_history)
    query_lower = query.lower()

    is_document_query = any(kw in query_lower for kw in DOCUMENT_KEYWORDS)
    is_web_query = any(kw in query_lower for kw in WEB_SEARCH_KEYWORDS)
    is_llm_query = any(kw in query_lower for kw in LLM_KNOWLEDGE_KEYWORDS)

    # Step 1 — document query → only use PDF, never web search
    if is_document_query:
        if collection.count() > 0:
            docs = search_docs(query)
            chunks = docs["documents"][0]
            sources = [m["source"] for m in docs["metadatas"][0]]
            context = "\n\n".join(chunks)
            return answer_with_context(query, context, sources, memory)
        return direct_answer(query, memory)

    # Step 2 — current/live info → web search only
    if is_web_query:
        web_context, web_sources = search_web(query)
        if web_context:
            return answer_with_context(query, web_context, web_sources, memory)
        return direct_answer(query, memory)

    # Step 3 — try PDF if docs indexed and relevant
    if collection.count() > 0:
        docs = search_docs(query)
        chunks = docs["documents"][0]
        sources = [m["source"] for m in docs["metadatas"][0]]
        context = "\n\n".join(chunks)

        relevance_prompt = f"""Context:
{context[:500]}

Question: "{query}"
Is this context relevant to answer the question? Reply YES or NO only."""
        relevance = generate_llm_response(relevance_prompt).strip().upper()

        if "YES" in relevance:
            return answer_with_context(query, context, sources, memory)

    # Step 4 — general knowledge → LLM direct
    if is_llm_query:
        return direct_answer(query, memory)

    # Step 5 — ambiguous → web search
    web_context, web_sources = search_web(query)
    if web_context:
        return answer_with_context(query, web_context, web_sources, memory)

    # Step 6 — final fallback
    return direct_answer(query, memory)