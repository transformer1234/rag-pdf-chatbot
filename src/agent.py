from tools import search_docs, answer_with_context, direct_answer
from web_search import search_web
from memory import format_chat_history
from rag_pipeline import collection, generate_llm_response


# Queries that should always use web search
WEB_SEARCH_KEYWORDS = [
    "today", "latest", "current", "news", "recent", "now",
    "price", "weather", "score", "live", "2024", "2025", "2026"
]

# Queries that should always use LLM knowledge
LLM_KNOWLEDGE_KEYWORDS = [
    "what is", "define", "explain", "how does", "difference between",
    "tell me about", "describe", "meaning of", "example of"
]


def agent_decide_and_act(query, chat_history):
    memory = format_chat_history(chat_history)
    query_lower = query.lower()

    # Step 1 — current/live info always goes to web search
    if any(kw in query_lower for kw in WEB_SEARCH_KEYWORDS):
        web_context, web_sources = search_web(query)
        if web_context:
            return answer_with_context(query, web_context, web_sources, memory)
        return direct_answer(query, memory)

    # Step 2 — try PDF if docs are indexed
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

    # Step 3 — general knowledge question, use LLM directly
    if any(kw in query_lower for kw in LLM_KNOWLEDGE_KEYWORDS):
        return direct_answer(query, memory)

    # Step 4 — ambiguous, try web search
    web_context, web_sources = search_web(query)
    if web_context:
        return answer_with_context(query, web_context, web_sources, memory)

    # Step 5 — final fallback, LLM own knowledge
    return direct_answer(query, memory)