from tools import search_docs, answer_with_context, direct_answer
from memory import format_chat_history


def agent_decide_and_act(query, chat_history):
    memory = format_chat_history(chat_history)

    # Keyword-based intent detection — more reliable than LLM YES/NO
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
        context = "\n\n".join(chunks)
        return answer_with_context(query, context, sources, memory)
    else:
        return direct_answer(query, memory)