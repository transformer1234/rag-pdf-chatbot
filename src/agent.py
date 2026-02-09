from tools import search_docs, answer_with_context, direct_answer
from memory import format_chat_history
from rag_pipeline import generate_llm_response

def agent_decide_and_act(query, chat_history):
    memory = format_chat_history(chat_history)

    # Simple agent reasoning

    intent_prompt = f"""
    User query: "{query}"

    Decide if answering this requires additional context.
    Respond with YES or NO in your response based on whether you need context or not.
    """

    decision = generate_llm_response(intent_prompt).strip().upper()
    print(decision)
    needs_retrieval = "YES" in decision


    if needs_retrieval:
        docs = search_docs(query)
        context = docs["documents"][0]
        print(f"RAG retrieved {len(context)} documents from db.")
        context = "\n\n".join(context)
        return answer_with_context(query, context, memory)
    else:
        return direct_answer(query, memory)
"""
    if any(word in query.lower() for word in ["pdf", "document", "chapter", "explain", "what"]):
        docs = search_docs(query)
        print(f"RAG retrived {len(docs)} element long dict from db.")
        context = docs["documents"][0]
        print(f"RAG retrieved {len(context)} documents from db.")
        context = "\n\n".join(context)
        return answer_with_context(query, context, memory)
    else:
        return direct_answer(query, memory)
"""