from rag_pipeline import retrieve_docs, generate_llm_response


def search_docs(query):
    return retrieve_docs(query)


def answer_with_context(query, context, sources, memory):
    # Detect source type
    is_web = any(s.startswith("http") for s in sources)

    if is_web:
        system_context = "web search results"
        source_label = "web"
        unique_sources = list(dict.fromkeys(sources))  # preserve order, dedupe
    else:
        system_context = "uploaded documents"
        source_label = "pdf"
        unique_sources = list(set(sources))  # dedupe filenames

    prompt = f"""Chat History:
{memory}

Context from {system_context}:
{context}

Question:
{query}

Answer based on the context above. Be specific and accurate."""

    answer = generate_llm_response(prompt)
    return answer, (unique_sources, source_label)


def direct_answer(query, memory):
    prompt = f"""Chat History:
{memory}

Question:
{query}

Answer:"""
    return generate_llm_response(prompt), (None, "llm")