from rag_pipeline import retrieve_docs, generate_llm_response


def search_docs(query):
    return retrieve_docs(query)


def answer_with_context(query, context, sources, memory):
    sources_text = "\n".join(set(sources))
    prompt = f"""Chat History:
{memory}

Context from uploaded documents:
{context}

Question:
{query}

Answer based on the context above. Be specific and cite relevant details."""
    answer = generate_llm_response(prompt)
    return answer, sources_text


def direct_answer(query, memory):
    prompt = f"""Chat History:
{memory}

Question:
{query}

Answer:"""
    return generate_llm_response(prompt), None