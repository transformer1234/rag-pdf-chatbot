from rag_pipeline import retrieve_docs, generate_llm_response

#sources = results["metadatas"][0]

def search_docs(query):
    return retrieve_docs(query)


def answer_with_context(query, context, memory):
    prompt = f"""

Chat History:
{memory}

Context:
{context}

Question:
{query}

Answer:
"""
    return generate_llm_response(prompt)


def direct_answer(query, memory):
    prompt = f"""

Chat History:
{memory}

Question:
{query}

Answer:
"""
    return generate_llm_response(prompt)
