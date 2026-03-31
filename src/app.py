import streamlit as st
from pdf_utils import load_pdf_text, load_txt_text, chunk_text
from rag_pipeline import add_documents, collection
from agent import agent_decide_and_act
from memory import update_memory

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")
st.caption("Upload PDFs/TXTs and ask questions. Powered by ChromaDB + Groq (llama-3.1-8b-instant).")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("Knowledge Base")
pdfs = st.sidebar.file_uploader(
    "Upload PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if pdfs:
    for pdf in pdfs:
        with st.spinner(f"Indexing {pdf.name}...this may take a moment for large files."):
            if pdf.name.endswith(".txt"):
                text = load_txt_text(pdf)
            else:
                text = load_pdf_text(pdf)
            chunks = chunk_text(text)
            add_documents(chunks, pdf.name)
    st.sidebar.success(f"Indexed {len(pdfs)} PDF/TXT(s) successfully!")

# Show indexed docs count
doc_count = collection.count()
st.sidebar.metric("Chunks indexed", doc_count)

if st.sidebar.button("Clear knowledge base"):
    collection.delete(where={"source": {"$ne": ""}})
    st.sidebar.success("Knowledge base cleared.")
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {msg['sources']}")

# Chat input
user_query = st.chat_input("Ask me anything about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = agent_decide_and_act(
                user_query,
                st.session_state.chat_history
            )
        st.write(response)

        if isinstance(sources, tuple):
            source_list, source_label = sources
        else:
            source_list, source_label = None, "llm"

        if source_label == "web" and source_list:
            st.caption(f"🌐 Web: {', '.join(source_list)}")
        elif source_label == "pdf/txt" and source_list:
            st.caption(f"📄 PDF/TXT: {', '.join(source_list)}")
        else:
            st.caption("🧠 LLM knowledge")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": source_list
    })
    update_memory(st.session_state.chat_history, user_query, response)
