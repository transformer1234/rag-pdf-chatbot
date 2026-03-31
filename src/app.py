import streamlit as st
from pdf_utils import load_pdf_text, chunk_text
from rag_pipeline import add_documents, collection
from agent import agent_decide_and_act
from memory import update_memory

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("RAG PDF Chatbot")
st.caption("Upload PDFs and ask questions. Powered by ChromaDB + Groq (llama-3.1-8b-instant).")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("Knowledge Base")
pdfs = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if pdfs:
    for pdf in pdfs:
        with st.spinner(f"Indexing {pdf.name}..."):
            text = load_pdf_text(pdf)
            chunks = chunk_text(text)
            add_documents(chunks, pdf.name)
    st.sidebar.success(f"Indexed {len(pdfs)} PDF(s) successfully!")

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
        if sources:
            if sources[0].startswith("http"):
                st.caption(f"🌐 Web: {sources}")
            else:
                st.caption(f"📄 PDF: {sources}")
        else:
            st.caption("🧠 LLM knowledge")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    update_memory(st.session_state.chat_history, user_query, response)
