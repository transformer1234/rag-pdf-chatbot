import streamlit as st
from pdf_utils import load_pdf_text, chunk_text
from rag_pipeline import add_documents
from agent import agent_decide_and_act
from memory import update_memory

st.set_page_config(page_title="AI Agent RAG", layout="wide")
st.title("🤖 AI Agent – RAG PDF Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar PDF Upload
st.sidebar.header("Upload PDF")
pdfs = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdfs:
    for pdf in pdfs:
        text = load_pdf_text(pdf)
        chunks = chunk_text(text)
        add_documents(chunks, pdf.name)
    st.sidebar.success("PDF indexed successfully!")

# Chat Input
user_query = st.chat_input("Ask me anything...")

if user_query:
    response = agent_decide_and_act(
        user_query,
        st.session_state.chat_history
    )

    update_memory(st.session_state.chat_history, user_query, response)

    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write(response)

# Show memory
with st.expander("🧠 Agent Memory"):
    for msg in st.session_state.chat_history:
        st.write(msg)
