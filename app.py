import streamlit as st
import os
from config.config import config
from models.llm import get_chatgroq_model
from models.embeddings import get_embedding_model
from utils.rag_tools import create_rag_retriever, get_rag_tool
from utils.web_tools import get_web_search_tool


def chat_page():
    st.title("🤖 NeoStats AI Chatbot")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    groq_key = st.sidebar.text_input("Groq API Key", type="password", value=config.GROQ_API_KEY or "")
    tavily_key = st.sidebar.text_input("Tavily API Key (optional)", type="password", value=config.TAVILY_API_KEY or "")

    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"),
        index=0
    )

    response_mode = st.sidebar.radio("Response Mode", ["Concise", "Detailed"], index=1)
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF for RAG", type=["pdf"])

    # Clear Chat Button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Initialize session memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Stop if no key entered
    if not groq_key:
        st.warning("⚠️ Please enter your Groq API key to continue.")
        st.stop()

    # Initialize Models
    chat_model = get_chatgroq_model(groq_key, model_name)
    embed_model = get_embedding_model()
    tools = []

    # RAG setup (PDF Upload)
    if uploaded_pdf:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        retriever = create_rag_retriever(pdf_path, embed_model)
        rag_tool = get_rag_tool(retriever)
        tools.append(rag_tool)

    # Web Search setup (optional)
    if tavily_key:
        web_tool = get_web_search_tool(tavily_key)
        tools.append(web_tool)

    # Chat Input
    prompt = st.chat_input("Ask your question...")

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = ""
                for t in tools:
                    try:
                        context_piece = t(prompt)
                        if context_piece:
                            context += context_piece + "\n"
                    except Exception as e:
                        context += f"\n[Error using tool: {e}]"

                system_prompt = "Be helpful, accurate, and polite."
                if response_mode == "Concise":
                    system_prompt += " Keep answers brief and to the point."
                else:
                    system_prompt += " Give detailed, structured responses."

                try:
                    response = chat_model.generate(prompt, context, system_prompt)
                except Exception as e:
                    response = f"❌ Error calling Groq model: {e}"

                st.markdown(response)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="NeoStats AI Chatbot", layout="wide")
    chat_page()


if __name__ == "__main__":
    main()
