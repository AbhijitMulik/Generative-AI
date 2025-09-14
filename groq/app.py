import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ==============================
# Load environment variables
# ==============================
load_dotenv()

# Load Groq API key
 
if not groq_api_key:
    st.error("⚠️ Please set your GROQ_API_KEY in the .env file")
    st.stop()

# ==============================
# Streamlit UI
# ==============================
st.title("🤖 Website Q&A with Groq + HuggingFace")

# User enters website link
website_url = st.text_input("🌐 Enter website URL:", placeholder="https://example.com")

# Load and process website only when link is provided
if website_url:
    if "vectors" not in st.session_state or st.session_state.get("loaded_url") != website_url:
        with st.spinner("🔄 Loading and processing website..."):
            # Save the URL to session_state (so we don’t reload unnecessarily)
            st.session_state.loaded_url = website_url

            # Use HuggingFace Embeddings
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load website
            loader = WebBaseLoader(website_url)
            docs = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            final_documents = text_splitter.split_documents(docs[:10])  # process first 10 docs

            # Create FAISS vector DB
            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

    # Groq LLM (smaller model to avoid memory issue)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it"
    )

    # Prompt Template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Provide the most accurate and concise response.
        <context>
        {context}
        <context>
        Question: {input}
        """
    )

    # Create QA chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ==============================
    # User Query
    # ==============================
    user_input = st.text_input("💬 Ask a question:")

    if user_input:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        elapsed = time.process_time() - start

        st.subheader("🧠 Answer")
        st.write(response["answer"])
        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        # Optional: show retrieved docs
        with st.expander("📑 Relevant Document Chunks"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
else:
    st.info("👆 Please enter a website URL to get started.")
