# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set env variables explicitly (optional)
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# -----------------------------
# Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Langchain Demo With OLLAMA")
input_text = st.text_input("Search the topic you want")

# -----------------------------
# LLaMA2 with Ollama
# -----------------------------
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Chain = prompt → llm → parser
chain = prompt | llm | output_parser

# -----------------------------
# Run when user enters input
# -----------------------------
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
