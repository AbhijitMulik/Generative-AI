import requests
import streamlit as st

def get_ollama_response(input_text: str):
    # Send a POST request to the FastAPI endpoint
    response = requests.post(
        "http://localhost:8000/essay/invoke",  # API endpoint
        json={
            "input": {
                "topic": input_text   # Pass user input inside JSON body
            }
        }
    )

    if response.status_code == 200:
        return response.json().get("output")
    else:
        return f"Error {response.status_code}: {response.text}"

st.title("Langchain Demo With LLAMA2 API")

# Input fields
input_text = st.text_input("Write an essay on")

if input_text:
    st.write(get_ollama_response(input_text)) 