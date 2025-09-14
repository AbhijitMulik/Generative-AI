import streamlit as st 
import os 

from langchain_community.document_loaders import PyPDFDirectoryLoader
loader=PyPDFDirectoryLoader("./us_census")
docs=loader.load()
docs













groq_api_key=os.getenv["GROQ_API_KEY"]
from langchain_groq import ChatGroq
model=ChatGroq(groq_api_key=groq_api_key,
               model_name="gamma-9b-it")
