from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn

from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version=1.0,
    description="A simple API server"

    )

llm=Ollama(model="llama2")
prompt=ChatPromptTemplate.from_template("write an essay about {topic} wiith 250 words")

add_routes(
     app,
    prompt|llm,
    path="/essay"
)
   
    

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)