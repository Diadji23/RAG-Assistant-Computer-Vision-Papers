import os
from dotenv import load_dotenv

load_dotenv(".env.local")

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.embedder import MyEmbeddings
from src.retriever import Retriever
from src.llm import OllamaLLM
from src.pipeline import RAGPipeline

rag = None  # variable globale


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    # Initialisation au démarrage
    if os.getenv("PROVIDER", "ollama") == "azure":
        from src.azure_embedder import AzureEmbeddings
        from src.azure_llm import AzureLLM
        embedder, llm = AzureEmbeddings(), AzureLLM()
    else:
        embedder, llm = MyEmbeddings(), OllamaLLM()
    retriever = Retriever(embedder)
    rag = RAGPipeline(retriever=retriever, llm=llm)
    yield  # l'app tourne ici


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = rag.run(request.question)
    return answer