from fastapi import FastAPI
from .rag.pipeline import RAGPipeline
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

pipeline = RAGPipeline()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QuestionRequest):
    answer = pipeline.answer(request.question)
    return {"answer": answer}

app.mount("/", StaticFiles(directory="static", html=True), name="static")