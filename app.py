from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="RAG API", version="0.1.0")

class AskRequest(BaseModel):
    question: str 

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return AskResponse(answer=answer(request.question))