import os

from fastapi import FastAPI , Header, HTTPException
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="RAG API", version="0.1.0")

API_KEY = os.environ.get("APP_API_KEY", "")

class AskRequest(BaseModel):
    question: str 

class AskResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"service": "rag-app", "docs": "/docs", "health": "/health"}
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, x_api_key: str = Header(default="")):   # CHANGE 3a: new parameter
    if not API_KEY or x_api_key != API_KEY:                          # CHANGE 3b: the guard
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return AskResponse(answer=answer(request.question))