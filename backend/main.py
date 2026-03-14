import os
import sys
import uuid
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Print immediately so Render knows we're alive
print("=== BIS Chatbot Backend Starting ===", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"PORT env: {os.getenv('PORT', 'not set')}", flush=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="BIS Chatbot API")

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Lazy import - only loads query.py (and its heavy deps) on first request
    from query import generate_answer
    
    message = request.message
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Received query: {message} for session: {session_id}")
        result = generate_answer(message, session_id)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"status": "BIS Chatbot API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting uvicorn on port {port}...", flush=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
