from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import uvicorn
import logging

# Import the RAG pipeline generating function
from query import generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="BIS Chatbot API", description="API for answering questions about the BIS website via RAG")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to the frontend domain
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
    message = request.message
    session_id = request.session_id
    
    # Generate new session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        
    try:
        logger.info(f"Received query: {message} for session: {session_id}")
        
        # Simple out-of-scope check before even embedding
        lower_message = message.lower()
        strict_keywords = ['bis', 'bureau', 'indian standard', 'hallmark', 'certification', 'isi']
        
        # We'll rely on the LLM to strictly follow the prompt, 
        # but here is a simple pre-check example.
        # This is commented out to allow the LLM to handle rejections gracefully
        # if not any(kw in lower_message for kw in strict_keywords) and len(lower_message) > 100:
        #    return generate_answer(message, session_id) # Process anyway to let LLM decide
            
        result = generate_answer(message, session_id)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing request")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
