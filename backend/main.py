import os
import sys
import uuid
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image_endpoint(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """Accept an image + question and return a Gemini-generated answer."""
    from gemini_vision import analyze_image_with_gemini

    session_id = session_id or str(uuid.uuid4())

    try:
        logger.info(
            "Received image query: '%s' (file=%s, type=%s) for session: %s",
            message,
            image.filename,
            image.content_type,
            session_id,
        )
        image_bytes = await image.read()
        mime_type = image.content_type
        if not mime_type:
            raise ValueError("Could not determine image MIME type. Please upload a valid image file.")
        answer = analyze_image_with_gemini(image_bytes, mime_type, message)
        return ChatResponse(answer=answer, sources=[], session_id=session_id)
    except ValueError as exc:
        logger.warning("Image validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Gemini error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in chat-with-image endpoint: %s", exc)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
async def root():
    return {"status": "BIS Chatbot API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting uvicorn on port {port}...", flush=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
