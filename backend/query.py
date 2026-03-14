import os
import logging
import traceback
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "bis_knowledge"

# ── Lazy-loaded globals (initialized on first request, not at import time) ──
# Heavy libraries (PyTorch, SentenceTransformers) are imported INSIDE this function
# so the server can start and bind the port IMMEDIATELY on Render.
_embedder = None
_qdrant = None
_groq_client = None

def _get_models():
    """Lazy-initialize heavy models on first call."""
    global _embedder, _qdrant, _groq_client
    if _embedder is None:
        logger.info("Loading FastEmbed model (lightweight ONNX)...")
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Embedding model loaded.")
    if _qdrant is None:
        logger.info("Connecting to Qdrant DB...")
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(path="./qdrant_db")
        logger.info("Qdrant connected.")
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. LLM queries will fail.")
        _groq_client = Groq(api_key=api_key)
    return _embedder, _qdrant, _groq_client

# Simple in-memory memory management (session_id -> list of messages)
# In production, use Redis or a database
conversation_memory: Dict[str, List[Dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Optimized for hackathon evaluation robustness
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the official AI assistant for the Bureau of Indian Standards (BIS) website (https://www.bis.gov.in).
Your sole purpose is to answer questions about BIS using ONLY the retrieved context provided to you.

═══════════════════════════════════════════
RESPONSE STYLE — NATURAL & ADAPTIVE
═══════════════════════════════════════════

Write naturally like a knowledgeable human expert — NOT like a template.
NEVER use rigid section labels such as "[Direct Answer]", "[Explanation]", or section headers in brackets.
Instead, write a flowing, informative response that reads like a well-written article.

Adapt your format to the TYPE of question:

• "What is…" / "Tell me about…" → Start with a clear 1-2 sentence definition, then expand with a short paragraph of explanation. Bold key terms.
• "How do I…" / "Steps to…" / "Process for…" → Provide a brief intro sentence, then a **numbered step-by-step list** using 1. 2. 3. format (NOT bullet points). Each step must be a clear action.
• "Types of…" / "Schemes" / "List…" / "Examples of…" → Provide a brief intro, then a **bullet-point list** with each item bolded and briefly described.
• "Difference between…" / "Compare…" → Explain each item briefly, then highlight the key differences clearly.
• "Yes/No" questions → Start with a clear Yes or No, then explain why.
• Follow-up questions ("Tell me more", "How do I apply for that?") → Use the conversation history to resolve what "that" refers to, and answer seamlessly.
• Multi-part questions → Answer each part under a short bold heading (not bracketed labels).

═══════════════════════════════════════════
CONTENT RULES
═══════════════════════════════════════════

1. CONTEXT IS YOUR ONLY SOURCE OF TRUTH
   • Extract and present ALL relevant information from the retrieved documents below.
   • Synthesize information from multiple documents when needed.
   • NEVER say "the text does not explicitly mention…" if any relevant detail exists. Use what's available.

2. BE SPECIFIC AND USEFUL
   • NEVER produce vague sentences like "You can follow the steps on the BIS website" or "Please visit the BIS portal for more details."
   • Instead, extract the actual steps, names, schemes, or details from the context and explain them.
   • If the context mentions a process but lacks full detail, explain what IS available and note: "For complete details and forms, visit https://www.bis.gov.in"

3. NEVER HALLUCINATE
   • Do NOT invent facts, statistics, dates, fee amounts, URLs, form names, or scheme names not in the context.
   • If context is insufficient, say: "The retrieved BIS website content mentions [topic] but does not provide [specific detail]. You can find more at https://www.bis.gov.in"
   • Then provide the CLOSEST relevant information you DO have.

4. READABILITY
   • Use **bold** for key terms, scheme names, and important phrases.
   • Use short paragraphs (2-3 sentences max per paragraph).
   • Use bullet points for lists of 3+ items.
   • Use numbered lists for sequential steps/procedures.
   • Keep the total response concise but complete — aim for 150-300 words.

5. SOURCES — CRITICAL RULE
   • Do NOT embed any URLs or links inside the main answer text.
   • Do NOT write a "Sources:" section yourself.
   • Do NOT add "Note: visit https://..." or "For more details, see https://..." at the end.
   • Do NOT include ANY string starting with "http" anywhere in your response.
   • The frontend will automatically display source links below your answer.
   • Your response should contain ONLY the answer content, with ZERO URLs.

6. OUT-OF-SCOPE QUESTIONS
   If the user asks about topics completely unrelated to BIS (stocks, weather, sports, other companies, recipes, coding, etc.), respond EXACTLY with:
   "I am a BIS website assistant and can only answer questions related to the Bureau of Indian Standards, BIS certifications, standards, consumer programs, laboratories, and BIS services.

   Here are some example questions I can help with:
   • What is the ISI mark?
   • How do I apply for BIS certification?
   • What is the BIS hallmarking scheme?
   • What are BIS laboratories?"

7. MULTI-TURN CONVERSATION
   • Use the Previous Conversation History to understand follow-up questions.
   • If the user says "Tell me more" or "How do I apply for that?", resolve the reference using history.

8. AMBIGUOUS QUESTIONS
   • If genuinely unclear, ask ONE short clarifying question. Example: "Are you asking about BIS product certification or hallmarking?"

9. PARAPHRASED QUESTIONS
   Recognise equivalent phrasings:
   "BIS schemes" = "types of BIS certification" = "BIS approvals" = "certification programs"

10. KNOWLEDGE AREAS (answer confidently when context supports):
    BIS overview & functions • ISI Mark (Product Certification) • Compulsory Registration Scheme (CRS) • Hallmarking (gold/silver) • Foreign Manufacturers Certification (FMCS) • Standards development • BIS laboratories • Consumer programs • Grievance redressal • Application procedures • Benefits of certification

═══════════════════════════════════════════
RETRIEVED CONTEXT DOCUMENTS
═══════════════════════════════════════════
{context}

═══════════════════════════════════════════
PREVIOUS CONVERSATION HISTORY
═══════════════════════════════════════════
{history}
"""

# Out-of-scope keywords for pre-filtering
OUT_OF_SCOPE_KEYWORDS = [
    'stock price', 'share price', 'weather', 'cricket', 'football',
    'movie', 'recipe', 'song', 'lyrics', 'joke', 'poem',
    'nifty', 'sensex', 'bitcoin', 'crypto', 'forex',
    'write code', 'python code', 'javascript', 'html code',
    'who is the president of usa', 'capital of',
]

OUT_OF_SCOPE_RESPONSE = """I am a BIS website assistant and can only answer questions related to the Bureau of Indian Standards, BIS certifications, standards, consumer programs, laboratories, and BIS services.

Here are some example questions I can help with:
• What is the ISI mark?
• How do I apply for BIS certification?
• What is the BIS hallmarking scheme?
• What are BIS laboratories?
• What is the Compulsory Registration Scheme?"""


def is_out_of_scope(query: str) -> bool:
    """Quick pre-check for obviously out-of-scope questions."""
    query_lower = query.lower().strip()
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


def retrieve_context(query: str, session_id: str = "", top_k: int = 5) -> Tuple[str, List[str]]:
    """Retrieve relevant chunks from Qdrant based on the query.
    
    Uses conversation history to enrich the query for better retrieval
    on follow-up questions.
    """
    try:
        # Enrich the query with conversation context for follow-up questions
        enriched_query = query
        if session_id and session_id in conversation_memory:
            history = conversation_memory[session_id]
            if len(history) >= 2:
                # Append the last assistant response topic to help retrieval
                last_assistant_msg = history[-1].get("content", "")
                # Take first 200 chars of last response as context hint
                context_hint = last_assistant_msg[:200]
                enriched_query = f"{query} {context_hint}"
        
        # Convert query to vector
        embedder, qdrant, _ = _get_models()
        query_vector = list(embedder.embed([enriched_query]))[0].tolist()
        
        # Search Qdrant (using query_points for qdrant-client >= 1.12)
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        
        points = search_result.points
        if not points:
            return "No relevant context found in the BIS website database.", []
            
        context_parts = []
        unique_urls = set()
        
        for i, hit in enumerate(points):
            payload = hit.payload
            text = payload.get("text", "")
            url = payload.get("url", "")
            title = payload.get("title", "")
            
            context_parts.append(
                f"--- Document {i+1} ---\n"
                f"Title: {title}\n"
                f"Source URL: {url}\n"
                f"Content: {text}\n"
            )
            unique_urls.add(url)
            
        return "\n".join(context_parts), list(unique_urls)
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}\n{traceback.format_exc()}")
        return "", []


def get_chat_history_string(session_id: str) -> str:
    """Format chat history into a string for the prompt."""
    history = conversation_memory.get(session_id, [])
    if not history:
        return "No previous conversation."
        
    # Keep the last 10 messages (5 turns) for better multi-turn support
    recent_history = history[-10:]
    
    formatted = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
        
    return "\n".join(formatted)


def generate_answer(query: str, session_id: str) -> Dict[str, Any]:
    """Complete RAG pipeline to generate an answer.
    
    Pipeline:
    1. Pre-filter obvious out-of-scope questions
    2. Embed query (enriched with conversation context) and retrieve documents
    3. Build structured prompt with system instructions + context + history
    4. Call LLM via Groq
    5. Update conversation memory
    """
    
    # Step 0: Pre-flight check — Is Groq API configured?
    if not os.getenv("GROQ_API_KEY"):
        return {
            "answer": "Error: Groq API key is not configured. Please set the GROQ_API_KEY environment variable.",
            "sources": []
        }
    
    # Step 1: Quick out-of-scope check
    if is_out_of_scope(query):
        # Still update memory so user can ask BIS questions next
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        conversation_memory[session_id].append({"role": "user", "content": query})
        conversation_memory[session_id].append({"role": "assistant", "content": OUT_OF_SCOPE_RESPONSE})
        return {
            "answer": OUT_OF_SCOPE_RESPONSE,
            "sources": []
        }
    
    # Step 2: Embed query and retrieve context (enriched with history for follow-ups)
    context, sources = retrieve_context(query, session_id=session_id)
    
    # Step 3: Get conversation history
    history_str = get_chat_history_string(session_id)
    
    # Step 4: Build the system prompt
    system_message = SYSTEM_PROMPT.format(context=context, history=history_str)
    
    # Step 5: Call LLM
    try:
        _, _, groq_client = _get_models()
        response = groq_client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=0.2,       # Low temperature for factual accuracy
            max_tokens=1500,       # Allow longer responses for detailed answers
            top_p=0.9,
        )
        
        answer = response.choices[0].message.content
        
        # Step 6: Update Memory
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
            
        conversation_memory[session_id].append({"role": "user", "content": query})
        conversation_memory[session_id].append({"role": "assistant", "content": answer})
        
        # Trim memory if it grows too large (keep last 20 messages = 10 turns)
        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-20:]
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}\n{traceback.format_exc()}")
        return {
            "answer": "Sorry, I encountered an error while generating a response. Please try again later.",
            "sources": []
        }
