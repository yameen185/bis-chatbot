# BIS RAG Chatbot — Architecture & Workflow

## Overview

This is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the **Bureau of Indian Standards (BIS)** website. It crawls [bis.gov.in](https://www.bis.gov.in), stores page content as vector embeddings, and uses an LLM to generate answers grounded in the retrieved context.

## Architecture Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  BIS Website │────▶│  crawler.py  │────▶│ scraped_data │
│  bis.gov.in  │     │  (scraping)  │     │    .json     │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  ingest.py   │
                                        │ (embed+store)│
                                        └──────┬───────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Frontend   │────▶│   main.py    │────▶│  query.py    │
│ (Vercel)    │◀────│  (FastAPI)   │◀────│  (RAG logic) │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                             ┌────────────┐          ┌────────────┐
                             │  Qdrant DB │          │  Groq API  │
                             │ (vectors)  │          │   (LLM)    │
                             └────────────┘          └────────────┘
```

## Data Pipeline (Run Once)

These scripts run **locally** to prepare the data. You don't run them on the server.

### Step 1: `backend/crawler.py` — Web Scraper

- Crawls pages from `bis.gov.in` using `requests` + `BeautifulSoup`
- Extracts page title, URL, and text content
- **Output:** `scraped_data.json` — array of `{url, title, content}` objects

### Step 2: `backend/ingest.py` — Embedding & Storage

- Reads `scraped_data.json`
- Splits content into overlapping chunks (~500 words each)
- Generates vector embeddings using **FastEmbed** (ONNX-based, lightweight)
- Stores vectors + metadata in a local **Qdrant** database
- **Output:** `qdrant_db/` folder (SQLite-based vector database)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimension vectors)

## Backend (FastAPI)

### `backend/main.py` — API Server

- **Framework:** FastAPI with Uvicorn
- **Endpoints:**
  - `POST /chat` — Receives a question, returns an answer + sources
  - `GET /health` — Health check
  - `GET /` — Root status check
- **Important:** `query.py` is imported **lazily** inside the `/chat` endpoint (not at module level) to ensure the server binds its port immediately on Render's free tier
- **CORS:** Allows all origins (`*`)

### `backend/query.py` — RAG Pipeline (Core Logic)

This is the brain of the chatbot. It handles:

1. **Lazy Model Loading** (`_get_models()`)
   - Loads FastEmbed, Qdrant client, and Groq client on first request
   - Keeps them cached in global variables for subsequent requests

2. **Context Retrieval** (`retrieve_context()`)
   - Enriches follow-up questions with conversation history
   - Converts query to a 384-dim vector using FastEmbed
   - Searches Qdrant for the top-5 most similar document chunks
   - Returns the matched text chunks + source URLs

3. **System Prompt**
   - Adaptive formatting based on question type (what/how/list/compare)
   - Strict anti-hallucination rules
   - Out-of-scope detection (non-BIS questions are rejected)
   - Source URLs are NOT included in the answer text (frontend handles them)

4. **Answer Generation** (`generate_answer()`)
   - Builds the prompt with retrieved context + conversation history
   - Calls **Groq API** (`llama-3.1-8b-instant` model)
   - Manages per-session conversation memory (last 10 turns)
   - Returns `{answer, sources}`

## Frontend

### `frontend/index.html` — Chat Interface (Single File)

A self-contained HTML/CSS/JS chat UI with:

- **Dark/Light mode** toggle (persisted in localStorage)
- **Suggestion chips** for common questions
- **Source deduplication** — URLs stripped from answer text, shown as clickable chips
- **Markdown rendering** — bold, lists, headings
- **Copy button** on each bot response
- **Typing indicator** animation while waiting for response
- **Responsive layout** for mobile/tablet/desktop
- **API URL config** — set `RENDER_BACKEND_URL` at the top of the `<script>` section

### `frontend/vercel.json` — Vercel Config

SPA routing — serves `index.html` for all routes.

## Configuration Files

| File | Purpose |
|------|---------|
| `backend/.env` | `GROQ_API_KEY` — required for LLM calls |
| `backend/requirements.txt` | Python dependencies |
| `backend/.python-version` | Pins Python 3.11 for Render compatibility |
| `backend/build.sh` | Render build script |
| `.gitignore` | Excludes `venv/`, `.env`, `__pycache__/` |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq API (`llama-3.1-8b-instant`) |
| **Embeddings** | FastEmbed (ONNX) — `all-MiniLM-L6-v2` |
| **Vector DB** | Qdrant (local SQLite mode) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Hosting** | Render (backend) + Vercel (frontend) |

## How to Run Locally

```bash
# 1. Clone and setup
git clone https://github.com/yameen185/bis-chatbot.git
cd bis-chatbot/backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Set API key
echo GROQ_API_KEY=your_key_here > .env

# 3. Run data pipeline (only needed once)
python crawler.py            # Scrapes BIS website → scraped_data.json
python ingest.py             # Embeds + stores → qdrant_db/

# 4. Start backend
python main.py               # Runs on http://localhost:8000

# 5. Start frontend (new terminal)
cd ../frontend
python -m http.server 3000   # Opens on http://localhost:3000
```

## Live URLs

| Component | URL |
|-----------|-----|
| Frontend | https://bis-chatbot-e93sjugoq-yameen185s-projects.vercel.app |
| Backend API | https://bis-chatbot-backend.onrender.com |
| API Docs | https://bis-chatbot-backend.onrender.com/docs |
| GitHub | https://github.com/yameen185/bis-chatbot |
