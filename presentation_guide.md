# BIS Chatbot — Presentation Guide

This guide breaks down everything you need to know for your presentation, structured in a way that's easy to explain to an audience.

---

## 1. Introduction (The "Why")

**What is it?**
An intelligent AI assistant designed specifically for the **Bureau of Indian Standards (BIS)** website. 

**The Problem:**
Government websites like BIS have thousands of pages of complex information (certifications, hallmarking, ISI marks). Finding specific answers is difficult and time-consuming for regular users.

**The Solution:**
A **Retrieval-Augmented Generation (RAG)** chatbot. Instead of users searching through pages, they just ask a question in plain English. The AI finds the exact official documents and summarizes the answer instantly.

---

## 2. The Tech Stack (What we used)

We built this using modern, lightweight, and incredibly fast technologies:

### 🧠 AI & Logic Layer
*   **Groq API (Llama 3.1 8B):** The brain of the chatbot. We use Groq because it uses specialized hardware (LPUs) to generate text incredibly fast. It's much faster than standard OpenAI or Google APIs.
*   **FastEmbed (ONNX):** Converts text into numbers (vector embeddings). We chose FastEmbed because it doesn't require heavy libraries like PyTorch. It runs in just ~100MB of RAM, making it perfect for free cloud hosting.

### 🗄️ Database & Storage
*   **Qdrant (Vector Database):** We use Qdrant in local SQLite mode. Instead of storing data in rows and columns, it stores concepts as vectors. This allows the AI to find information based on *meaning* rather than just exact keyword matches.

### ⚙️ Backend (The Engine)
*   **Python & FastAPI:** The backend server. FastAPI is lightning fast and handles all the communication between the database, the AI, and the user interface.
*   **Hosting:** Deployed on **Render** (Cloud Application Hosting).

### 🎨 Frontend (The User Interface)
*   **HTML, CSS, Vanilla JS:** A highly optimized, custom-built, responsive user interface. No bulky frameworks like React were used, ensuring it loads instantly. Features include Dark Mode, Markdown formatting, and mobile responsiveness.
*   **Hosting:** Deployed on **Vercel** (Edge Network Hosting).

---

## 3. How It Works (The Workflow)

Explain this in two phases: **Phase 1 (Data Prep)** and **Phase 2 (Answering a Question)**.

### Phase 1: Ingestion (How the AI learns)
*This happens once, behind the scenes.*
1.  **Crawling (`crawler.py`):** We scraped the official BIS website to collect all the text, titles, and URLs.
2.  **Chunking (`ingest.py`):** The text is too big to feed into the AI all at once. We break it down into smaller "chunks" of about 500 words each.
3.  **Embedding:** *FastEmbed* takes each chunk and converts it into a mathematical vector (a list of 384 numbers) that represents the *meaning* of that text.
4.  **Database:** These vectors are saved into the **Qdrant Vector Database**.

### Phase 2: Generation (How it answers a user)
*This happens in real-time when a user types a question.*
1.  **The User Asks:** A user asks, *"What is the ISI mark?"* on the Vercel frontend.
2.  **Embedding the Question:** The backend takes this exact question and converts it into a vector using the same FastEmbed model.
3.  **Similarity Search (Retrieval):** The Qdrant database compares the *question vector* against all the *document vectors* to find the top 5 closest matches. (It finds the official BIS paragraphs about the ISI mark).
4.  **Prompt Construction:** The backend takes the user's question, the 5 retrieved paragraphs, and strict rules (e.g., "Don't hallucinate", "Write in bullet points") and sends a massive prompt to the **Groq AI**.
5.  **The Answer (Generation):** Groq reads the official context and instantly generates a human-like, beautifully formatted answer, which is sent back to the user's screen along with the original source URLs.

---

## 4. Key Features to Highlight

If the judges/audience ask what makes this special, point these out:

*   **Zero Hallucination Guarantee:** Because it uses the "RAG" architecture, the AI is physically restricted to *only* answering based on the retrieved BIS documents. If it doesn't know, it refuses to guess.
*   **Memory Optimization:** We bypassed Render's strict 512MB RAM limit by using ONNX/FastEmbed instead of PyTorch, and using "Lazy Loading" (only booting up heavy models when the first message is sent).
*   **Conversational Memory:** The bot remembers the last 10 messages. If you ask "What is hallmarking?", and then follow up with "How do I apply for *it*?", the AI knows "it" means hallmarking.
*   **Out-of-Scope Filtering:** If you ask it about sports, recipes, or weather, an internal filter immediately kicks in and reminds the user that it only answers BIS-related questions.

---

## 5. Why Did We Switch from Gemini to Groq?

*You can mention this if they ask about challenges faced.*
*   **Challenge:** We initially tried Google's Gemini, but we ran into persistent API authentication errors and environment issues that blocked the pipeline.
*   **Solution:** We pivoted to **Groq** using the Llama 3.1 model. Moving to Groq turned out to be a massive upgrade because their LPUs (Language Processing Units) generate text almost instantaneously, making the chatbot feel much snappier for the end user.
