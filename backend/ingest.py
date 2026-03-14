import json
import logging
import uuid
from typing import List, Dict, Any
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "scraped_data.json"
COLLECTION_NAME = "bis_knowledge"
CHUNK_SIZE = 500  # approximate tokens/words per chunk
CHUNK_OVERLAP = 50

# Initialize embedding model (lightweight ONNX-based)
embedder = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
VECTOR_SIZE = 384 # Size of embeddings for all-MiniLM-L6-v2

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    i = 0
    while i < len(words):
        # Create chunk of specified size
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Move forward by chunk_size - overlap
        i += chunk_size - overlap
        
    return chunks

def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and collection."""
    # Using local persistent storage for simplicity
    # For production, you could connect to Qdrant Cloud: qdrant_client.QdrantClient(url="...", api_key="...")
    qdrant = QdrantClient(path="./qdrant_db")
    
    # Create collection if it doesn't exist
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        logger.info(f"Creating collection {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return qdrant

def main():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file {INPUT_FILE} not found. Run crawler.py first.")
        return

    qdrant = init_qdrant()
    
    points = []
    logger.info("Processing articles and generating embeddings...")
    
    for item in data:
        url = item['url']
        title = item['title']
        content = item['content']
        
        # Chunk text
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20: # Skip very small chunks
                continue
                
            # Generate embedding
            vector = list(embedder.embed([chunk]))[0].tolist()
            
            # Create a unique ID for the chunk
            point_id = str(uuid.uuid4())
            
            # Create payload
            payload = {
                "url": url,
                "title": title,
                "text": chunk,
                "chunk_index": i
            }
            
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            
    # Upload to Qdrant in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        logger.info(f"Uploaded batch of {len(batch)} chunks (Processed {min(i + batch_size, len(points))}/{len(points)})")

    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()
