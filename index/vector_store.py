import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import chromadb.errors

INPUT_FILE = "eval/corpus_chunks.jsonl"
DB_DIR = "index/chroma_db"
COLLECTION_NAME = "agent_papers"

# Lowered from 250 to 64 to prevent CUDA Out-of-Memory (OOM) errors 
# on a 6GB RTX 4050 when using the 'large' model.
BATCH_SIZE = 64  

def build_index():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the chunker first.")
        return

    # 1. Initialize the BGE-Base embedding model on the GPU
    print("Loading BGE-Base Embedding Model into VRAM...")
    try:
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
        print("Successfully loaded onto GPU.")
    except Exception as e:
        print(f"Failed to load on CUDA: {e}")
        print("Falling back to CPU. (Ensure you installed the PyTorch CUDA version!)")
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')

    # 2. Initialize ChromaDB persistent storage
    print(f"Initializing ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Create or get the collection (clears old data if you re-run)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    # FIX applied here: Catching the specific NotFoundError in newer ChromaDB versions
    except (ValueError, chromadb.errors.NotFoundError):
        pass # Collection didn't exist yet, which is completely fine!
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Cosine similarity is best for BGE
    )

    # 3. Load chunks
    chunks = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"Loaded {len(chunks)} chunks. Starting embedding process...")

    # 4. Process and insert in batches
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding & Indexing"):
        batch = chunks[i:i + BATCH_SIZE]
        
        documents = [item["content"] for item in batch]
        metadatas = [{"arxiv_id": item["arxiv_id"]} for item in batch]
        ids = [item["chunk_id"] for item in batch]
        
        # Generate embeddings (GPU accelerated)
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        
        # Insert into ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    print("\nSuccess! Vector index built and saved successfully.")
    print(f"Total vectors in database: {collection.count()}")

if __name__ == "__main__":
    build_index()