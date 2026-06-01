import json
import os
from tqdm import tqdm

INPUT_FILE = "eval/parsed_texts.jsonl"
OUTPUT_FILE = "eval/corpus_chunks.jsonl"

# Tuning parameters for your retrieval stack
CHUNK_SIZE = 1000  # Number of characters per chunk
OVERLAP = 200      # Number of characters to overlap to preserve context

def chunk_text(text, chunk_size, overlap):
    """Slices text into overlapping windows."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the parser first.")
        return

    all_chunks = []
    
    # Read the JSONL file line by line
    with open(INPUT_FILE, 'r', encoding='utf-8') as in_f:
        lines = in_f.readlines()
        
    for line in tqdm(lines, desc="Chunking Text"):
        record = json.loads(line)
        arxiv_id = record['arxiv_id']
        text = record['text']
        
        text_chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        
        # Bind the metadata to each chunk
        for i, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{arxiv_id}_{i}",
                "arxiv_id": arxiv_id,
                "content": chunk_content
            })

    # Save the final chunks
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for chunk in all_chunks:
            out_f.write(json.dumps(chunk) + "\n")
            
    print(f"\nGenerated {len(all_chunks)} total chunks from {len(lines)} papers.")

if __name__ == "__main__":
    main()