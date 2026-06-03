import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI


# Setup configurations
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")



# ==========================================
# Caching System
# ==========================================
CACHE_FILE = "llm_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def cached_llm_call(messages, model, temperature=0.0):
    """Wrapper that hashes the LLM request and checks a local JSON cache first."""
    cache = load_cache()
    
    # Create a unique fingerprint for this exact request
    key_string = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    if key_hash in cache:
        print("  [CACHE HIT] Using cached response from llm_cache.json")
        return cache[key_hash]
        
    print("  [CACHE MISS] Calling local Qwen model...")
    # Execute actual API call if not in cache
    response = ollama_client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature
    )
    
    response_text = response.choices[0].message.content
    cache[key_hash] = response_text
    save_cache(cache)
    
    return response_text

# ==========================================
# Database Setup
# ==========================================
DB_DIR = "index/chroma_db"
COLLECTION_NAME = "agent_papers"
# Initialize DB connection and Embedding model once globally
db_client = chromadb.PersistentClient(path=DB_DIR)
# We use get_collection, assuming it was created by vector_store.py
try:
    collection = db_client.get_collection(name=COLLECTION_NAME)
except ValueError:
     # Fallback for testing if run before indexing
    collection = db_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# Initialize SentenceTransformer
print("Loading BGE-Base model for Retrieval...")
import torch
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device_type)

# ==========================================
# Module Classes
# ==========================================

class Planner:
    """Decomposes a complex query into specific search queries."""
    
    SYSTEM_PROMPT = "You are a research planner. Break the user's question into 1 to 3 distinct search queries to query a vector database."
    USER_PROMPT_TEMPLATE = """
    User Question: {question}

    Output ONLY a valid JSON list of strings, representing the search queries. Example: ["query 1", "query 2"]
    Do not output any markdown formatting like ```json. Just the raw array.
    """

    @classmethod
    def decompose(cls, question: str) -> List[str]:
        prompt = cls.USER_PROMPT_TEMPLATE.format(question=question)
        
        response_text = cached_llm_call(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        ).strip()
        
        try:
            # Parse the JSON array string
            queries = json.loads(response_text)
            if not isinstance(queries, list):
                queries = [question] # Fallback
        except json.JSONDecodeError:
            queries = [question] # Fallback
            
        return queries

class Retriever:
    """Embeds queries and retrieves chunks from ChromaDB."""
    
    @classmethod
    def fetch_evidence(cls, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns a list of deduplicated chunks with metadata."""
        all_results = []
        seen_ids = set()
        
        for q in queries:
            query_embedding = embedder.encode([q], normalize_embeddings=True).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # Format results into a cleaner list of dictionaries
            # Chroma returns lists inside lists. We assume 1 query per call.
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            chunk_ids = results['ids'][0]
            
            for doc, meta, cid in zip(docs, metadatas, chunk_ids):
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_results.append({
                        "chunk_id": cid,
                        "content": doc,
                        "arxiv_id": meta.get("arxiv_id", "UNKNOWN")
                    })
                    
        return all_results

class Reflector:
    """Evaluates if current evidence answers the query."""
    
    SYSTEM_PROMPT = "You are an evaluator. Determine if the provided evidence contains relevant information to form a reasonable answer to the user's question. It does not need to be perfect, just sufficient to address the core inquiry."
    USER_PROMPT_TEMPLATE = """
    User Question: {question}
    
    Evidence:
    {evidence}
    
    Does the evidence provide sufficient information to construct a reasonable answer to the question?
    Respond with exactly ONE word: "YES" or "NO".
    """

    @classmethod
    def is_sufficient(cls, question: str, chunks: List[Dict[str, Any]]) -> bool:
        evidence_text = "\n\n".join([f"Chunk: {c['content']}" for c in chunks])
        prompt = cls.USER_PROMPT_TEMPLATE.format(question=question, evidence=evidence_text)
        
        response_text = cached_llm_call(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        ).strip().upper()
        
        return "YES" in response_text

class Synthesizer:
    """Drafts final answer using retrieved evidence and inline citations."""
    
    SYSTEM_PROMPT = """You are an expert AI research assistant. You MUST answer the user's question using ONLY the provided evidence. 
    CRITICAL: You MUST cite your sources inline using brackets containing the EXACT arxiv_id provided in the 'Source Arxiv ID' field of the evidence. 
    DO NOT use numbers like [1] or [54]. ONLY use the exact string like [2405.00010].
    Example: "Agentic models utilize recursive memory structures [2405.00010]."
    If the evidence does not contain the answer, state that you do not have enough information and do not include any citations."""
    
    # Length guidance per question type, matching SUBMISSION_FORMAT.md
    LENGTH_GUIDANCE = {
        "factoid": "Your answer MUST be concise: exactly 1 to 3 sentences. Do not exceed 3 sentences.",
        "comparative": "Your answer MUST be 100 to 300 words. Aim for a focused, structured comparison within this range.",
        "survey": "Your answer MUST be 250 to 600 words. Provide a thorough, well-structured survey within this range.",
    }
    
    USER_PROMPT_TEMPLATE = """
    User Question: {question}
    
    Evidence:
    {evidence}
    
    {length_instruction}
    
    Write the final drafted answer. Cite your sources inline using [arxiv_id].
    """
    
    @classmethod
    def generate_draft(cls, question: str, chunks: List[Dict[str, Any]], question_type: str = "factoid", use_gemini=False) -> str:
        # Format the evidence clearly mapping the ID to the text
        evidence_str = ""
        for chunk in chunks:
            evidence_str += f"Source Arxiv ID: {chunk['arxiv_id']}\nText: {chunk['content']}\n\n"
        
        length_instruction = cls.LENGTH_GUIDANCE.get(question_type, cls.LENGTH_GUIDANCE["factoid"])
            
        prompt = cls.USER_PROMPT_TEMPLATE.format(question=question, evidence=evidence_str, length_instruction=length_instruction)
        
        response_text = cached_llm_call(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        )
        return response_text

class Verifier:
    """Audits the draft to ensure all claims are supported by their citations."""
    
    SYSTEM_PROMPT = "You are an auditor. Verify that the core claims in the drafted answer are supported by the provided evidence and that citations are used appropriately."
    
    USER_PROMPT_TEMPLATE = """
    Drafted Answer:
    {draft}
    
    Original Evidence Provided:
    {evidence}
    
    Are the main claims in the drafted answer supported by the Original Evidence?
    If the draft is generally supported and does not contain major hallucinations, output "PASS".
    If the draft contains severe hallucinations or completely unsupported major claims, output "FAIL".
    Do not output any other text than "PASS" or "FAIL".
    """
    
    @classmethod
    def audit_draft(cls, draft: str, chunks: List[Dict[str, Any]]) -> bool:
        evidence_str = ""
        for chunk in chunks:
            evidence_str += f"Source Arxiv ID: {chunk['arxiv_id']}\nText: {chunk['content']}\n\n"
            
        prompt = cls.USER_PROMPT_TEMPLATE.format(draft=draft, evidence=evidence_str)
        
        response_text = cached_llm_call(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        )
        result = response_text.strip().upper()
        
        return "PASS" in result
