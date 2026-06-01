import os
import json
import time
from typing import List, Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
import chromadb
import groq
from openai import OpenAI
from dotenv import load_dotenv

# Setup configurations
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

groq_client = groq.Groq(api_key=GROQ_API_KEY)
nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ==========================================
# Rate Limit Retry Helper
# ==========================================
GROQ_CALL_DELAY = 2.5  # seconds between calls to stay under 30 RPM

def retry_api_call(api_func, max_retries=5, initial_wait=10):
    """Wraps an API call with exponential backoff for 429 rate limits."""
    # Pre-call delay to respect RPM limits
    time.sleep(GROQ_CALL_DELAY)
    
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return api_func()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  [RATE LIMITED] Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                raise  # Re-raise non-rate-limit errors
    # Final attempt without catching
    return api_func()

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
embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu') # Keep on CPU for inference stability, or 'cuda' if preferred

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
        
        response = retry_api_call(lambda: ollama_client.chat.completions.create(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        ))
        response_text = response.choices[0].message.content.strip()
        
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
    
    SYSTEM_PROMPT = "You are an evaluator. Determine if the provided evidence contains enough information to fully and accurately answer the user's question."
    USER_PROMPT_TEMPLATE = """
    User Question: {question}
    
    Evidence:
    {evidence}
    
    Does the evidence provide sufficient information to answer the question?
    Respond with exactly ONE word: "YES" or "NO".
    """

    @classmethod
    def is_sufficient(cls, question: str, chunks: List[Dict[str, Any]]) -> bool:
        evidence_text = "\n\n".join([f"Chunk: {c['content']}" for c in chunks])
        prompt = cls.USER_PROMPT_TEMPLATE.format(question=question, evidence=evidence_text)
        
        response = retry_api_call(lambda: ollama_client.chat.completions.create(
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="qwen2.5-coder:7b",
            temperature=0.0
        ))
        response_text = response.choices[0].message.content.strip().upper()
        
        return "YES" in response_text

class Synthesizer:
    """Drafts final answer using retrieved evidence and inline citations."""
    
    SYSTEM_PROMPT = """You are an expert AI research assistant. You MUST answer the user's question using ONLY the provided evidence. 
    You MUST cite your sources inline using brackets containing the exact arxiv_id metadata. 
    Example: "Agentic models utilize recursive memory structures [2405.00010]."
    If the evidence does not contain the answer, state that you do not have enough information."""
    
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
        
        response = retry_api_call(lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        ))
        return response.choices[0].message.content

class Verifier:
    """Audits the draft to ensure all claims are supported by their citations."""
    
    SYSTEM_PROMPT = "You are a rigid academic auditor. Your job is to verify that a drafted answer is fully supported by the referenced source evidence and does not hallucinate facts."
    
    USER_PROMPT_TEMPLATE = """
    Drafted Answer:
    {draft}
    
    Original Evidence Provided:
    {evidence}
    
    Does the drafted answer contain any claims that are NOT supported by the Original Evidence, or utilize incorrect inline citations?
    If the draft is fully supported and correct, output "PASS".
    If there are unsupported claims or hallucinations, output "FAIL".
    Do not output any other text than "PASS" or "FAIL".
    """
    
    @classmethod
    def audit_draft(cls, draft: str, chunks: List[Dict[str, Any]]) -> bool:
        evidence_str = ""
        for chunk in chunks:
            evidence_str += f"Source Arxiv ID: {chunk['arxiv_id']}\nText: {chunk['content']}\n\n"
            
        prompt = cls.USER_PROMPT_TEMPLATE.format(draft=draft, evidence=evidence_str)
        
        response = retry_api_call(lambda: ollama_client.chat.completions.create(
            model="qwen2.5-coder:7b",
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        ))
        result = response.choices[0].message.content.strip().upper()
        
        return "PASS" in result
