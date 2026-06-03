import json
import os
import sys
import re
import time
import requests
from dotenv import load_dotenv

# Ensure project root is on sys.path so src.modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

load_dotenv()

# We import the Retriever to dynamically fetch chunks from ChromaDB
from src.modules import Retriever

# ==========================================
# Configuration
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Strong model for reference answers
CALL_DELAY = 2.5  # 30 RPM limit → 60/30 = 2s minimum between calls

QUESTIONS_FILE = "eval/questions.jsonl"
OUTPUT_FILE = "eval/local_ground_truth.jsonl"

# ==========================================
# Length guidance per question type
# ==========================================
LENGTH_GUIDANCE = {
    "factoid": "Your answer MUST be concise: exactly 1 to 3 sentences.",
    "comparative": "Your answer MUST be 100 to 300 words. Provide a focused, structured comparison.",
    "survey": "Your answer MUST be 250 to 600 words. Provide a thorough, well-structured survey.",
}

# ==========================================
# System prompt for generating gold answers
# ==========================================
SYSTEM_PROMPT = """You are an expert AI research assistant producing gold-standard reference answers.
You MUST answer the user's question using ONLY the provided evidence chunks.
CRITICAL: You MUST cite your sources inline using brackets containing the EXACT arxiv_id provided. 
Example: "Agentic models utilize recursive memory structures [2405.00010]."
If the evidence does not contain the answer, state that clearly and do not fabricate information."""

USER_PROMPT_TEMPLATE = """Question: {question}

Evidence:
{evidence}

{length_instruction}

Write a high-quality reference answer. Cite sources inline using [arxiv_id].
"""


# ==========================================
# Groq API Call
# ==========================================
def call_groq(system_prompt, user_prompt):
    """Send a chat completion request to Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            print(f"    [RATE LIMITED] Waiting 30s before retry...")
            time.sleep(30)
            return call_groq(system_prompt, user_prompt)  # Retry once
        else:
            print(f"    [GROQ ERROR] {e}")
            return None
    except Exception as e:
        print(f"    [GROQ ERROR] {e}")
        return None


def extract_citations(text: str) -> list:
    """Extracts arxiv IDs from inline citations like [2405.00010]."""
    matches = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', text)
    return list(set(matches))


# ==========================================
# Main
# ==========================================
def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env")
        return

    # 1. Load questions
    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions.")
    print(f"Using Groq model: {GROQ_MODEL}")
    print(f"Delay between calls: {CALL_DELAY}s\n")

    # 2. Generate reference answers
    ground_truth = []

    for i, q_obj in enumerate(questions):
        qid = q_obj["id"]
        question = q_obj["question"]
        q_type = q_obj.get("type", "factoid")

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:80]}...")

        # Retrieve chunks from ChromaDB
        chunks = Retriever.fetch_evidence([question], top_k=5)
        evidence_str = ""
        for chunk in chunks:
            evidence_str += f"Source Arxiv ID: {chunk['arxiv_id']}\nText: {chunk['content']}\n\n"

        # Build prompt
        length_instruction = LENGTH_GUIDANCE.get(q_type, LENGTH_GUIDANCE["factoid"])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            evidence=evidence_str,
            length_instruction=length_instruction,
        )

        # Call Groq
        answer = call_groq(SYSTEM_PROMPT, user_prompt)

        if answer is None:
            print(f"    [FAILED] Skipping {qid}")
            continue

        # Extract citations from the generated answer
        cited_papers = extract_citations(answer)

        # Fallback: use chunk arxiv_ids if LLM didn't cite inline
        if not cited_papers:
            cited_papers = list(set(
                c["arxiv_id"] for c in chunks
                if c.get("arxiv_id") and c["arxiv_id"] != "UNKNOWN"
            ))

        ground_truth.append({
            "question_id": qid,
            "question": question,
            "reference_answer": answer,
            "required_arxiv_ids": cited_papers,
        })

        print(f"    [OK] {len(cited_papers)} citations extracted")

        # Rate limit delay (skip after last question)
        if i < len(questions) - 1:
            print(f"    Waiting {CALL_DELAY}s...")
            time.sleep(CALL_DELAY)

    # 3. Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in ground_truth:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSuccessfully generated {len(ground_truth)} reference answers to {OUTPUT_FILE}")
    print(f"Model used: {GROQ_MODEL} (via Groq)")


if __name__ == "__main__":
    main()
