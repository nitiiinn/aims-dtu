import json
import os
import glob
import requests

from src.modules import Retriever

# ==========================================
# Ollama LLM-as-Judge Configuration
# ==========================================
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "qwen2.5-coder:7b"

GROUND_TRUTH_FILE = "eval/local_ground_truth.jsonl"
PREDICTIONS_DIR = "predictions"


# ==========================================
# 1. Load Ground Truth
# ==========================================
def load_ground_truth(filepath):
    """Load ground truth into a dict keyed by question_id for O(1) lookups."""
    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                qid = record.get("question_id") or record.get("id")
                gt[qid] = record
    return gt


# ==========================================
# 2. Citation Metrics (Pure Python)
# ==========================================
def citation_precision_recall_f1(predicted_papers, required_papers):
    """Calculate Precision, Recall, and F1 for citation lists.
    Handles division-by-zero safely.
    """
    pred_set = set(predicted_papers)
    req_set = set(required_papers)

    if len(pred_set) == 0 and len(req_set) == 0:
        return 1.0, 1.0, 1.0  # Both empty = perfect match

    true_positives = len(pred_set & req_set)

    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(req_set) if len(req_set) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# ==========================================
# 3. Ollama Helper
# ==========================================
def call_ollama(system_prompt, user_prompt):
    """Send a chat completion request to the local Ollama server."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    [OLLAMA ERROR] {e}")
        return None


# ==========================================
# 4. Answer Accuracy (LLM Judge, 1-5)
# ==========================================
ACCURACY_SYSTEM = (
    "You are a strict academic grader. You will be given a predicted answer and a "
    "reference answer. Grade how semantically aligned the prediction is with the "
    "reference on a scale of 1 to 5:\n"
    "  1 = Completely wrong or irrelevant\n"
    "  2 = Mentions the topic but mostly incorrect\n"
    "  3 = Partially correct, missing key details\n"
    "  4 = Mostly correct with minor omissions\n"
    "  5 = Fully correct and well-aligned\n\n"
    "You MUST respond with ONLY a single integer digit (1, 2, 3, 4, or 5). "
    "Do NOT output any other text, explanation, or punctuation."
)

ACCURACY_USER_TEMPLATE = (
    "Reference Answer:\n{reference}\n\n"
    "Predicted Answer:\n{prediction}\n\n"
    "Grade (1-5):"
)


def judge_accuracy(prediction_answer, reference_answer):
    """Ask Ollama to score semantic alignment on a 1-5 scale."""
    user_prompt = ACCURACY_USER_TEMPLATE.format(
        reference=reference_answer,
        prediction=prediction_answer,
    )
    raw = call_ollama(ACCURACY_SYSTEM, user_prompt)
    if raw is None:
        return 0

    # Parse the first digit found in the response
    for ch in raw:
        if ch.isdigit() and ch in "12345":
            return int(ch)
    return 0  # Fallback if parsing fails


# ==========================================
# 5. Faithfulness (LLM Judge, 0 or 1)
# ==========================================
FAITHFULNESS_SYSTEM = (
    "You are a hallucination detector. You will be given a generated answer and "
    "the source context it was supposed to be based on. Determine whether the "
    "answer is fully faithful to the provided context.\n\n"
    "Rules:\n"
    "- If EVERY claim in the answer is supported by the context, output 1.\n"
    "- If the answer contains ANY unsupported or fabricated claims, output 0.\n\n"
    "You MUST respond with ONLY a single digit: 1 or 0. "
    "Do NOT output any other text, explanation, or punctuation."
)

FAITHFULNESS_USER_TEMPLATE = (
    "Source Context:\n{context}\n\n"
    "Generated Answer:\n{prediction}\n\n"
    "Faithful (1 or 0):"
)


def judge_faithfulness(prediction_answer, context):
    """Ask Ollama whether the answer is faithful to the context.
    Uses dynamically retrieved chunks from ChromaDB as the source context.
    """
    user_prompt = FAITHFULNESS_USER_TEMPLATE.format(
        context=context,
        prediction=prediction_answer,
    )
    raw = call_ollama(FAITHFULNESS_SYSTEM, user_prompt)
    if raw is None:
        return 0

    for ch in raw:
        if ch in "01":
            return int(ch)
    return 0  # Fallback


# ==========================================
# 6. Process a Single Prediction File
# ==========================================
def evaluate_prediction_file(filepath, ground_truth):
    """Score every row in a prediction JSONL file and return macro-averages."""
    predictions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    precisions, recalls, f1s = [], [], []
    accuracies = []
    faithfulness_scores = []

    for pred in predictions:
        qid = pred.get("id")
        gt = ground_truth.get(qid)

        if gt is None:
            print(f"    [WARN] No ground truth for question '{qid}', skipping.")
            continue

        pred_answer = pred.get("answer", "")
        pred_papers = pred.get("cited_papers", [])
        ref_answer = gt.get("reference_answer", "")
        req_papers = gt.get("required_arxiv_ids", [])

        # --- Citation Metrics ---
        p, r, f1 = citation_precision_recall_f1(pred_papers, req_papers)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        # --- Accuracy (LLM Judge) ---
        acc = judge_accuracy(pred_answer, ref_answer)
        accuracies.append(acc)

        # --- Faithfulness (LLM Judge) ---
        # Dynamically retrieve the actual context chunks from ChromaDB
        question_text = gt.get("question", "")
        retrieved_chunks = Retriever.fetch_evidence([question_text], top_k=3)
        context_text = "\n\n".join(
            f"[{c['arxiv_id']}]: {c['content']}" for c in retrieved_chunks
        )
        faith = judge_faithfulness(pred_answer, context_text)
        faithfulness_scores.append(faith)

    n = len(precisions)
    if n == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0, "faithfulness": 0, "count": 0}

    return {
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
        "accuracy": sum(accuracies) / n,
        "faithfulness": sum(faithfulness_scores) / n,
        "count": n,
    }


# ==========================================
# 7. Main Driver
# ==========================================
def main():
    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: {GROUND_TRUTH_FILE} not found. Run seed_ground_truth.py first.")
        return

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"Loaded {len(ground_truth)} ground-truth entries.\n")

    # Discover all prediction files
    pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.jsonl")))
    if not pred_files:
        print(f"Error: No .jsonl files found in {PREDICTIONS_DIR}/")
        return

    print(f"Found {len(pred_files)} prediction files to evaluate.\n")

    # Evaluate each configuration
    results = {}
    for filepath in pred_files:
        config_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"[EVALUATING] {config_name} ({filepath})")
        scores = evaluate_prediction_file(filepath, ground_truth)
        results[config_name] = scores
        print(f"  -> Evaluated {scores['count']} questions.\n")

    # ==========================================
    # 8. Print Global Telemetry Summary Table
    # ==========================================
    print("\n" + "=" * 85)
    print("  ABLATION STUDY — GLOBAL TELEMETRY SUMMARY")
    print("=" * 85 + "\n")

    # Markdown table header
    header = "| Configuration   | Precision | Recall | F1-Score | Accuracy (1-5) | Faithfulness |"
    separator = "| --------------- | --------- | ------ | -------- | -------------- | ------------ |"
    print(header)
    print(separator)

    for config_name, scores in results.items():
        row = (
            f"| {config_name:<15} "
            f"| {scores['precision']:.3f}     "
            f"| {scores['recall']:.3f}  "
            f"| {scores['f1']:.3f}    "
            f"| {scores['accuracy']:.2f}           "
            f"| {scores['faithfulness']:.2f}         |"
        )
        print(row)

    print("\n" + "=" * 85)
    print(f"  Evaluated {len(results)} configurations × {len(ground_truth)} questions each.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
