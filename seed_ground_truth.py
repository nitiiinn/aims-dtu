import json
import os

QUESTIONS_FILE = "eval/questions.jsonl"
BASELINE_FILE = "predictions/baseline.jsonl"
OUTPUT_FILE = "eval/local_ground_truth.jsonl"

def main():
    # 1. Load questions
    questions = {}
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                questions[record["id"]] = record["question"]

    # 2. Load baseline predictions
    predictions = {}
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                predictions[record["id"]] = {
                    "answer": record["answer"],
                    "cited_papers": record.get("cited_papers", []),
                }

    # 3. Match and format
    matched = []
    for qid, question_text in questions.items():
        if qid in predictions:
            matched.append({
                "question_id": qid,
                "question": question_text,
                "reference_answer": predictions[qid]["answer"],
                "required_arxiv_ids": predictions[qid]["cited_papers"],
            })

    # 4. Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in matched:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully mapped {len(matched)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
