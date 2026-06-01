import os
import json
import re
import traceback
from tqdm import tqdm
from src.loop import run_agent

INPUT_FILE = "eval/questions.jsonl"
PREDS_DIR = "predictions"

def extract_citations(text: str) -> list:
    """Extracts arxiv IDs from inline citations like [2405.00010]."""
    # Regex for standard arXiv format (e.g., 2405.00010 or 2405.00010v1)
    matches = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', text)
    # Return unique citations
    return list(set(matches))

def main():
    if not os.path.exists(PREDS_DIR):
        os.makedirs(PREDS_DIR)

    # 1. Load questions
    questions = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loaded {len(questions)} questions.")

    # 2. Define the exact configurations for ablation studies
    # You mentioned 6 configs but listed 5, so I implemented the 5 specified ones.
    configs = {
        "baseline": {
            "use_planner": False,
            "use_reflector": False,
            "use_verifier": False
        },
        "full_agent": {
            "use_planner": True,
            "use_reflector": True,
            "use_verifier": True
        },
        "no_planner": {
            "use_planner": False,
            "use_reflector": True,
            "use_verifier": True
        },
        "no_reflector": {
            "use_planner": True,
            "use_reflector": False,
            "use_verifier": True
        },
        "no_verifier": {
            "use_planner": True,
            "use_reflector": True,
            "use_verifier": False
        }
    }

    # 3. Execute the Q&A loop for each configuration
    for config_name, config_flags in configs.items():
        output_file = os.path.join(PREDS_DIR, f"{config_name}.jsonl")
        print(f"\n==================================================")
        print(f"STARTING CONFIGURATION: {config_name.upper()}")
        print(f"Flags: {config_flags}")
        print(f"==================================================")
        
        # Open the file in write mode to start fresh for this config
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for q_obj in tqdm(questions, desc=f"Processing {config_name}"):
                q_id = q_obj["id"]
                q_text = q_obj["question"]
                q_type = q_obj.get("type", "factoid")
                
                # Execute the agent block
                try:
                    result = run_agent(q_text, config_flags, question_type=q_type)
                    
                    # Extract the cited papers natively from the LLM's generated brackets
                    cited_papers = extract_citations(result["final_answer"])
                    
                    # Aligning exactly with eval/SUBMISSION_FORMAT.md requirements
                    prediction_obj = {
                        "id": q_id,
                        "answer": result["final_answer"],
                        "cited_papers": cited_papers
                    }
                    
                    # You can add custom tracked metrics optionally if needed outside submission formatting
                    # But the base JSON must exactly match the expected schema
                except Exception as e:
                    print(f"\n[ERROR] Error processing question {q_id}: {e}")
                    traceback.print_exc()
                    prediction_obj = {
                        "id": q_id,
                        "answer": f"ERROR: {str(e)}",
                        "cited_papers": []
                    }
                
                # Write immediately (newline-delimited JSON)
                out_f.write(json.dumps(prediction_obj) + '\n')
                out_f.flush() # Ensure it's saved in case of a crash midway
                
        print(f"Saved {config_name} predictions to {output_file}")

if __name__ == "__main__":
    main()
