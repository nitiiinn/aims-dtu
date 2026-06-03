import json
from src.modules import Planner, Retriever, Reflector, Synthesizer, Verifier

def run_agent(question: str, config: dict, question_type: str = "factoid") -> dict:
    print(f"\n{'='*50}")
    print(f"PROCESSING QUESTION: '{question}'")
    print(f"{'='*50}")
    
    # 1. Plan the queries
    if config.get("use_planner", True):
        print("\n[PLANNER] Decomposing query...")
        queries = Planner.decompose(question)
    else:
        print("\n[PLANNER] Skipped (Ablation). Using raw question.")
        queries = [question]
        
    print(f"-> Queries: {queries}")
    
    # Setup loop variables
    max_loops = 3
    loop_count = 0
    accumulated_chunks = []
    is_sufficient = False
    
    # 2. Enter a while loop (max 3 iterations)
    while loop_count < max_loops:
        loop_count += 1
        print(f"\n--- LOOP ITERATION {loop_count} ---")
        
        # 3. Retrieve chunks based on queries
        print(f"[RETRIEVER] Fetching chunks from ChromaDB...")
        new_chunks = Retriever.fetch_evidence(queries, top_k=3)
        
        # Deduplicate and accumulate chunks to retain context across loops
        existing_ids = {c["chunk_id"] for c in accumulated_chunks}
        for chunk in new_chunks:
            if chunk["chunk_id"] not in existing_ids:
                accumulated_chunks.append(chunk)
                
        print(f"-> Total unique chunks accumulated: {len(accumulated_chunks)}")
        
        # 4. Reflect on chunks
        if config.get("use_reflector", True):
            print("[REFLECTOR] Evaluating evidence sufficiency...")
            is_sufficient = Reflector.is_sufficient(question, accumulated_chunks)
            print(f"-> Sufficient? {is_sufficient}")
        else:
            print("[REFLECTOR] Skipped (Ablation). Assuming evidence is sufficient.")
            is_sufficient = True
            
        if is_sufficient:
            break
        else:
            if loop_count < max_loops:
                print("-> Generating new fallback queries to dig deeper...")
                if config.get("use_planner", True):
                    queries = Planner.decompose(f"Find more specific technical details to answer: {question}")
                else:
                    print("-> Planner disabled. Cannot generate new queries. Breaking loop.")
                    break
            else:
                print("-> Max loops reached. Forcing synthesis with current evidence.")

    # 5. Synthesize the final answer
    print("\n[SYNTHESIZER] Drafting final answer...")
    draft_answer = Synthesizer.generate_draft(question, accumulated_chunks, question_type=question_type)
    
    # 6. Verify the citations
    verification_passed = True
    if config.get("use_verifier", True):
        print("[VERIFIER] Auditing draft against source chunks...")
        verification_passed = Verifier.audit_draft(draft_answer, accumulated_chunks)
        print(f"-> Verification Passed? {verification_passed}")
    else:
        print("[VERIFIER] Skipped (Ablation).")
        
    return {
        "question": question,
        "final_answer": draft_answer,
        "verification_passed": verification_passed,
        "loops_taken": loop_count,
        "chunks_used": len(accumulated_chunks),
        "retrieved_chunks": accumulated_chunks
    }

if __name__ == "__main__":
    # Test execution with Ablation Configs
    test_question = "What exactly constitutes an 'agentic' workflow, according to Andrew Ng's recent publications?"
    
    test_config = {
        "use_planner": True,
        "use_reflector": True,
        "use_verifier": True
    }
    
    result = run_agent(test_question, test_config)
    print("\n=== FINAL RESULT ===")
    print(result["final_answer"])
    print(f"\nVerified: {result['verification_passed']} | Loops: {result['loops_taken']}")
