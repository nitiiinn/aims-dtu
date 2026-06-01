import os
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END

# Import the centralized modules
from src.modules import Planner, Retriever, Reflector, Synthesizer, Verifier

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    question: str
    question_type: str  # "factoid", "comparative", or "survey"
    sub_queries: List[str]
    # operator.add ensures that each loop appends new chunks rather than overwriting
    retrieved_chunks: Annotated[List[Dict[str, Any]], operator.add] 
    loop_count: int
    is_sufficient: bool
    draft_answer: str
    final_answer: str
    cited_papers: List[str]
    verification_passed: bool
    ablation_config: Dict[str, bool] # E.g., {"use_planner": True, "use_reflector": True}

# ==========================================
# 2. Graph Nodes
# ==========================================
def planner(state: AgentState):
    """Decomposes the main question into sub-queries."""
    if not state.get("ablation_config", {}).get("use_planner", True):
        return {"sub_queries": [state["question"]]}
    
    print("--- PLANNER ---")
    question = state["question"]
    sub_queries = Planner.decompose(question)
    
    return {"sub_queries": sub_queries}

def retriever(state: AgentState):
    """Queries ChromaDB using the sub-queries."""
    print("--- RETRIEVER ---")
    queries = state["sub_queries"]
    
    new_chunks = Retriever.fetch_evidence(queries, top_k=3)
    
    return {
        "retrieved_chunks": new_chunks, 
        "loop_count": state.get("loop_count", 0) + 1
    }

def reflector(state: AgentState):
    """Decides if the evidence is sufficient, or if we need to search again."""
    if not state.get("ablation_config", {}).get("use_reflector", True):
        return {"is_sufficient": True}
        
    print("--- REFLECTOR ---")
    is_sufficient = Reflector.is_sufficient(state["question"], state.get("retrieved_chunks", []))
    
    return {"is_sufficient": is_sufficient}

def synthesizer(state: AgentState):
    """Drafts final answer using only retrieved evidence and inline citations."""
    print("--- SYNTHESIZER ---")
    draft = Synthesizer.generate_draft(state["question"], state.get("retrieved_chunks", []), question_type=state.get("question_type", "factoid"))
    
    # We can extract cited papers from the draft string, similar to main.py
    import re
    matches = re.findall(r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]', draft)
    cited_papers = list(set(matches))
    
    return {"draft_answer": draft, "cited_papers": cited_papers, "final_answer": draft}

def verifier(state: AgentState):
    """Audits the synthesis against the original chunks."""
    if not state.get("ablation_config", {}).get("use_verifier", True):
        return {"final_answer": state["draft_answer"], "verification_passed": True}
        
    print("--- VERIFIER ---")
    passed = Verifier.audit_draft(state["draft_answer"], state.get("retrieved_chunks", []))
    
    return {"final_answer": state["draft_answer"], "verification_passed": passed}

# ==========================================
# 3. Routing Logic
# ==========================================
def should_continue(state: AgentState):
    """Routing function from Reflector."""
    if state["is_sufficient"]:
        return "synthesizer"
    
    if state["loop_count"] >= 3: # Hard cap on max loops
        print("--- MAX LOOPS REACHED, FORCING SYNTHESIS ---")
        return "synthesizer"
        
    return "planner" # Look for new sub-queries

# ==========================================
# 4. Build Graph
# ==========================================
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("retriever", retriever)
    workflow.add_node("reflector", reflector)
    workflow.add_node("synthesizer", synthesizer)
    workflow.add_node("verifier", verifier)
    
    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "reflector")
    
    # Conditional edge out of reflector
    workflow.add_conditional_edges(
        "reflector", 
        should_continue, 
        {"synthesizer": "synthesizer", "planner": "planner"}
    )
    
    workflow.add_edge("synthesizer", "verifier")
    workflow.add_edge("verifier", END)
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    graph = build_agent_graph()
    
    initial_state = {
        "question": "What kind of memory architectures do recent LLM agents use?",
        "ablation_config": {
            "use_planner": True,
            "use_reflector": True,
            "use_verifier": True
        }
    }
    
    # Run the graph
    print("Starting graph...")
    final_state = graph.invoke(initial_state)
    print("\n--- FINAL OUTPUT ---")
    print(final_state.get("final_answer", ""))

