from typing import List, Dict, Any
from utils import extract_numerical_answer, calculate_uncertainty_proxy
from config import config

def build_prompt(question: str, history: List[Dict[str, Any]], step: int) -> str:
    if step == 0:
        return f"Solve the following math problem step-by-step. End your solution with '#### [final answer]'.\n\nQuestion: {question}\n\nSolution:"
    
    prev_turn = history[-1]
    return (
        f"You are an expert auditor. Review the previous answer to this math problem.\n"
        f"Question: {question}\n\n"
        f"Previous Solution:\n{prev_turn['response']}\n\n"
        f"Audit and revise the solution if there are errors. Provide your complete revised step-by-step reasoning "
        f"and end with '#### [final answer]'."
    )

def execute_batch_audit_loop(dataset: List[Dict[str, Any]], engine_a, engine_b) -> List[Dict[str, Any]]:
    """Executes the iterative loop using massive batching."""
    
    active_items = [{
        "id": item["id"],
        "question": item["question"],
        "ground_truth_num": item["ground_truth_num"],
        "history": [],
        "completed": False,
        "total_calls": 0
    } for item in dataset]

    engines = [engine_a, engine_b, engine_a, engine_b]

    for step in range(config.MAX_AUDIT_STEPS):
        print(f"\n--- Starting Audit Step {step + 1} ---")
        
        pending_indices = [i for i, item in enumerate(active_items) if not item["completed"]]
        if not pending_indices:
            print("All questions converged early.")
            break
            
        current_engine = engines[step]
        auditor_label = "Model_A" if step % 2 == 0 else "Model_B"
        
        prompts = [build_prompt(active_items[i]["question"], active_items[i]["history"], step) for i in pending_indices]
        print(f"Generating {config.SAMPLES_PER_STEP} samples for {len(prompts)} questions...")
        
        batch_samples = current_engine.generate_batch(prompts)
        
        for idx, item_idx in enumerate(pending_indices):
            item = active_items[item_idx]
            samples = batch_samples[idx]
            
            extracted_nums = [extract_numerical_answer(s) for s in samples]
            maj_answer, agreement, uncertainty = calculate_uncertainty_proxy(extracted_nums)
            
            # Pick a representative response matching the majority
            selected_response = samples[0]
            for s, num in zip(samples, extracted_nums):
                if num == maj_answer:
                    selected_response = s
                    break
                    
            item["total_calls"] += config.SAMPLES_PER_STEP
            item["history"].append({
                "step": step + 1,
                "auditor": auditor_label,
                "majority_answer": maj_answer,
                "agreement_ratio": agreement,
                "uncertainty": uncertainty,
                "response": selected_response,
                "is_correct": (maj_answer == item["ground_truth_num"])
            })
            
            # Stop condition
            if agreement >= config.HIGH_AGREEMENT_THRESHOLD:
                item["completed"] = True

    return active_items