from typing import List, Dict, Any
from utils import extract_numerical_answer, calculate_uncertainty_proxy
from config import config

def build_prompt(question: str, audit_history: List[Dict[str, Any]], step: int) -> str:
    """Constructs prompt for initial proposal or auditor revision."""
    if step == 0:
        return f"Solve the following math problem step-by-step. End your solution with '#### [final answer]'.\n\nQuestion: {question}\n\nSolution:"
    
    prev_turn = audit_history[-1]
    prompt = (
        f"You are an expert auditor. Review the previous answer to the math problem below.\n"
        f"Question: {question}\n\n"
        f"Previous Solution:\n{prev_turn['response']}\n\n"
        f"Audit and revise the solution if there are errors. Provide your complete revised step-by-step reasoning "
        f"and end with '#### [final answer]'."
    )
    return prompt

def execute_audit_loop(dataset: List[Dict[str, Any]], engine_a, engine_b) -> List[Dict[str, Any]]:
    """
    Executes the 4-step iterative audit loop across all questions.
    Roadmap:
      Step 1: Model A proposes initial answer.
      Step 2: Model B audits and revises.
      Step 3: Model A audits and revises.
      Step 4: Model B audits and revises.
      Early Stopping: High numerical answer agreement (e.g. >= 80%).
    """
    print(f"[AuditLoop] Running Iterative Audit Loop over {len(dataset)} items...")
    
    # Track active questions through iterative turns
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
        pending_indices = [i for i, item in enumerate(active_items) if not item["completed"]]
        if not pending_indices:
            print(f"[AuditLoop] All questions converged early at Step {step}.")
            break
            
        current_engine = engines[step]
        auditor_label = "Model_A" if step % 2 == 0 else "Model_B"
        
        # Prepare prompts
        prompts = [
            build_prompt(active_items[i]["question"], active_items[i]["history"], step)
            for i in pending_indices
        ]
        
        # Batch generation (5 samples per question for uncertainty proxy)
        batch_samples = current_engine.generate_samples(
            prompts, 
            num_samples=config.SAMPLES_PER_STEP, 
            temperature=config.TEMPERATURE
        )
        
        for idx, item_idx in enumerate(pending_indices):
            item = active_items[item_idx]
            samples = batch_samples[idx]
            
            # Extract numerical answers from 5 auditor samples
            extracted_nums = [extract_numerical_answer(s) for s in samples]
            maj_answer, agreement, uncertainty = calculate_uncertainty_proxy(extracted_nums)
            
            # Select sample matching majority answer as representative response
            selected_response = samples[0]
            for s, num in zip(samples, extracted_nums):
                if num == maj_answer:
                    selected_response = s
                    break
                    
            item["total_calls"] += config.SAMPLES_PER_STEP
            turn_record = {
                "step": step + 1,
                "auditor": auditor_label,
                "samples": samples,
                "extracted_nums": extracted_nums,
                "majority_answer": maj_answer,
                "agreement_ratio": agreement,
                "uncertainty": uncertainty,
                "response": selected_response,
                "is_correct": (maj_answer == item["ground_truth_num"])
            }
            item["history"].append(turn_record)
            
            # Early stopping check: High agreement proxy
            if agreement >= config.HIGH_AGREEMENT_THRESHOLD:
                item["completed"] = True

    return active_items