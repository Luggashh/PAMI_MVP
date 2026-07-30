import json
from typing import List, Dict, Any
from config import config

def evaluate_audit_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes required project metrics:
      1. Accuracy per step
      2. Average number of LLM calls
      3. False-certainty cases
      4. Cases where uncertainty drops but final answer is still wrong
    """
    total_questions = len(results)
    step_correct_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    step_total_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    total_llm_calls = 0
    false_certainty_cases = []
    uncertainty_drop_wrong_cases = []

    for item in results:
        total_llm_calls += item["total_calls"]
        history = item["history"]
        
        # Evaluate step-by-step metrics
        for turn in history:
            step = turn["step"]
            step_total_counts[step] += 1
            if turn["is_correct"]:
                step_correct_counts[step] += 1
                
        # Final turn state
        final_turn = history[-1]
        
        # Metric 3: False-certainty cases[cite: 1]
        # High agreement / zero uncertainty (agreement = 1.0), but answer is incorrect
        if final_turn["agreement_ratio"] == 1.0 and not final_turn["is_correct"]:
            false_certainty_cases.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth_num"],
                "predicted": final_turn["majority_answer"],
                "agreement": final_turn["agreement_ratio"],
                "history": history
            })
            
        # Metric 4: Uncertainty drops (agreement increases), but final answer is wrong[cite: 1]
        if len(history) > 1:
            initial_uncertainty = history[0]["uncertainty"]
            final_uncertainty = final_turn["uncertainty"]
            
            if final_uncertainty < initial_uncertainty and not final_turn["is_correct"]:
                uncertainty_drop_wrong_cases.append({
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth_num"],
                    "predicted": final_turn["majority_answer"],
                    "initial_uncertainty": initial_uncertainty,
                    "final_uncertainty": final_uncertainty,
                    "history": history
                })

    step_accuracies = {
        f"step_{s}_accuracy": (step_correct_counts[s] / step_total_counts[s]) if step_total_counts[s] > 0 else 0.0
        for s in range(1, 5)
    }

    avg_calls = total_llm_calls / total_questions if total_questions > 0 else 0.0

    summary = {
        "total_questions_evaluated": total_questions,
        "average_llm_calls_per_question": avg_calls,
        "step_accuracies": step_accuracies,
        "num_false_certainty_cases": len(false_certainty_cases),
        "num_uncertainty_drop_wrong_cases": len(uncertainty_drop_wrong_cases)
    }

    # Save detailed JSON reports
    with open(f"{config.OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{config.OUTPUT_DIR}/false_certainty_cases.json", "w") as f:
        json.dump(false_certainty_cases, f, indent=2)

    with open(f"{config.OUTPUT_DIR}/uncertainty_drop_wrong.json", "w") as f:
        json.dump(uncertainty_drop_wrong_cases, f, indent=2)

    return summary