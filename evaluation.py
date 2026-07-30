import json
from typing import List, Dict, Any
from config import config

def evaluate_audit_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates all requested metrics."""
    total = len(results)
    step_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    step_total = {1: 0, 2: 0, 3: 0, 4: 0}
    
    total_calls = 0
    false_certainties = []
    uncertainty_drops_wrong = []

    for item in results:
        total_calls += item["total_calls"]
        history = item["history"]
        
        for turn in history:
            step = turn["step"]
            step_total[step] += 1
            if turn["is_correct"]:
                step_correct[step] += 1
                
        final = history[-1]
        
        # False-certainty cases
        if final["agreement_ratio"] == 1.0 and not final["is_correct"]:
            false_certainties.append(item)
            
        # Uncertainty drops but final answer is wrong
        if len(history) > 1:
            initial_uncertainty = history[0]["uncertainty"]
            final_uncertainty = final["uncertainty"]
            if final_uncertainty < initial_uncertainty and not final["is_correct"]:
                uncertainty_drops_wrong.append(item)

    step_accuracies = {
        f"Step_{s}_Accuracy": (step_correct[s] / step_total[s]) if step_total[s] > 0 else 0.0
        for s in range(1, 5)
    }

    summary = {
        "Total_Questions": total,
        "Average_Calls_Per_Question": total_calls / total if total > 0 else 0,
        "Step_Accuracies": step_accuracies,
        "False_Certainty_Count": len(false_certainties),
        "Uncertainty_Drop_Wrong_Count": len(uncertainty_drops_wrong)
    }

    with open(f"{config.OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(f"{config.OUTPUT_DIR}/false_certainty.json", "w") as f:
        json.dump(false_certainties, f, indent=4)
    with open(f"{config.OUTPUT_DIR}/uncertainty_drops_wrong.json", "w") as f:
        json.dump(uncertainty_drops_wrong, f, indent=4)

    return summary