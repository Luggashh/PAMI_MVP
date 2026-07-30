from data_loader import load_full_gsm8k
from llm_engine import ParallelAuditEngine
from audit_loop import execute_audit_loop
from evaluation import evaluate_audit_results
from config import config

def main():
    print("=== Starting Iterative Audit Loop Project (100% GSM8K) ===")
    
    # 1. Load 100% GSM8K Dataset (Unpartitioned)
    dataset = load_full_gsm8k()
    
    # 2. Initialize vLLM engines for Model A and Model B
    engine_a = ParallelAuditEngine(config.MODEL_A_NAME)
    engine_b = ParallelAuditEngine(config.MODEL_B_NAME)
    
    # 3. Execute 4-Step Iterative Audit Loop
    audit_results = execute_audit_loop(dataset, engine_a, engine_b)
    
    # 4. Generate & Save Metrics
    summary = evaluate_audit_results(audit_results)
    
    print("\n=== Execution Summary ===")
    print(f"Total Questions Evaluated: {summary['total_questions_evaluated']}")
    print(f"Average LLM Calls / Question: {summary['average_llm_calls_per_question']:.2f}")
    print("Step Accuracies:")
    for step, acc in summary['step_accuracies'].items():
        print(f"  - {step}: {acc * 100:.2f}%")
    print(f"False Certainty Cases: {summary['num_false_certainty_cases']}")
    print(f"Uncertainty Drop (Wrong Answer) Cases: {summary['num_uncertainty_drop_wrong_cases']}")
    print(f"Results saved to {config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()