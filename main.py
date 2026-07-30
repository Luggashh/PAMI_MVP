import os
from data_loader import load_full_gsm8k
from llm_engine import ParallelAuditEngine
from audit_loop import execute_batch_audit_loop
from evaluation import evaluate_audit_results
from config import config

# Critical for preventing multi-GPU NCCL hanging when using multiple vLLM engines
os.environ["VLLM_WORKER_MULTIPROCESS_METHOD"] = "spawn"

def main():
    print("=== Starting Full GSM8K Iterative Audit (100% Unpartitioned) ===")
    
    dataset = load_full_gsm8k()
    
    # Initialize both engines (Memory split safely via config)
    engine_a = ParallelAuditEngine(config.MODEL_A_NAME)
    engine_b = ParallelAuditEngine(config.MODEL_B_NAME)
    
    results = execute_batch_audit_loop(dataset, engine_a, engine_b)
    summary = evaluate_audit_results(results)
    
    print("\n=== Final Audit Metrics ===")
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for k, v in val.items():
                print(f"  {k}: {v:.2%}")
        else:
            print(f"{key}: {val}")
            
    print(f"\nAll results saved to {os.path.abspath(config.OUTPUT_DIR)}")

if __name__ == "__main__":
    main()