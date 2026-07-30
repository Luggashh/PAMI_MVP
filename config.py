import os
from pydantic import BaseModel

class AuditConfig(BaseModel):
    # Models
    MODEL_A_NAME: str = "llama3.2:3b"
    MODEL_B_NAME: str = "llama3.2:3b"
    
    # GSM8K Dataset Settings
    DATASET_NAME: str = "openai/gsm8k"
    DATASET_CONFIG: str = "main"
    USE_FULL_DATASET: bool = True  # Combine train + test splits (100%)
    
    # Audit Loop Settings
    MAX_AUDIT_STEPS: int = 4       # Max 4 turns (A -> B -> A -> B)
    SAMPLES_PER_STEP: int = 5      # 5 samples per auditor step
    TEMPERATURE: float = 0.7       # Sampling temperature for uncertainty estimation
    HIGH_AGREEMENT_THRESHOLD: float = 0.8  # Stop early if agreement >= 80% (4/5)
    
    # Hardware & VLLM Optimization for 4x A100-40GB
    TENSOR_PARALLEL_SIZE: int = 4   # Distribute across 4 GPUs
    GPU_MEMORY_UTILIZATION: float = 0.90
    MAX_MODEL_LEN: int = 4096
    BATCH_SIZE: int = 512
    
    # Output Directory
    OUTPUT_DIR: str = "./results"

config = AuditConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)