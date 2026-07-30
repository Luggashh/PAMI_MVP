from vllm import LLM, SamplingParams
from typing import List
from config import config

class ParallelAuditEngine:
    """
    High-throughput vLLM engine leveraging Tensor Parallelism across 4x A100 GPUs.
    """
    def __init__(self, model_name: str):
        print(f"[LLMEngine] Initializing vLLM with model: {model_name} on {config.TENSOR_PARALLEL_SIZE} GPUs...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
            trust_remote_code=True,
        )

    def generate_samples(self, prompts: List[str], num_samples: int = 5, temperature: float = 0.7) -> List[List[str]]:
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=1024,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
        outputs = self.llm.generate(prompts, sampling_params, show_progress_bar=False)
        
        batch_results = []
        for output in outputs:
            samples = [out.text for out in output.outputs]
            batch_results.append(samples)
            
        return batch_results