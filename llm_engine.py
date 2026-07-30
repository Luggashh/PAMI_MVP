from vllm import LLM, SamplingParams
from typing import List
from config import config

class ParallelAuditEngine:
    def __init__(self, model_name: str):
        print(f"Initializing vLLM Engine: {model_name} on {config.TENSOR_PARALLEL_SIZE} GPUs...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
            trust_remote_code=True,
            enforce_eager=True # Helps prevent CUDA graph conflicts with dual engines
        )
        self.sampling_params = SamplingParams(
            n=config.SAMPLES_PER_STEP,
            temperature=config.TEMPERATURE,
            max_tokens=1024,
            stop=["<|endoftext|>", "<|im_end|>"]
        )

    def generate_batch(self, prompts: List[str]) -> List[List[str]]:
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        return [[out.text for out in output.outputs] for output in outputs]