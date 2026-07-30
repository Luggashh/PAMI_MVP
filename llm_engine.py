import asyncio
from typing import List
from config import config
from ollama_client import generate_from_ollama

class ParallelAuditEngine:
    def __init__(self, model_name: str):
        print(f"[LLMEngine] Initializing Async Ollama Engine for model: {model_name} on Port 11435...")
        self.model_name = model_name

    def generate_batch(self, prompts: List[str]) -> List[List[str]]:
        """
        Synchronous wrapper that executes the asynchronous batch generation.
        """
        return asyncio.run(self._generate_batch_async(prompts))

    async def _generate_batch_async(self, prompts: List[str]) -> List[List[str]]:
        """
        Asynchronously fetches 5 samples per question concurrently.
        """
        batch_results = []
        
        # Iterate over all prompts in the current batch
        for prompt in prompts:
            # Create 5 concurrent async tasks for the current question
            tasks = [
                generate_from_ollama(model=self.model_name, prompt=prompt)
                for _ in range(config.SAMPLES_PER_STEP)
            ]
            
            # Await all 5 samples to generate simultaneously
            question_samples = await asyncio.gather(*tasks)
            batch_results.append(question_samples)
            
        return batch_results