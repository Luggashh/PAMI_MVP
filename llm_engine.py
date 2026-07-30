from typing import List
from config import config

# Import your custom Ollama client here. 
# (Adjust the import statement below to match the exact class/function inside your ollama_client.py)
from ollama_client import generate_from_ollama 

class ParallelAuditEngine:
    def __init__(self, model_name: str):
        print(f"[LLMEngine] Initializing Ollama Engine for model: {model_name}...")
        self.model_name = model_name

    def generate_batch(self, prompts: List[str]) -> List[List[str]]:
        """
        Since Ollama does not natively support vLLM's massive parallel batching out-of-the-box,
        we iterate through the prompts and request samples via your Ollama client.
        """
        batch_results = []
        
        # Iterate over all prompts in the current batch
        for prompt in prompts:
            question_samples = []
            
            # Generate the requested number of samples (e.g., 5) per question
            for _ in range(config.SAMPLES_PER_STEP):
                # Replace 'generate_from_ollama' with the actual function from your ollama_client.py
                response = generate_from_ollama(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=config.TEMPERATURE
                )
                question_samples.append(response)
                
            batch_results.append(question_samples)
            
        return batch_results