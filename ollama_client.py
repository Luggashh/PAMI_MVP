import asyncio
import logging
from typing import List, Dict, Optional
from ollama import AsyncClient, ResponseError

# Configure logging to track concurrent execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AsyncOllamaClient:
    def __init__(self, host: str = "http://localhost:11435", max_concurrent_requests: int = 32):
        """
        Initializes the asynchronous Ollama client.
        
        Args:
            host: The Ollama server URL.
            max_concurrent_requests: Limits the number of simultaneous active requests.
                                     Adjust this based on the OLLAMA_NUM_PARALLEL server setting.
        """
        self.client = AsyncClient(host=host)
        # Semaphore prevents overwhelming the Ollama server queue
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
    async def chat_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, retries: int = 3) -> Optional[str]:
        """
        Asynchronously generates a chat completion for a single prompt with retry logic.
        """
        for attempt in range(retries):
            async with self.semaphore:
                try:
                    response = await self.client.chat(
                        model=model,
                        messages=messages,
                        options={"temperature": temperature}
                    )
                    return response['message']['content']
                
                except ResponseError as e:
                    logging.error(f"Ollama API Error on attempt {attempt + 1}: {e.error}")
                    # Auto-pull the model if it hasn't been downloaded to the server yet
                    if e.status_code == 404:
                        logging.info(f"Model '{model}' not found. Attempting to pull...")
                        await self.client.pull(model)
                    
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except Exception as e:
                    logging.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
        
        logging.error(f"Failed to get response after {retries} attempts.")
        return None

    async def generate_batch(self, model: str, batch_messages: List[List[Dict[str, str]]], temperature: float = 0.7) -> List[Optional[str]]:
        """
        Processes a large batch of message lists concurrently.
        This is the primary method to use when looping over the GSM8K dataset.
        """
        logging.info(f"Starting concurrent generation for batch of {len(batch_messages)} items using model {model}.")
        
        # Create asynchronous tasks for every item in the batch
        tasks = [
            self.chat_completion(model=model, messages=msgs, temperature=temperature)
            for msgs in batch_messages
        ]
        
        # Execute tasks concurrently and gather results while respecting the Semaphore limits
        results = await asyncio.gather(*tasks)
        logging.info("Batch generation complete.")
        return results

# --- Example Usage for Testing ---
if __name__ == "__main__":
    async def test_client():
        client = AsyncOllamaClient(max_concurrent_requests=5)
        
        # Simulate a small batch from GSM8K
        prompts = [
            [{"role": "user", "content": "What is 2+2? Think step by step."}],
            [{"role": "user", "content": "If I have 3 apples and eat 1, how many are left? Think step by step."}],
            [{"role": "user", "content": "What is 10 multiplied by 5? Think step by step."}]
        ]
        
        results = await client.generate_batch(model="llama3", batch_messages=prompts)
        for i, res in enumerate(results):
            print(f"\n--- Result {i+1} ---\n{res}")

    # Run the test
    asyncio.run(test_client())