import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from ollama import AsyncClient, ResponseError

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435")
MODEL_A = os.getenv("MODEL_A", "llama3")  # Replace with Model A name (e.g., llama3, qwen2.5)
MAX_CONCURRENCY = 32  # Saturated queue for 4x A100 GPUs
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---
def extract_numerical_answer(text: str) -> Optional[str]:
    """
    Extracts the numerical answer from text.
    Looks for standard GSM8K '#### <number>' pattern, or extracts the last standalone number.
    """
    if not text:
        return None
    
    # 1. Look for explicit GSM8K answer delimiter "#### <number>"
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # 2. Look for "The answer is <number>" pattern
    match = re.search(r"(?:final answer|answer is)\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # 3. Fallback: extract the very last number in the generated text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None

def normalize_number(num_str: Optional[str]) -> Optional[float]:
    """Converts string number to float for robust comparison."""
    if num_str is None:
        return None
    try:
        return float(num_str)
    except ValueError:
        return None

# --- ASYNC EVALUATOR CLASS ---
class ModelAEvaluator:
    def __init__(self, host: str, model_name: str, max_concurrency: int):
        self.client = AsyncClient(host=host)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_response(self, question: str, retries: int = 3) -> str:
        """Queries Model A asynchronously with retry backoff."""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a mathematical reasoning assistant. Solve the following problem step-by-step. "
                    "Show your clear reasoning trace, and end your final response with 'The answer is #### <number>'."
                )
            },
            {"role": "user", "content": question}
        ]

        for attempt in range(retries):
            async with self.semaphore:
                try:
                    response = await self.client.chat(
                        model=self.model_name,
                        messages=prompt,
                        options={"temperature": 0.0}  # Deterministic decoding for Model A baseline
                    )
                    return response['message']['content']
                except ResponseError as e:
                    if e.status_code == 404:
                        logging.info(f"Model '{self.model_name}' not found. Pulling...")
                        await self.client.pull(self.model_name)
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logging.warning(f"Error on attempt {attempt+1}: {e}")
                    await asyncio.sleep(2 ** attempt)

        return ""

    async def process_item(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Processes a single GSM8K item."""
        question = item['question']
        raw_gold = item['answer']
        gold_num_str = extract_numerical_answer(raw_gold)
        gold_val = normalize_number(gold_num_str)

        start_time = time.time()
        cot_response = await self.generate_response(question)
        latency = time.time() - start_time

        pred_num_str = extract_numerical_answer(cot_response)
        pred_val = normalize_number(pred_num_str)

        is_correct = (gold_val is not None) and (pred_val is not None) and (abs(gold_val - pred_val) < 1e-4)

        return {
            "id": idx,
            "question": question,
            "gold_raw": raw_gold,
            "gold_num": gold_num_str,
            "model_a_cot": cot_response,
            "model_a_pred_num": pred_num_str,
            "is_correct": is_correct,
            "latency_seconds": round(latency, 3)
        }

# --- MAIN RUNNER ---
async def main():
    print(f"=== Starting GSM8K Benchmark for Model A ONLY ({MODEL_A}) ===")
    print(f"Server Host: {OLLAMA_HOST}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENCY}")

    # Load GSM8K dataset (test split = 1,319 questions)
    logging.info("Loading GSM8K test dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    total_questions = len(dataset)
    logging.info(f"Loaded {total_questions} questions.")

    evaluator = ModelAEvaluator(host=OLLAMA_HOST, model_name=MODEL_A, max_concurrency=MAX_CONCURRENCY)

    start_time = time.time()
    tasks = [evaluator.process_item(item, idx) for idx, item in enumerate(dataset)]
    
    # Run all GSM8K evaluations concurrently
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_time

    # Calculate Metrics
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / total_questions) * 100
    avg_latency = sum(r['latency_seconds'] for r in results) / total_questions

    summary = {
        "model_name": MODEL_A,
        "eval_type": "Model A Baseline Only (No Iterative Audit)",
        "total_samples": total_questions,
        "correct_samples": correct_count,
        "accuracy_percentage": round(accuracy, 2),
        "total_execution_time_seconds": round(total_duration, 2),
        "avg_sample_latency_seconds": round(avg_latency, 3),
        "throughput_samples_per_sec": round(total_questions / total_duration, 2)
    }

    # Save Detailed Results JSON
    results_file = os.path.join(RESULTS_DIR, "model_a_full_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save Summary JSON
    summary_file = os.path.join(RESULTS_DIR, "model_a_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CoT Traces
    traces_file = os.path.join(RESULTS_DIR, "model_a_cot_traces.md")
    with open(traces_file, "w", encoding="utf-8") as f:
        f.write(f"# Model A Standalone Reasoning Traces ({MODEL_A})\n\n")
        for r in results[:50]:  # Save first 50 detailed traces to markdown for inspection
            f.write(f"### Question {r['id'] + 1}\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Model A CoT Response:**\n{r['model_a_cot']}\n\n")
            f.write(f"**Predicted Num:** `{r['model_a_pred_num']}` | **Gold Num:** `{r['gold_num']}` | **Correct:** `{r['is_correct']}`\n\n")
            f.write("---\n\n")

    print("\n" + "="*50)
    print("=== MODEL A BASELINE EVALUATION COMPLETE ===")
    print("="*50)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
    print(f"Total Time: {total_duration:.2f}s ({total_questions / total_duration:.2f} samples/sec)")
    print(f"Full results saved to: {results_file}")
    print(f"Summary saved to:      {summary_file}")
    print(f"Traces saved to:       {traces_file}")

if __name__ == "__main__":
    asyncio.run(main())