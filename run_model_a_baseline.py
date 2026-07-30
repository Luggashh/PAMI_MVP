import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional
from datasets import load_dataset, concatenate_datasets
from ollama import AsyncClient, ResponseError

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435")
MODEL_A = os.getenv("MODEL_A", "llama3.2:3b")  # Model A name
MAX_CONCURRENCY = 32  # Concurrency limit for 4x A100 GPUs
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
                        options={"temperature": 0.0}  # Deterministic decoding
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
        split_origin = item.get('split', 'unknown')
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
            "split": split_origin,
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
    print(f"=== Starting Combined GSM8K (Train + Test) Benchmark for Model A ({MODEL_A}) ===")
    print(f"Server Host: {OLLAMA_HOST}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENCY}")

    # 1. Load both Train (7,473) and Test (1,319) splits
    logging.info("Loading GSM8K train and test datasets...")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Add split tag to metadata
    train_dataset = train_dataset.add_column("split", ["train"] * len(train_dataset))
    test_dataset = test_dataset.add_column("split", ["test"] * len(test_dataset))

    # Combine into full 8,792 dataset
    full_dataset = concatenate_datasets([train_dataset, test_dataset])
    total_questions = len(full_dataset)
    logging.info(f"Successfully combined splits: Total {total_questions} questions ({len(train_dataset)} train + {len(test_dataset)} test).")

    evaluator = ModelAEvaluator(host=OLLAMA_HOST, model_name=MODEL_A, max_concurrency=MAX_CONCURRENCY)

    start_time = time.time()
    tasks = [evaluator.process_item(item, idx) for idx, item in enumerate(full_dataset)]
    
    # Process all 8,792 tasks asynchronously
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_time

    # Calculate Overall Metrics
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / total_questions) * 100

    # Calculate Split-specific Metrics
    train_results = [r for r in results if r['split'] == 'train']
    test_results = [r for r in results if r['split'] == 'test']

    train_correct = sum(1 for r in train_results if r['is_correct'])
    test_correct = sum(1 for r in test_results if r['is_correct'])

    summary = {
        "model_name": MODEL_A,
        "eval_type": "Model A Standalone (Full Train + Test GSM8K)",
        "overall": {
            "total_samples": total_questions,
            "correct_samples": correct_count,
            "accuracy_percentage": round(accuracy, 2)
        },
        "train_split": {
            "total_samples": len(train_results),
            "correct_samples": train_correct,
            "accuracy_percentage": round((train_correct / len(train_results)) * 100, 2)
        },
        "test_split": {
            "total_samples": len(test_results),
            "correct_samples": test_correct,
            "accuracy_percentage": round((test_correct / len(test_results)) * 100, 2)
        },
        "performance": {
            "total_execution_time_seconds": round(total_duration, 2),
            "avg_sample_latency_seconds": round(sum(r['latency_seconds'] for r in results) / total_questions, 3),
            "throughput_samples_per_sec": round(total_questions / total_duration, 2)
        }
    }

    # Save Results
    results_file = os.path.join(RESULTS_DIR, "model_a_full_gsm8k_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_file = os.path.join(RESULTS_DIR, "model_a_full_gsm8k_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("=== FULL GSM8K (8,792 SAMPLES) EVALUATION COMPLETE ===")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
    print(f"Train Split Acc:  {summary['train_split']['accuracy_percentage']}% ({train_correct}/{len(train_results)})")
    print(f"Test Split Acc:   {summary['test_split']['accuracy_percentage']}% ({test_correct}/{len(test_results)})")
    print(f"Total Time:       {total_duration:.2f}s ({total_questions / total_duration:.2f} samples/sec)")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())