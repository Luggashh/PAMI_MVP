#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.

Usage:
    python main.py --output_dir results/ --max_workers 32

This runs the full pipeline:
    1. Load GSM8K train examples
    2. For each example, run the iterative audit loop (Model A -> B -> A -> B)
    3. At each step, sample 5 times for majority-vote uncertainty
    4. Evaluate and report results
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import OUTPUT_DIR 
from data_loader import load_gsm8k
from ollama_client import check_ollama_ready
from audit_loop import run_audit_loop
from evaluation import evaluate_results

def process_example_with_retry(example_data, max_retries=8, initial_backoff=2):
    """Worker function to process a single example concurrently with aggressive retry/timeout logic."""
    i, example = example_data
    
    for attempt in range(max_retries):
        try:
            # The run_audit_loop will execute the A->B->A->B logic 
            # and perform the 5 samples per step for uncertainty.
            result = run_audit_loop(question=example["question"])
            result["gold_answer"] = example["answer"]
            result["example_idx"] = i
            return result
        except Exception as e:
            error_msg = str(e).lower()
            # If we hit the max retries, return a failed stub to prevent pipeline crash
            if attempt == max_retries - 1:
                tqdm.write(f"❌ Failed processing example {i} after {max_retries} attempts. Error: {e}")
                return {
                    "example_idx": i,
                    "gold_answer": example["answer"],
                    "question": example["question"],
                    "error": str(e)
                }
            
            # Exponential backoff (e.g., 2s, 4s, 8s, 16s...) to let the GPU queue recover
            sleep_time = initial_backoff ** attempt
            time.sleep(sleep_time)

def main():
    parser = argparse.ArgumentParser(
        description="Iterative Audit Loops with Language Models (GSM8K)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None, # None forces the entire train subset
        help="Number of GSM8K examples to evaluate (default: None for all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save results (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train", # Target the train subset
        help="GSM8K split to use (default: train)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32, # Concurrency limit tuned for 4x A100s
        help="Number of parallel threads for concurrent model inference",
    )
    args = parser.parse_args()

    # ── Preflight checks ─────────────────────────────────────────
    print("🔍 Checking Ollama availability...")
    if not check_ollama_ready():
        print("❌ Ollama is not ready. Please start it and pull the model.")
        sys.exit(1)
    print("✅ Ollama is ready.\n")

    # ── Load data ─────────────────────────────────────────────────
    print(f"📚 Loading GSM8K ({args.split}) — Full Dataset...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)
    print(f"   Loaded {len(examples)} examples.\n")

    # ── Run audit loops ──────────────────────────────────────────
    print(f"🔄 Running iterative audit loops concurrently with {args.max_workers} workers...\n")
    results = []
    start_time = time.time()

    # Enumerate examples to preserve original dataset order
    indexed_examples = list(enumerate(examples))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_example_with_retry, ex): ex for ex in indexed_examples}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing")):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                tqdm.write(f"❌ Unexpected Thread Error: {e}")

            # Progress update every 50 examples
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(examples) - i - 1) / rate
                tqdm.write(
                    f"   [{i+1}/{len(examples)}] "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"ETA: {remaining:.0f}s | "
                    f"Rate: {rate:.2f} ex/s"
                )

    total_time = time.time() - start_time
    print(f"\n⏱  Total time: {total_time:.1f}s "
          f"({total_time/len(examples):.1f}s per example)\n")

    # ── Evaluate and report ──────────────────────────────────────
    results.sort(key=lambda x: x["example_idx"])
    summary = evaluate_results(results, output_dir=args.output_dir)

    return summary

if __name__ == "__main__":
    main()