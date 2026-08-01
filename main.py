#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.

Usage:
    python main.py --num_examples 50 --output_dir results/

This runs the full pipeline:
    1. Spin up 4 background vLLM instances (one per GPU)
    2. Load GSM8K test examples
    3. For each example, run the iterative audit loop (Model A → B → A → B)
    4. At each step, sample 5 times for majority-vote uncertainty
    5. Evaluate and report results
    6. Clean up background server processes
"""

import argparse
import json
import sys
import time
import subprocess
import atexit

from tqdm import tqdm

from config import NUM_EXAMPLES, OUTPUT_DIR, GSM8K_SPLIT
from data_loader import load_gsm8k
from ollama_client import check_ollama_ready
from audit_loop import run_audit_loop
from evaluation import evaluate_results

# Liste, um die Hintergrundprozesse der Server zu tracken
vllm_processes = []

def start_vllm_servers():
    """Startet 4 vLLM-Instanzen im Hintergrund auf den GPUs 0 bis 3."""
    print("🚀 Starting 4 vLLM instances in background (4x A100 GPUs)...")
    
    ports_and_gpus = [(8000, 0), (8001, 1), (8002, 2), (8003, 3)]
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    for port, gpu in ports_and_gpus:
        # Konstruktion des CLI-Befehls
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--port", str(port),
            "--model", model_name
        ]
        
        # Umgebungsvariablen kopieren und CUDA_VISIBLE_DEVICES setzen
        env = dict(subprocess.os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        # Prozess asynchron im Hintergrund starten
        proc = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.DEVNULL, # Verhindert, dass Server-Logs den Output überfluten
            stderr=subprocess.DEVNULL
        )
        vllm_processes.append(proc)
        print(f"   -> vLLM on GPU {gpu} (Port {port}) is spinning up...")

    # Den Servern Zeit geben, das Modell in den GPU-Speicher zu laden
    print("⏳ Waiting 45 seconds for models to load into GPU memory...")
    time.sleep(45)

def cleanup_vllm_servers():
    """Schließt alle Hintergrund-Server, wenn das Skript beendet wird."""
    if vllm_processes:
        print("\n🛑 Terminating vLLM background instances...")
        for proc in vllm_processes:
            proc.terminate()
            proc.wait()
        print("✅ All background server processes successfully terminated.")

# Registriert den Cleanup-Mechanismus für ein sauberes Beenden bei Skriptende/Abbruch
atexit.register(cleanup_vllm_servers)

def main():
    parser = argparse.ArgumentParser(
        description="Iterative Audit Loops with Language Models (GSM8K)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=NUM_EXAMPLES,
        help=f"Number of GSM8K examples to evaluate (default: {NUM_EXAMPLES})",
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
        default=GSM8K_SPLIT,
        help=f"GSM8K split to use (default: {GSM8K_SPLIT})",
    )
    args = parser.parse_args()

    # ── Start background servers ──────────────────────────────────
    start_vllm_servers()

    # ── Preflight checks ─────────────────────────────────────────
    print("🔍 Checking vLLM cluster availability...")
    if not check_ollama_ready():
        print("❌ vLLM cluster is not ready. Shutting down.")
        sys.exit(1)
    print("✅ vLLM cluster is ready.\n")

    # ── Load data ─────────────────────────────────────────────────
    print(f"📚 Loading GSM8K ({args.split}) — {args.num_examples} examples...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)
    print(f"   Loaded {len(examples)} examples.\n")

    # ── Run audit loops ──────────────────────────────────────────
    print("🔄 Running iterative audit loops...\n")
    results = []
    start_time = time.time()

    for i, example in enumerate(tqdm(examples, desc="Processing")):
        result = run_audit_loop(question=example["question"])
        result["gold_answer"] = example["answer"]
        result["example_idx"] = i
        results.append(result)

        # Progress update every 10 examples
        if (i + 1) % 10 == 0:
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
    summary = evaluate_results(results, output_dir=args.output_dir)

    return summary

if __name__ == "__main__":
    main()
