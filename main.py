#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.

Usage:
    python main.py --num_examples 50 --output_dir results/

This runs the full pipeline:
    1. Spin up 4 background vLLM instances (one per GPU) using LOCAL model weights (Fully Offline)
    2. Load GSM8K test examples
    3. For each example, run the iterative audit loop (Model A → B → A → B)
    4. At each step, sample 5 times for majority-vote uncertainty
    5. Evaluate and report results
    6. Clean up background server processes and close file handles
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

# Listen, um die Hintergrundprozesse und offenen Log-Dateien der Server zu tracken
vllm_processes = []
vllm_log_files = []

def start_vllm_servers():
    """Startet 4 vLLM-Instanzen im Hintergrund unter Verwendung des lokalen Modell-Pfads."""
    print("🚀 Starting 4 vLLM instances in background (4x A100 GPUs Offline)...")
    
    ports_and_gpus = [(8000, 0), (8001, 1), (8002, 2), (8003, 3)]
    
    # Importiert den Pfad direkt aus Ihrer config.py (zeigt nun auf das lokale Verzeichnis)
    from config import MODEL_NAME 

    for port, gpu in ports_and_gpus:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--port", str(port),
            "--model", MODEL_NAME  # Nutzt den lokalen Pfad
        ]
        
        env = dict(subprocess.os.environ)

        # ZWINGT vLLM und Hugging Face Bibliotheken im lokalen Netzwerk / Offline zu bleiben
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        # Injektiert die jeweilige GPU-ID
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_file = open(f"vllm_port_{port}.log", "w")
        vllm_log_files.append(log_file)

        proc = subprocess.Popen(
            cmd, 
            env=env,
            stdout=log_file,
            stderr=log_file
        )
        vllm_processes.append(proc)
        print(f"   -> vLLM on GPU {gpu} (Port {port}) is spinning up offline... (Logs: vllm_port_{port}.log)")

    # Da keine Internetverbindung geprüft werden muss, laden lokale Modelle meist sehr schnell.
    # 45 Sekunden sind im Offline-Modus für ein 3B Modell auf A100 GPUs absolut ausreichend.
    print("⏳ Waiting 45 seconds for local models to load into GPU memory...")
    time.sleep(45)


def cleanup_vllm_servers():
    """Schließt alle Hintergrund-Server und Dateihandles, wenn das Skript beendet wird."""
    if vllm_processes:
        print("\n🛑 Terminating vLLM background instances...")
        for proc in vllm_processes:
            proc.terminate()
            proc.wait()
        print("✅ All background server processes successfully terminated.")
    
    # Offene Log-Dateien sauber schließen
    if vllm_log_files:
        for f in vllm_log_files:
            f.close()

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
