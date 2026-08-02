#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.
Automated local Ollama Multi-GPU lifecycle management without sudo.
"""

import argparse
import json
import sys
import time
import subprocess
import os
import requests
from tqdm import tqdm

from config import NUM_EXAMPLES, OUTPUT_DIR, GSM8K_SPLIT, OLLAMA_BASE_URL
from data_loader import load_gsm8k
from ollama_client import check_ollama_ready, generate
from audit_loop import run_audit_loop
from evaluation import evaluate_results

def ensure_local_ollama_running():
    """Prüft, ob Ollama läuft. Wenn nicht, wird es mit Multi-GPU Support im Hintergrund gestartet."""
    try:
        # Versuche den Server zu pingen
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        if response.status_code == 200:
            print("✅ Ollama-Server läuft bereits und ist bereit.")
            return
    except requests.exceptions.ConnectionError:
        print("🚀 Ollama läuft noch nicht. Starte lokalen Server im Hintergrund...")
        
    # Pfade definieren (außerhalb des Git-Repositorys im Home-Verzeichnis)
    home_dir = os.path.expanduser("~")
    ollama_path = os.path.join(home_dir, "ollama_local", "bin", "ollama")
    models_path = os.path.join(home_dir, "ollama_local", "models")
    log_path = "ollama_server.log"

    if not os.path.exists(ollama_path):
        print(f"❌ FEHLER: Das Ollama-Binary wurde unter {ollama_path} nicht gefunden!")
        print("👉 Bitte führen Sie einmalig die Download-Befehle im Home-Verzeichnis aus.")
        sys.exit(1)

    # Umgebungsvariablen für Multi-GPU und maximale Parallelisierung vorbereiten
    env = dict(os.environ)
    env["OLLAMA_MODELS"] = models_path
    env["OLLAMA_NUM_PARALLEL"] = "64"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Nutzt alle 4 A100-GPUs automatisch

    # Server als entkoppelten Hintergrundprozess starten (wie nohup)
    log_file = open(log_path, "w")
    subprocess.Popen(
        [ollama_path, "serve"],
        env=env,
        stdout=log_file,
        stderr=log_file,
        preexec_fn=os.setpgrp  # Verhindert, dass der Server stirbt, wenn main.py fertig ist
    )
    
    # Dem Server kurz Zeit geben zum Initialisieren
    print("⏳ Warte 5 Sekunden auf die Server-Initialisierung...")
    time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description="Iterative Audit Loops with Language Models (GSM8K)")
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--split", type=str, default=GSM8K_SPLIT)
    args = parser.parse_args()

    # ── Automatischer Server-Check & Start ────────────────────────
    ensure_local_ollama_running()

    # ── Preflight checks ─────────────────────────────────────────
    print("🔍 Überprüfe Modellverfügbarkeit im Cluster...")
    if not check_ollama_ready():
        print("❌ Das Modell llama3.2:3b ist noch nicht geladen.")
        print("⏳ Starte automatischen Download des Modells (dauert nur beim ersten Mal)...")
        
        home_dir = os.path.expanduser("~")
        ollama_bin = os.path.join(home_dir, "ollama_local", "bin", "ollama")
        
        # Führt den Pull-Befehl direkt aus
        subprocess.run([ollama_bin, "pull", "llama3.2:3b"])
        
        # Erneuter Check nach dem Download
        if not check_ollama_ready():
            print("❌ Download fehlgeschlagen. Bitte prüfen Sie ollama_server.log")
            sys.exit(1)
            
    print("✅ Ollama und das Modell llama3.2:3b sind startbereit.\n")

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

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(examples) - i - 1) / rate
            tqdm.write(f"   [{i+1}/{len(examples)}] Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s | Rate: {rate:.2f} ex/s")

    total_time = time.time() - start_time
    print(f"\n⏱  Total time: {total_time:.1f}s ({total_time/len(examples):.1f}s per example)\n")

    summary = evaluate_results(results, output_dir=args.output_dir)
    return summary

if __name__ == "__main__":
    main()
