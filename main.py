#!/usr/bin/env python3
"""
Main entry point for the Iterative Audit Loop pipeline.
Automated local Ollama AMD Multi-GPU lifecycle management without sudo.
"""

import argparse
import json
import sys
import time
import subprocess
import os
import requests
from tqdm import tqdm

from config import NUM_EXAMPLES, OUTPUT_DIR, GSM8K_SPLIT, OLLAMA_BASE_URL, MODEL_NAME
from data_loader import load_gsm8k
from ollama_client import generate
from audit_loop import run_audit_loop
from evaluation import evaluate_results

def ensure_local_ollama_running():
    """Prüft, ob Ollama läuft. Wenn nicht, wird es mit AMD Multi-GPU Support im Hintergrund gestartet."""
    try:
        # Versuche den Server zu pingen (Nutzt den /api/tags Endpunkt, da Ollama auf der Base-URL oft 404 wirft)
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama-Server läuft bereits und ist bereit.")
            return
    except requests.exceptions.ConnectionError:
        print("🚀 Ollama läuft noch nicht. Starte lokalen AMD-Server im Hintergrund...")
        
    # Korrekte Pfade aus Ihrem entpackten tar.zst-Archiv im Home-Verzeichnis
    home_dir = os.path.expanduser("~")
    ollama_path = os.path.join(home_dir, "ollama_root", "bin", "ollama")
    ollama_lib = os.path.join(home_dir, "ollama_root", "lib", "ollama")
    log_path = os.path.join(home_dir, "ollama_root", "ollama_server.log")

    if not os.path.exists(ollama_path):
        print(f"❌ FEHLER: Das Ollama-Binary wurde unter {ollama_path} nicht gefunden!")
        sys.exit(1)

    # Umgebungsvariablen für AMD Multi-GPU und maximale Parallelisierung vorbereiten
    env = dict(os.environ)
    env["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"      # Nutzt alle 4 AMD-GPUs
    env["OLLAMA_NUM_PARALLEL"] = "16"            # Stabile Parallelität für 800 Aufgaben
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"   # Erzwingt AMD ROCm Erkennung im User-Space
    
    if os.path.exists(ollama_lib):
        env["LD_LIBRARY_PATH"] = f"{ollama_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    # Server als entkoppelten Hintergrundprozess starten
    log_file = open(log_path, "w")
    subprocess.Popen(
        [ollama_path, "serve"],
        env=env,
        stdout=log_file,
        stderr=log_file,
        preexec_fn=os.setpgrp  # Verhindert, dass der Server stirbt, wenn main.py fertig ist
    )
    
    # Dynamisch warten, bis der Server antwortet (maximal 15 Sekunden)
    print("⏳ Initialisiere AMD-Grafikkarten (Warte auf API)...")
    for _ in range(15):
        time.sleep(1)
        try:
            res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            if res.status_code == 200:
                print("🟢 AMD-GPUs erfolgreich initialisiert. Server ist bereit.")
                return
        except requests.exceptions.ConnectionError:
            pass
    print("⚠️ Server-Start verzögert sich. Versuche trotzdem fortzufahren...")

def check_and_pull_model():
    """Prüft, ob das Modell lokal vorhanden ist, andernfalls wird es gepullt."""
    home_dir = os.path.expanduser("~")
    ollama_bin = os.path.join(home_dir, "ollama_root", "bin", "ollama")
    
    print(f"🔍 Überprüfe Modellverfügbarkeit für '{MODEL_NAME}'...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m['name'] for m in response.json().get('models', [])]
        
        # Falls das Modell (oder Kurzform) bereits im Server registriert ist, überspringen
        if MODEL_NAME in models or f"{MODEL_NAME}:latest" in models:
            print(f"✅ Modell {MODEL_NAME} ist bereits geladen.")
            return
    except Exception:
        pass

    print(f"📥 Modell {MODEL_NAME} fehlt oder wird verifiziert. Starte Pull-Prozess...")
    subprocess.run([ollama_bin, "pull", MODEL_NAME], check=True)
    print(f"✅ Modell {MODEL_NAME} steht im GPU-Cluster bereit.\n")

def main():
    parser = argparse.ArgumentParser(description="Iterative Audit Loops with Language Models (GSM8K)")
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--split", type=str, default=GSM8K_SPLIT)
    args = parser.parse_args()

    # ── Automatischer Server-Check & Start ────────────────────────
    ensure_local_ollama_running()
    check_and_pull_model()

    # ── Load data ─────────────────────────────────────────────────
    print(f"📚 Loading GSM8K ({args.split}) — {args.num_examples} examples...")
    examples = load_gsm8k(split=args.split, num_examples=args.num_examples)
    print(f"   Loaded {len(examples)} examples.\n")

    # ── Run audit loops ──────────────────────────────────────────
    print("🔄 Running iterative audit loops...\n")
    results = []
    start_time = time.time()
    
    os.makedirs(args.output_dir, exist_ok=True)
    backup_file = os.path.join(args.output_dir, "results_partial.json")

    for i, example in enumerate(tqdm(examples, desc="Processing")):
        result = run_audit_loop(question=example["question"])
        result["gold_answer"] = example["answer"]
        result["example_idx"] = i
        results.append(result)

        # Inkrementelle Zwischenspeicherung alle 10 Aufgaben (Sicherheit bei 800 Runs!)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(examples) - i - 1) / rate
            tqdm.write(f"   [{i+1}/{len(examples)}] Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s | Rate: {rate:.2f} ex/s")
            
            # Backup wegschreiben
            with open(backup_file, "w") as f:
                json.dump(results, f, indent=4)

    total_time = time.time() - start_time
    print(f"\n⏱  Total time: {total_time:.1f}s ({total_time/len(examples):.1f}s per example)\n")

    summary = evaluate_results(results, output_dir=args.output_dir)
    
    # Temporäres Backup bei Erfolg entfernen
    if os.path.exists(backup_file):
        os.remove(backup_file)
        
    return summary

if __name__ == "__main__":
    main()
