import os
import subprocess
import time
import shutil
import urllib.request

# ── Automatische Ollama Verwaltung (Ohne Sudo) ────────────────
def ensure_ollama_running(base_url, model_name):
    # Pfade im Home-Verzeichnis definieren
    home = os.path.expanduser("~")
    ollama_dir = os.path.join(home, "ollama")
    ollama_bin = os.path.join(ollama_dir, "ollama")
    
    # 1. Prüfen, ob Ollama bereits unter der URL erreichbar ist
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2):
            print("🟢 Ollama-Server läuft bereits.")
    except Exception:
        print("🟡 Ollama läuft nicht. Starte automatische Einrichtung...")
        
        # 2. Falls nicht installiert, ohne Sudo herunterladen
        if not os.path.exists(ollama_bin):
            print("📥 Ollama-Binärdatei wird heruntergeladen...")
            os.makedirs(ollama_dir, exist_ok=True)
            url = "https://ollama.com"
            urllib.request.urlretrieve(url, ollama_bin)
            os.chmod(ollama_bin, 0o755)
            print("✅ Download abgeschlossen.")
        
        # 3. Ollama Server im Hintergrund starten
        print("🚀 Starte Ollama-Server im Hintergrund...")
        log_file = open(os.path.join(ollama_dir, "ollama.log"), "w")
        
        # Umgebungsvariablen für Parallelität setzen (passend zu Ihren MAX_WORKERS)
        env = os.environ.copy()
        env["OLLAMA_NUM_PARALLEL"] = "64"
        
        subprocess.Popen([ollama_bin, "serve"], stdout=log_file, stderr=log_file, env=env)
        
        # Warten bis der Server antwortet
        for _ in range(10):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1):
                    print("🟢 Ollama-Server erfolgreich gestartet.")
                    break
            except Exception:
                pass
                
    # 4. Prüfen, ob das gewünschte Modell bereits geladen ist, sonst "pullen"
    try:
        # Falls Ollama lokal liegt, direkt die Binärdatei zum Pullen nutzen
        if os.path.exists(ollama_bin):
            print(f"📦 Überprüfe Modell '{model_name}'...")
            # Ein simpler 'pull' Befehl überspringt den Download, wenn das Modell schon da ist
            subprocess.run([ollama_bin, "pull", model_name], check=True)
            print(f"✅ Modell '{model_name}' ist einsatzbereit.")
    except Exception as e:
        print(f"⚠️ Fehler beim Laden des Modells: {e}")

# ── Ollama Settings ──────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

# Automatischen Start triggern, sobald diese Config geladen wird
ensure_ollama_running(OLLAMA_BASE_URL, MODEL_NAME)

# ── Generation Parameters ────────────────────────────────────
TEMPERATURE = 0.7
TEMPERATURE_GREEDY = 0.0
MAX_TOKENS = 1024
SEED_MAIN = 42

# ── Audit Loop Settings ──────────────────────────────────────
MAX_STEPS = 4
UNCERTAINTY_SAMPLES = 5
AGREEMENT_THRESHOLD = 4

# ── Dataset Settings ─────────────────────────────────────────
GSM8K_SPLIT = "train"
NUM_EXAMPLES = 10            # 10 für den schnellen Testlauf

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True

# ── Concurrency ──────────────────────────────────────────────
MAX_WORKERS = 64             # Passt perfekt zu OLLAMA_NUM_PARALLEL=64
