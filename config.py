import os
import subprocess
import time
import urllib.request

# ── Automatische Ollama Verwaltung (Über tar.zst Pfad im User-Space) ──
def ensure_ollama_running(base_url, model_name):
    if os.environ.get("OLLAMA_INITIALIZED") == "true":
        return

    # Pfad zur entpackten Binärdatei im Home-Verzeichnis
    home = os.path.expanduser("~")
    ollama_bin = os.path.join(home, "ollama_root", "bin", "ollama")
    ollama_lib = os.path.join(home, "ollama_root", "lib", "ollama")

    # 1. Prüfen, ob der Server bereits aktiv ist
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2):
            os.environ["OLLAMA_INITIALIZED"] = "true"
            return
    except Exception:
        print("\n[Ollama] 🟡 Server läuft nicht. Starte automatischen Setup aus tar.zst...")
        
        # 2. Sicherheitsprüfung: Existiert die entpackte Datei?
        if not os.path.exists(ollama_bin):
            raise FileNotFoundError(
                f"[Ollama] 🔥 Kritischer Fehler: Die Binärdatei wurde unter '{ollama_bin}' nicht gefunden! "
                "Haben Sie den 'tar -C ~/ollama_root -xf ollama.tar.zst' Befehl ausgeführt?"
            )
        
        # Sicherstellen, dass Ausführungsrechte vorliegen
        os.chmod(ollama_bin, 0o755)
        
        # 3. Server im Hintergrund starten
        print("[Ollama] 🚀 Starte Ollama-Server im Hintergrund...")
        log_file = open(os.path.join(home, "ollama_root", "ollama.log"), "w")
        
        # Umgebungsvariablen setzen: Parallelität erzwingen & Pfad zu den GPU-Bibliotheken mitgeben
        env = os.environ.copy()
        env["OLLAMA_NUM_PARALLEL"] = "64"
        if os.path.exists(ollama_lib):
            env["LD_LIBRARY_PATH"] = f"{ollama_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        
        subprocess.Popen([ollama_bin, "serve"], stdout=log_file, stderr=log_file, env=env)
        
        # Warten, bis der Port antwortet
        for i in range(15):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1):
                    print("[Ollama] 🟢 Server antwortet und ist bereit.")
                    break
            except Exception:
                if i == 14:
                    print("[Ollama] ❌ Server konnte nicht rechtzeitig gestartet werden. Siehe ~/ollama_root/ollama.log")
                pass
                
    # 4. Modell für die Audit-Schleife bereitstellen
    try:
        print(f"[Ollama] 📦 Überprüfe Modell '{model_name}'...")
        subprocess.run([ollama_bin, "pull", model_name], check=True)
        print(f"[Ollama] ✅ Modell '{model_name}' ist einsatzbereit.\n")
    except Exception as e:
        print(f"[Ollama] ⚠️ Fehler beim Vorbereiten des Modells: {e}\n")

    os.environ["OLLAMA_INITIALIZED"] = "true"

# ── Ollama Settings ──────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

# Triggert den Start
ensure_ollama_running(OLLAMA_BASE_URL, MODEL_NAME)

# ── Generation Parameters ────────────────────────────────────
# ... (Hier unverändert Ihre restlichen Parameter wie TEMPERATURE, MAX_WORKERS etc. belassen)


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
