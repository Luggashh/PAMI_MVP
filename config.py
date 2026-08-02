import os
import subprocess
import time
import urllib.request

# ── Automatische Ollama Verwaltung (Stabiles AMD Multi-GPU Setup) ──
def ensure_ollama_running(base_url, model_name):
    if os.environ.get("OLLAMA_INITIALIZED") == "true":
        return

    home = os.path.expanduser("~")
    ollama_bin = os.path.join(home, "ollama_root", "bin", "ollama")
    ollama_lib = os.path.join(home, "ollama_root", "lib", "ollama")

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2):
            os.environ["OLLAMA_INITIALIZED"] = "true"
            return
    except Exception:
        print("\n[Ollama] 🟡 Server läuft nicht. Initialisiere AMD Multi-GPU Setup...")
        
        if not os.path.exists(ollama_bin):
            raise FileNotFoundError(f"[Ollama] Kritischer Fehler: Datei fehlt unter {ollama_bin}")
        
        os.chmod(ollama_bin, 0o755)
        log_file = open(os.path.join(home, "ollama_root", "ollama.log"), "w")
        
        env = os.environ.copy()
        env["ROCR_VISIBLE_DEVICES"] = "0,1,2,3" 
        
        if os.path.exists(ollama_lib):
            env["LD_LIBRARY_PATH"] = f"{ollama_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        
        # MASSIV WICHTIG FÜR AMD: Zwingt ROCm dazu, die Server-GPUs (wie Instinct/Pro) zu akzeptieren
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0" 
        
        # Kontrollierte Parallelität, um VRAM-Crashes bei 800 Aufgaben zu verhindern
        env["OLLAMA_NUM_PARALLEL"] = "16"  # 16 parallele Streams reichen bei 4 GPUs für extremen Speed

        print("[Ollama] 🚀 Starte Ollama-Server mit AMD-GPU-Support im Hintergrund...")
        subprocess.Popen([ollama_bin, "serve"], stdout=log_file, stderr=log_file, env=env)
        
        for i in range(15):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1):
                    print("[Ollama] 🟢 Server antwortet und AMD-GPUs sind initialisiert.")
                    break
            except Exception:
                if i == 14:
                    print("[Ollama] ❌ Start fehlgeschlagen. Siehe ~/ollama_root/ollama.log")
                pass
                
    try:
        print(f"[Ollama] 📦 Überprüfe Modell '{model_name}'...")
        subprocess.run([ollama_bin, "pull", model_name], check=True)
        print(f"[Ollama] ✅ Modell '{model_name}' ist auf den GPUs einsatzbereit.\n")
    except Exception as e:
        print(f"[Ollama] ⚠️ Fehler beim Laden des Modells: {e}\n")

    os.environ["OLLAMA_INITIALIZED"] = "true"

# ── Ollama Settings ──────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

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

# ── Dataset Settings (Konfiguriert für Ihren 800-Aufgaben-Run) ──
GSM8K_SPLIT = "train"
NUM_EXAMPLES = 800           # Auf 800 hochgesetzt

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True

# ── Concurrency ──────────────────────────────────────────────
MAX_WORKERS = 16             # Reduziert von 64 auf 16, um Timeouts bei Großrechnungen zu verhindern
