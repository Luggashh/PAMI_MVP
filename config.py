# ── Ollama Settings ──────────────────────────────────────────
# Die URL Ihres lokal gestarteten Ollama-Servers
OLLAMA_BASE_URL = "http://localhost:11434"

# Der permanente, lokale Name des Modells in Ollama
MODEL_NAME = "llama3.2:3b"

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
