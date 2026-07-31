# ── Ollama Port & URL Configuration ──────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
VLLM_BASE_URLS = ["http://localhost:11434/v1"]
MODEL_NAME = "llama3.2"

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
GSM8K_SPLIT = "test"
NUM_EXAMPLES = None

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True

# ── Concurrency ──────────────────────────────────────────────
MAX_WORKERS = 32
