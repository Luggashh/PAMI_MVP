


# ── vLLM / OpenAI-compatible Settings ────────────────────────
# vLLM serves an OpenAI-compatible API on this base URL.
# We run 4 vLLM instances (one per GPU) behind a simple round-robin.
VLLM_BASE_URLS = [
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]
MODEL_NAME = "llama-3.2.3:b"

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
NUM_EXAMPLES = None          # None = full train set (7473 examples)

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True

# ── Concurrency ──────────────────────────────────────────────
MAX_WORKERS = 64             # Tuned for 4× A100s with vLLM batching

