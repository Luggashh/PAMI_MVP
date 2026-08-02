# ── Model Settings ───────────────────────────────────────────
# Offizieller Hugging-Face-Name für das direkte Laden über Transformers
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

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
NUM_EXAMPLES = 10            # Geändert von None auf 10 für einen schnellen Testlauf

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True

# ── Concurrency ──────────────────────────────────────────────
MAX_WORKERS = 64             # Wird bei direkter Pipeline-Nutzung intern verwaltet
