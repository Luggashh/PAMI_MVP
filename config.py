"""
Configuration for the Iterative Audit Loop pipeline.
"""

# ── Ollama Settings ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

# We use the same model for both A and B, but with different system prompts.
# This mirrors the LMvLM (Cohen et al., 2023) approach of using the same LM
# with different prompts for examiner vs. examinee roles.

# ── Generation Parameters ────────────────────────────────────────
TEMPERATURE = 0.7          # For diversity in uncertainty sampling
TEMPERATURE_GREEDY = 0.0   # For the main proposal/audit pass
MAX_TOKENS = 512
SEED_MAIN = 42             # Deterministic main pass
# Uncertainty samples use no seed (stochastic)

# ── Audit Loop Settings ─────────────────────────────────────────
MAX_STEPS = 4              # Maximum total answers before stopping
UNCERTAINTY_SAMPLES = 5    # Number of samples per step for majority vote
AGREEMENT_THRESHOLD = 4    # Out of 5 samples: if >= 4 agree, stop early

# ── Dataset Settings ─────────────────────────────────────────────
GSM8K_SPLIT = "test"
NUM_EXAMPLES = 50          # Default number of examples to evaluate

# ── Output ───────────────────────────────────────────────────────
OUTPUT_DIR = "results"