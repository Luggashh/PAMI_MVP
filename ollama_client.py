"""
Client wrapper using Hugging Face Transformers directly.
Handles automated GPU sharding across 4x A100s via device_map='auto'.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, MAX_TOKENS

print("⏳ Loading model into GPU memory via Transformers (this may take a minute)...")

# Lädt Tokenizer und Modell direkt mit dem in config.py definierten Pfad/Namen
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    token="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"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto",            # Verteilt das Modell automatisch auf alle verfügbaren GPUs
    torch_dtype=torch.bfloat16,   # Spart VRAM und ist optimal für A100-GPUs
    token="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"
)

print("✅ Model loaded successfully onto GPUs.")

def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    seed: int | None = None,
    model_name: str = MODEL_NAME,
) -> str:
    """
    Generates text using the locally loaded Transformers model.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Chat-Struktur für das Instruct-Modell aufbauen
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Chat-Template anwenden
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Parameter für deterministische (greedy) oder stochastische Generierung setzen
    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_TOKENS,
    }
    
    if temperature == 0.0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature

    # Text generieren
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # Nur die neu generierten Tokens ausschneiden und decodieren
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response_text.strip()

def check_ollama_ready() -> bool:
    """Da das Modell direkt in Python geladen ist, ist der Client immer bereit."""
    return model is not None
