"""
Thin wrapper around the local Ollama REST API.
Handles both deterministic (greedy) and stochastic (sampling) generation.
"""

import requests
import json
from config import OLLAMA_BASE_URL, MODEL_NAME, MAX_TOKENS

def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    seed: int | None = None,
    model: str = MODEL_NAME,
) -> str:
    """
    Send a generation request to the local user-space Ollama server.
    """
    # Nutzt den standardmäßigen Ollama-Endpunkt für Textgenerierung
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
    }

    if seed is not None:
        payload["options"]["seed"] = seed

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Verbindung zu Ollama unter {OLLAMA_BASE_URL} fehlgeschlagen. Läuft der Server?"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama-Anfrage lief in ein Timeout nach 120s.")

def check_ollama_ready() -> bool:
    """Überprüft, ob der lokale Ollama-Server läuft und das Modell geladen ist."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        
        # Prüft, ob das in config.py eingetragene Modell (llama3.2:3b) vorhanden ist
        return any(MODEL_NAME in m for m in models)
    except Exception:
        return False
