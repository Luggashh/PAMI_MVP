"""
Thin wrapper around the vLLM OpenAI-compatible REST API.
Handles Round-Robin load balancing across multiple vLLM instances.
"""

import requests
import json
import itertools
import threading
from config import VLLM_BASE_URLS, MODEL_NAME, MAX_TOKENS

# Thread-sicherer Round-Robin-Generator für die URLs
url_pool = itertools.cycle(VLLM_BASE_URLS)
url_lock = threading.Lock()

def get_next_base_url() -> str:
    """Get the next available vLLM base URL in a thread-safe manner."""
    with url_lock:
        return next(url_pool)

def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    seed: int | None = None,
    model: str = MODEL_NAME,
) -> str:
    """
    Send a generation request to one of the local vLLM instances using Round-Robin.
    """
    base_url = get_next_base_url()
    # vLLM nutzt den standardisierten OpenAI Chat-Endpunkt
    url = f"{base_url}/v1/chat/completions"

    # OpenAI-kompatibles JSON-Payload für vLLM
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }

    # vLLM/OpenAI nutzt 'seed' direkt auf oberster Ebene
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        # Antwort-Parsing für OpenAI-Format
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to vLLM instance at {base_url}. Is the service running?"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"vLLM request to {base_url} timed out after 120s.")
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected response format from vLLM: {response.text}")

def check_ollama_ready() -> bool:
    """Verify vLLM instances are running and responding."""
    all_ready = True
    for base_url in VLLM_BASE_URLS:
        try:
            # vLLM listet verfügbare Modelle unter /v1/models
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            
            if MODEL_NAME not in models:
                print(f"⚠  Model '{MODEL_NAME}' not loaded on {base_url}. Available: {models}")
                all_ready = False
        except Exception as e:
            print(f"⚠  vLLM instance not reachable at {base_url}: {e}")
            all_ready = False
            
    return all_ready
