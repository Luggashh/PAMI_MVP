import requests
import time

def generate(prompt, system_prompt=None, seed=42, timeout=240, **kwargs):
    """
    Universelle Generierungsfunktion für Ollama.
    **kwargs fängt zusätzliche Argumente wie temperature, max_tokens etc. flexibel ab.
    """
    url = "http://localhost:11434/api/generate"
    
    # Basis-Optionen für Ollama vorbereiten
    options = {
        "seed": seed,
        "temperature": kwargs.get("temperature", 0.7)  # Nutzt übergebene Temp, sonst Default 0.7
    }
    
    # Falls weitere Optionen übergeben wurden (z.B. max_tokens oder num_predict), fügen wir sie hinzu
    if "max_tokens" in kwargs:
        options["num_predict"] = kwargs["max_tokens"]
    if "temperature_greedy" in kwargs and kwargs.get("temperature_greedy") == 0.0:
        # Falls die Audit-Schleife für deterministische Schritte Temp=0 fordert
        options["temperature"] = 0.0

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": options
    }
    
    if system_prompt:
        payload["system"] = system_prompt

    # Bis zu 3 Wiederholungsversuche bei einem Timeout einplanen
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "")
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt < 2:
                wartezeit = 5 * (attempt + 1)
                print(f"\n[⚠️ Timeout/Fehler] Versuch {attempt+1} fehlgeschlagen: {e}. Warte {wartezeit}s...")
                time.sleep(wartezeit)
            else:
                raise TimeoutError(f"Ollama-Anfrage permanent fehlgeschlagen nach 3 Versuchen. Letzter Fehler: {e}")
