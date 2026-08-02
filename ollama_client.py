import requests
import time

def generate(prompt, system_prompt=None, seed=42, timeout=240):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": 0.7
        }
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
