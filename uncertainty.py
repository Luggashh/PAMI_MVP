import os
from ollama_client import generate
from utils import extract_numerical_answer
from config import SEED_MAIN, TEMPERATURE

def compute_uncertainty(prompt: str, system_prompt: str, k: int) -> dict:
    """
    Generiert k stochastische Antworten mit höherer Temperatur,
    um die Übereinstimmung (Agreement) und Unsicherheit zu berechnen.
    """
    samples = []
    answers = []
    
    for i in range(k):
        # Nutzt die flexible generate-Funktion mit stochastischer Temperatur
        response = generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=TEMPERATURE,  # Nutzt z.B. 0.7 aus der config.py
            seed=SEED_MAIN + i        # Variiert den Seed für echte Stichproben
        )
        samples.append(response)
        answers.append(extract_numerical_answer(response))
        
    # Berechne Mehrheits-Antwort (Majority Vote)
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        majority_answer = None
        agreement_count = 0
        agreement = 0.0
    else:
        from collections import Counter
        counter = Counter(valid_answers)
        majority_answer, agreement_count = counter.most_common(1)[0]
        agreement = agreement_count / k

    return {
        "samples": samples,
        "answers": answers,
        "majority_answer": majority_answer,
        "agreement_count": agreement_count,
        "agreement": agreement
    }
