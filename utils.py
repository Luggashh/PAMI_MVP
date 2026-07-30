import re
from typing import List, Tuple, Dict, Any
from collections import Counter

def extract_numerical_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a chain-of-thought response.
    Handles '#### 123', 'The answer is 123', and raw numbers.
    """
    if "####" in text:
        match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "").strip()
            
    # Look for explicitly stated answers
    matches = re.findall(r"(?:answer is|equals|=)\s*\\?\$?\s*(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(",", "").strip()
        
    # Fallback to the last standalone number in the text
    all_nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if all_nums:
        return all_nums[-1].replace(",", "").strip()
        
    return ""

def calculate_uncertainty_proxy(extracted_answers: List[str]) -> Tuple[str, float, float]:
    """
    Computes majority vote agreement over extracted numerical answers.
    Returns:
        - majority_answer (str)
        - agreement_ratio (float): Agreement score in [0.0, 1.0]
        - uncertainty_score (float): 1.0 - agreement_ratio
    """
    valid_answers = [ans for ans in extracted_answers if ans != ""]
    if not valid_answers:
        return "", 0.0, 1.0
        
    counts = Counter(valid_answers)
    majority_answer, top_count = counts.most_common(1)[0]
    total_samples = len(extracted_answers)
    
    agreement_ratio = top_count / total_samples
    uncertainty_score = 1.0 - agreement_ratio
    
    return majority_answer, agreement_ratio, uncertainty_score