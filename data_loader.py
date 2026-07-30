from datasets import load_dataset
from typing import List, Dict, Any
from config import config

def load_full_gsm8k() -> List[Dict[str, Any]]:
    """Loads 100% of the GSM8K dataset (train + test combined)."""
    print(f"Loading FULL GSM8K dataset...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    
    combined_data = []
    
    # Merge train and test with no partition
    for split in ['train', 'test']:
        for idx, item in enumerate(dataset[split]):
            combined_data.append({
                "id": f"{split}_{idx}",
                "question": item["question"],
                "ground_truth_raw": item["answer"],
                "ground_truth_num": extract_gsm8k_ground_truth(item["answer"])
            })
            
    print(f"Loaded {len(combined_data)} total questions.")
    return combined_data

def extract_gsm8k_ground_truth(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
    return ""