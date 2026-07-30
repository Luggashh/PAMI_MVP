from datasets import load_dataset
from typing import List, Dict, Any
from config import config

def load_full_gsm8k() -> List[Dict[str, Any]]:
    """
    Loads 100% of the GSM8K dataset by concatenating both 'train' and 'test' splits,
    removing any partition between train/test data.
    """
    print(f"[DataLoader] Loading GSM8K dataset ({config.DATASET_NAME})...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    
    combined_data = []
    
    if config.USE_FULL_DATASET:
        splits = ['train', 'test']
    else:
        splits = ['test']
        
    for split in splits:
        for idx, item in enumerate(dataset[split]):
            combined_data.append({
                "id": f"{split}_{idx}",
                "question": item["question"],
                "ground_truth_raw": item["answer"],
                "ground_truth_num": extract_gsm8k_ground_truth(item["answer"])
            })
            
    print(f"[DataLoader] Successfully loaded total of {len(combined_data)} questions (100% unpartitioned).")
    return combined_data

def extract_gsm8k_ground_truth(answer_str: str) -> str:
    """Extracts the final numerical value following #### in GSM8K answers."""
    if "####" in answer_str:
        num = answer_str.split("####")[-1].strip().replace(",", "")
        return num
    return ""