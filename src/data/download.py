from datasets import load_dataset
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def download_indicqa():
    """Download IndicQA dataset — has Gujarati QA pairs"""
    print("Downloading IndicQA (Gujarati)...")
    dataset = load_dataset("ai4bharat/IndicQA", "indicqa.gu")
    return dataset

def download_wikipedia_gujarati():
    """Download latest Gujarati Wikipedia (updated monthly) for knowledge base"""
    print("Downloading latest Gujarati Wikipedia...")
    wiki = load_dataset("omarkamali/wikipedia-monthly", "latest.gu", 
                        split="train",
                        #streaming=True  # stream it so you don't need to load all into RAM at once
                    )   
    return wiki

def save_raw_data():
    os.makedirs("data/raw", exist_ok=True)

    # Save IndicQA
    qa_data = download_indicqa()
    # qa_data['train'].to_json("data/raw/indicqa_gu_train.jsonl")
    # qa_data['validation'].to_json("data/raw/indicqa_gu_val.jsonl")
    
    # Only 'test' split exists — split it 80/20 manually
    full_data = qa_data['test'].train_test_split(test_size=0.2, seed=42)
    
    full_data['train'].to_json("data/raw/indicqa_gu_train.jsonl")
    full_data['test'].to_json("data/raw/indicqa_gu_val.jsonl")
    
    print(f"Train samples: {len(full_data['train'])}")
    print(f"Val samples  : {len(full_data['test'])}")

    # Save Wikipedia
    wiki = download_wikipedia_gujarati()
    # wiki['train'].to_json("data/raw/gu_wikipedia.jsonl")
    wiki.to_json("data/raw/gu_wikipedia.jsonl")

    print("Done. Raw data saved to data/raw/")

if __name__ == "__main__":
    save_raw_data()