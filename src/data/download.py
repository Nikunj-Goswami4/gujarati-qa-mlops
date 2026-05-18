from datasets import load_dataset
import pandas as pd
import os

def download_indicqa():
    """Download IndicQA dataset — has Gujarati QA pairs"""
    print("Downloading IndicQA (Gujarati)...")
    dataset = load_dataset("ai4bharat/IndicQA", "gu")
    return dataset

def download_wikipedia_gujarati():
    """Download latest Gujarati Wikipedia (updated monthly) for knowledge base"""
    print("Downloading latest Gujarati Wikipedia...")
    wiki = load_dataset("omarkamali/wikipedia-monthly", "latest.gu", 
                        split="train",
                        streaming=True  # stream it so you don't need to load all into RAM at once
                    )   
    return wiki

def save_raw_data():
    os.makedirs("data/raw", exist_ok=True)

    # Save IndicQA
    qa_data = download_indicqa()
    qa_data['train'].to_json("data/raw/indicqa_gu_train.jsonl")
    qa_data['validation'].to_json("data/raw/indicqa_gu_val.jsonl")

    # Save Wikipedia
    wiki = download_wikipedia_gujarati()
    wiki['train'].to_json("data/raw/gu_wikipedia.jsonl")

    print("Done. Raw data saved to data/raw/")

if __name__ == "__main__":
    save_raw_data()