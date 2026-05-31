from datasets import load_dataset
from flask import json
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

def download_sangraha_gujarati(max_articles: int = 50000):
    """Download Sangraha verified Gujarati corpus — richer than Wikipedia"""
    print("Downloading Sangraha Gujarati (verified)...")
    dataset = load_dataset(
        "ai4bharat/sangraha",
        data_dir="verified/guj",
        split="train",
        streaming=True   # 705GB total - stream so RAM doesn't explode
    )
    articles = []
    for i, record in enumerate(dataset):
        if i >= max_articles:
            break
        articles.append(record)
        if i % 1000 == 0:
            print(f"  Downloaded {i} articles so far...")
    print(f"Total Gujarati articles: {len(articles)}")
    return articles

def download_indic_squad_gujarati():
    """Download l3cube IndicSQuAD — Gujarati rows only"""
    print("Downloading IndicSQuAD (filtering Gujarati)...")
    from datasets import load_dataset
    dataset = load_dataset("l3cube-pune/indic-squad", split="train")

    # Filter Gujarati rows — detect by Gujarati unicode range
    def is_gujarati(text):
        return any('\u0A80' <= c <= '\u0AFF' for c in text)

    gujarati_rows = [r for r in dataset if is_gujarati(r.get('context', ''))]
    print(f"Gujarati rows found: {len(gujarati_rows)}")
    return gujarati_rows

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
    print(f"Wikipedia samples saved: {len(wiki)}")

    # Save Sangraha
    sangraha = download_sangraha_gujarati()
    with open("data/raw/sangraha_gu.jsonl", "w", encoding="utf-8") as f:
        for record in sangraha:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Sangraha samples saved: {len(sangraha)}")

    # Save IndicSQuAD Gujarati
    indic_squad = download_indic_squad_gujarati()
    with open("data/raw/indic_squad_gu.jsonl", "w", encoding="utf-8") as f:
        for record in indic_squad:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"IndicSQuAD Gujarati saved: {len(indic_squad)}")

    print(f"\n\n\n\nDone. Raw data saved to data/raw/")

if __name__ == "__main__":
    save_raw_data()