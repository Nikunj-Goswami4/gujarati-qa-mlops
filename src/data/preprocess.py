import json
import re
from pathlib import Path
import random
import os

def clean_gujarati_text(text: str) -> str:
    """Basic cleaning for Gujarati text"""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_jsonl(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def preprocess_qa_dataset(input_path: str, output_path: str):
    """Clean and validate QA pairs with dynamic index correction"""
    records = load_jsonl(input_path)
    cleaned = []

    for record in records:
        orig_context = record.get('context', '')
        orig_question = record.get('question', '')
        answers = record.get('answers', {})

        # Skip records with missing core blocks or empty answer arrays
        if not orig_context or not orig_question or not answers.get('text'):
            continue
            
        ans_text = answers['text'][0]
        # Skip explicitly empty answer texts
        if not ans_text or not ans_text.strip():
            continue

        # 1. Apply cleaning to text fields
        context = clean_gujarati_text(orig_context)
        question = clean_gujarati_text(orig_question)
        cleaned_ans_text = clean_gujarati_text(ans_text)

        # 2. Re-locate the exact substring index inside the newly cleaned text
        new_start = context.find(cleaned_ans_text)
        
        # If the text cannot be found due to severe cleaning alterations, fall back to the original text lookup
        if new_start == -1:
            new_start = context.find(ans_text)
            
        # If the span alignment fails entirely, drop this sample to preserve training quality
        if new_start == -1:
            continue

        cleaned.append({
            "id": record.get("id"),
            "context": context,
            "question": question,
            "answers": {
                "text": [cleaned_ans_text],
                "answer_start": [new_start]
            }
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in cleaned:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Processed {len(cleaned)} records → {output_path}")

def preprocess_wikipedia_for_kb(input_path: str, output_path: str, max_articles=5000):
    """Extract clean Gujarati paragraphs for knowledge base"""
    records = load_jsonl(input_path)
    docs = []

    for i, record in enumerate(records[:max_articles]):
        text = clean_gujarati_text(record.get('text', ''))
        if len(text) > 100:  # Skip very short articles
            docs.append({
                "id": record.get("id"),
                "title": record.get("title", ""),
                "text": text[:2000]  # Limit to 2000 chars per article
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"Processed {len(docs)} Wikipedia articles → {output_path}")

def preprocess_sangraha_for_kb(input_path: str, output_path: str, max_articles: int = 5000):
    """Add Sangraha articles to knowledge base"""
    records = load_jsonl(input_path)
    docs = []
    for i, record in enumerate(records[:max_articles]):
        text = clean_gujarati_text(record.get('text', ''))
        if len(text) > 100:
            docs.append({
                "id": record.get("doc_id", f"sangraha_{i}"),
                "title": record.get("doc_id", ""),
                "text": text[:2000]
            })

    # Append to existing kb.jsonl (don't overwrite Wikipedia)
    with open(output_path, 'a', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"Added {len(docs)} Sangraha articles → {output_path}")


if __name__ == "__main__":
    preprocess_qa_dataset("data/raw/indicqa_gu_train.jsonl", "data/processed/train.jsonl")
    preprocess_qa_dataset("data/raw/indicqa_gu_val.jsonl", "data/processed/val.jsonl")
    preprocess_wikipedia_for_kb("data/raw/gu_wikipedia.jsonl", "data/knowledge_base/kb.jsonl", max_articles=5000)
    preprocess_sangraha_for_kb("data/raw/sangraha_gu.jsonl", "data/knowledge_base/kb.jsonl", max_articles=5000)

    # Preprocess each optional source if it exists
    optional_sources = {
        "data/raw/indic_squad_gu.jsonl":  "data/processed/indic_squad_gu.jsonl",
        "data/raw/synthetic_gu_qa.jsonl": "data/processed/synthetic_train.jsonl",
    }
    for raw_path, processed_path in optional_sources.items():
        if os.path.exists(raw_path):
            print(f"\nProcessing {raw_path}...")
            preprocess_qa_dataset(raw_path, processed_path)

    # Combine ALL processed sources into final train.jsonl
    all_train_sources = [
        "data/processed/train.jsonl",           # IndicQA
        "data/processed/indic_squad_gu.jsonl", # IndicSQuAD
        "data/processed/synthetic_train.jsonl",   # Synthetic
    ]
    combined = []
    for path in all_train_sources:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f]
            combined.extend(records)
            print(f"  Loaded {len(records)} from {path}")

    random.seed(42)
    random.shuffle(combined)

    with open("data/processed/train.jsonl", 'w', encoding='utf-8') as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n\n\n\nFinal train.jsonl: {len(combined)} samples")
    print(f"Val.jsonl stays as is: IndicQA native Gujarati only")