import json
import re
from pathlib import Path

def clean_gujarati_text(text: str) -> str:
    """Basic cleaning for Gujarati text"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    return text

def load_jsonl(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def preprocess_qa_dataset(input_path: str, output_path: str):
    """Clean and validate QA pairs"""
    records = load_jsonl(input_path)
    cleaned = []

    for record in records:
        context = clean_gujarati_text(record.get('context', ''))
        question = clean_gujarati_text(record.get('question', ''))
        answers = record.get('answers', {})

        # Skip records with empty context or question
        if not context or not question:
            continue

        # Skip records with no answers
        if not answers.get('text'):
            continue

        cleaned.append({
            "id": record.get("id"),
            "context": context,
            "question": question,
            "answers": answers
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in cleaned:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Processed {len(cleaned)} records → {output_path}")

def preprocess_wikipedia(input_path: str, output_path: str, max_articles=5000):
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

if __name__ == "__main__":
    preprocess_qa_dataset("data/raw/indicqa_gu_train.jsonl", "data/processed/train.jsonl")
    preprocess_qa_dataset("data/raw/indicqa_gu_val.jsonl", "data/processed/val.jsonl")
    preprocess_wikipedia("data/raw/gu_wikipedia.jsonl", "data/knowledge_base/kb.jsonl")