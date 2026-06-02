import os
import json
import sys
import time
import random
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
# genai.configure(api_key=os.getenv("GEMINI_API_KEY_TEMP_7"))
# model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ── All API keys and models — auto-rotated on 429 ──
API_KEYS = [v for v in [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_TEMP_1"),
    os.getenv("GEMINI_API_KEY_TEMP_2"),
    os.getenv("GEMINI_API_KEY_TEMP_3"),
    os.getenv("GEMINI_API_KEY_TEMP_4"),
    os.getenv("GEMINI_API_KEY_TEMP_5"),
    os.getenv("GEMINI_API_KEY_TEMP_6"),
] if v]

MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

# Current state — rotated automatically
current_key_idx   = 0
current_model_idx = 0

def get_model():
    """Returns a fresh Gemini model using current key + model combo."""
    genai.configure(api_key=API_KEYS[current_key_idx])
    return genai.GenerativeModel(MODELS[current_model_idx])

print(f"Loaded {len(API_KEYS)} API keys")
print(f"Starting: {MODELS[current_model_idx]} | key[{current_key_idx}]\n")

def load_wikipedia_articles(path: str, max_articles: int = 500) -> list:
    """Load Gujarati Wikipedia articles"""
    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_articles:
                break
            record = json.loads(line)
            text = record.get('text', '').strip()
            # Only use articles with enough text
            if len(text) > 200:
                articles.append({
                    'title': record.get('title', ''),
                    'text': text[:1000]  # Limit context size
                })
    print(f"Loaded {len(articles)} Wikipedia articles")
    return articles

def load_sangraha_articles(path: str, max_articles: int = 500) -> list:
    """Load Sangraha Gujarati articles"""
    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_articles:
                break
            record = json.loads(line)
            text = record.get('text', '').strip()
            if len(text) > 200:
                articles.append({
                    'title': record.get('doc_id', f'sangraha_{i}'),
                    'text': text[:1000]
                })
    print(f"Loaded {len(articles)} Sangraha articles")
    return articles

def generate_qa_pairs(article: dict, num_pairs: int = 3) -> list:
    global current_key_idx, current_model_idx

    prompt = f"""You are a Gujarati QA dataset creator.

    Given this Gujarati text, generate exactly {num_pairs} question-answer pairs.

    STRICT RULES:
    1. Questions MUST be in Gujarati
    2. Answers MUST be exact substrings from the context (word-for-word copy)
    3. Answers must be short (1-8 words)
    4. Return ONLY valid JSON, no explanation, no markdown

    Context:
    {article['text']}

    Return this exact JSON format:
    [
    {{
        "question": "Gujarati question here?",
        "answer": "exact answer from context"
    }}
    ]"""

    MAX_RETRIES_PER_MODEL = 4  # after 4 fails → switch model

    while True:
        retries_this_model = 0

        while retries_this_model < MAX_RETRIES_PER_MODEL:
            try:
                model = get_model()
                response = model.generate_content(prompt)
                text = response.text.strip()

                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]

                return json.loads(text.strip())

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "quota" in error_msg.lower():
                    retries_this_model += 1
                    print(f"  ⚠️  429 on {MODELS[current_model_idx]} key[{current_key_idx}] "
                          f"(attempt {retries_this_model}/{MAX_RETRIES_PER_MODEL}) — waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"  ⚠️  Non-rate error: {e}")
                    return []

        # ── This model's quota done → try next model ──
        current_model_idx += 1

        if current_model_idx < len(MODELS):
            print(f"\n🔄 Model exhausted → switching to: {MODELS[current_model_idx]} (key[{current_key_idx}])")
            continue

        # ── All 3 models done → switch API key ──
        current_model_idx = 0
        current_key_idx += 1

        if current_key_idx < len(API_KEYS):
            print(f"\nAll models exhausted → switching to key[{current_key_idx}] | "
                  f"model: {MODELS[current_model_idx]}")
            continue

        # ── All keys + all models exhausted → stop ──
        print(f"\n⛔ All {len(API_KEYS)} keys × {len(MODELS)} models exhausted.")
        print("Stopping. Run again tomorrow when quota resets.")
        sys.exit(0)


def find_answer_start(context: str, answer: str) -> int:
    """Find exact character position of answer in context"""
    pos = context.find(answer)
    return pos


def build_squad_record(article: dict, qa: dict, idx: int) -> dict | None:
    """Convert to SQuAD format matching your existing dataset"""
    context = article['text']
    question = qa.get('question', '').strip()
    answer = qa.get('answer', '').strip()
    
    if not question or not answer:
        return None
    
    # Find answer position in context
    answer_start = find_answer_start(context, answer)
    
    if answer_start == -1:
        # Answer not found exactly — skip this pair
        return None
    
    return {
        # "id": f"synthetic_{idx:05d}", # if using wikipedia
        "id": f"synthetic_sangraha_{idx:05d}", # if using sangraha
        "context": context,
        "question": question,
        "answers": {
            "text": [answer],
            "answer_start": [answer_start]
        }
    }


def generate_dataset(
    output_path: str = "data/raw/synthetic_gu_qa.jsonl",
    max_articles: int = 1000,
    pairs_per_article: int = 3,
    sleep_between_requests: float = 4.5  # Stay under 15 RPM limit
):
    """
    Main generation function.
    
    Default settings (safe for free tier):
    - 1000 articles × 3 pairs = ~3000 QA pairs
    - 4.5 sec sleep = ~13 requests/min (under 15 RPM limit)
    - Total time: ~60 minutes
    - Daily quota used: 1000/3000 requests
    """
    
    # articles = load_wikipedia_articles("data/raw/gu_wikipedia.jsonl", max_articles)   # Wikipedia
    articles = load_sangraha_articles("data/raw/sangraha_gu.jsonl", max_articles)  # Sangraha
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # SMART RESUME: Check what is already processed
    existing_contexts = set()
    total_generated = 0
    
    if Path(output_path).exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing_contexts.add(record.get('context', ''))
                    total_generated += 1
        print(f"Found existing file. Loaded {total_generated} QA pairs. Skipping completed articles...")
    
    # Calculate starting index for IDs based on existing records
    idx = total_generated
    total_skipped = 0
    
    print(f"\nGenerating QA pairs from {len(articles)} articles...")
    print(f"Target: ~{len(articles) * pairs_per_article} pairs")
    print(f"Estimated time: ~{int(len(articles) * sleep_between_requests / 60)} minutes\n")
    
    # Change 'w' to 'a' to APPEND instead of overwriting
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for i, article in enumerate(articles):

            # Skip if this article text was already processed in a previous run
            if article['text'] in existing_contexts:
                continue

            print(f"[{i+1}/{len(articles)}] {article['title'][:40]}...")
            
            pairs = generate_qa_pairs(article, pairs_per_article)
            
            for qa in pairs:
                record = build_squad_record(article, qa, idx)
                if record:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    out_f.flush()  # Write immediately in case of crash
                    total_generated += 1
                    idx += 1
                else:
                    total_skipped += 1
            
            print(f"  Generated: {len(pairs)} pairs | Total so far: {total_generated}")
            
            # Rate limit — stay under 15 RPM
            time.sleep(sleep_between_requests)
    
    print(f"\nDone!")
    print(f"Total QA pairs generated : {total_generated}")
    print(f"Skipped (bad format)     : {total_skipped}")
    print(f"Saved to                 : {output_path}")


if __name__ == "__main__":
    generate_dataset()