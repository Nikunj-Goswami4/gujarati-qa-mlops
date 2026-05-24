import os
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY_TEMP_7"))
model = genai.GenerativeModel("gemini-3.5-flash")

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


def generate_qa_pairs(article: dict, num_pairs: int = 3) -> list:
    """Ask Gemini to generate QA pairs from a Gujarati article with auto-retry on 429 rate limits"""
    
    prompt = f"""You are a Gujarati QA dataset creator.

    Given this Gujarati text, generate exactly {num_pairs} question-answer pairs.

    STRICT RULES:
    1. Questions MUST be in Gujarati
    2. Answers MUST be exact substrings from the context (word-for-word copy)
    3. Answers must be short (1-5 words)
    4. Return ONLY valid JSON, no explanation, no markdown

    Context:
    {article['text']}

    Return this exact JSON format:
    [
    {{
        "question": "Gujarati question here?",
        "answer": "exact answer from context"
    }},
    {{
        "question": "another Gujarati question?",
        "answer": "exact answer from context"
    }}
    ]"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean response — remove markdown if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            pairs = json.loads(text)
            return pairs
            
        except Exception as e:
                error_msg = str(e)
                # Check if the error is due to rate limiting/quota issues
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\n  ⚠️ Rate limit hit (429) on article: '{article['title'][:30]}'")
                    print(f"  Waiting 60 seconds before retry (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(60)
                else:
                    # If it's a structural or parsing error, log it and break to skip this specific article
                    print(f"  Error generating QA: {e}")
                    return []
                
    print(f"Skipped article '{article['title'][:30]}' permanently after failing {max_retries} retries.")
    return []


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
        "id": f"synthetic_{idx:05d}",
        "context": context,
        "question": question,
        "answers": {
            "text": [answer],
            "answer_start": [answer_start]
        }
    }


def generate_dataset(
    wiki_path: str = "data/raw/gu_wikipedia.jsonl",
    output_path: str = "data/raw/synthetic_gu_qa.jsonl",
    max_articles: int = 500,
    pairs_per_article: int = 3,
    sleep_between_requests: float = 4.5  # Stay under 15 RPM limit
):
    """
    Main generation function.
    
    Default settings (safe for free tier):
    - 500 articles × 3 pairs = ~1500 QA pairs
    - 4.5 sec sleep = ~13 requests/min (under 15 RPM limit)
    - Total time: ~30 minutes
    - Daily quota used: 500/1500 requests
    """
    
    articles = load_wikipedia_articles(wiki_path, max_articles)
    
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