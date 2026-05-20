import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import evaluate_dataset
import json

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    model_path = "models/gujarati-qa-best"
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    print("Loading validation data...")
    val_data = load_jsonl("data/processed/val.jsonl")

    predictions = []
    references = []

    print(f"Running inference on {len(val_data)} samples...")
    for i, record in enumerate(val_data):
        question = record["question"]
        context = record["context"]
        answers = record["answers"]

        inputs = tokenizer(question, context, return_tensors="pt",
                          max_length=384, truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.decode(inputs["input_ids"][0][start:end],
                                  skip_special_tokens=True)

        predictions.append(answer)
        references.append(answers["text"][0] if answers["text"] else "")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_data)}")

    print("\nCalculating scores...")
    results = evaluate_dataset(predictions, references)
    
    print("\n========== RESULTS ==========")
    print(f"Exact Match : {results['exact_match']}%  (target: >40%)")
    print(f"F1 Score    : {results['f1']}%  (target: >60%)")
    print(f"Samples     : {results['num_samples']}")
    print("==============================")

if __name__ == "__main__":
    main()