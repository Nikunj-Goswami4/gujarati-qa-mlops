import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import evaluate_dataset
import json
import os
import yaml

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    params = load_params()
    model_path = "models/gujarati-qa-best"
    max_length = params['model']['max_length']
    max_ans_len = params['evaluation']['max_answer_length']
    
    # Run evaluation on GPU to leverage your hardware speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} onto {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
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

        # Tokenize matching your explicit pipeline requirements
        inputs = tokenizer(
            question, 
            context, 
            return_tensors="pt",
            max_length=max_length, 
            truncation="only_second", 
            padding="max_length",
            return_offsets_mapping=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device) if "token_type_ids" in inputs else None

        with torch.no_grad():
            if token_type_ids is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Bring logits back to CPU for tracking indices
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()
        
        # Identify token positions belonging ONLY to the context string
        sequence_ids = inputs.sequence_ids(0)
        
        # Heavily penalize non-context tokens (Question, Special Tokens, Padding)
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id != 1:  # 1 means context token
                start_logits[idx] = -10000.0
                end_logits[idx] = -10000.0
        
        seq_len = input_ids.shape[1]
        best_score = -float('inf')
        best_start, best_end = 0, 0
        
        # Locate the highest scoring valid span within context boundaries
        for s in range(seq_len):
            if sequence_ids[s] != 1:
                continue
            for e in range(s, min(s + max_ans_len, seq_len)):
                if sequence_ids[e] != 1:
                    continue
                score = start_logits[s] + end_logits[e]
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

        # Safe character mapping slice using context offsets
        if best_score == -float('inf') or best_start == 0:
            answer = ""
        else:
            offsets = inputs["offset_mapping"][0]
            start_char = offsets[best_start][0].item()
            end_char = offsets[best_end][1].item()
            answer = context[start_char:end_char].strip()
        
        predictions.append(answer)
        references.append(answers["text"][0] if answers["text"] else "")

    # Clean debug logs
    print("\n--- FIXED DEBUG SAMPLES ---")
    for i in range(min(10, len(predictions))):
        print(f"Pred : '{predictions[i]}'")
        print(f"Ref  : '{references[i]}'")
        print(f"EM   : {1 if predictions[i].strip() == references[i].strip() else 0}\n")
    print("--- END DEBUG ---\n")

    print("Calculating scores...")
    results = evaluate_dataset(predictions, references)
    
    print("\n========== RESULTS ==========")
    print(f"Exact Match : {results['exact_match']}%  (target: >40%)")
    print(f"F1 Score    : {results['f1']}%  (target: >60%)")
    print(f"Samples     : {results['num_samples']}")
    print("==============================")

    metrics_path = "reports/metrics.json"

    # Load existing metrics.json (has eval_loss from train.py)
    existing_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)

    # Merge F1 and Exact Match into it
    existing_metrics["exact_match"] = results['exact_match']
    existing_metrics["f1"] = results['f1']

    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()