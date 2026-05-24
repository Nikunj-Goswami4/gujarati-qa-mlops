import json
from collections import Counter
import string

def normalize_answer(s):
    import re
    if not s:
        return ""
    # Only remove actual punctuation, keep all Unicode (Gujarati) chars
    s = re.sub(r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    return s.strip()

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# Fix 1 — fix empty string handling in f1_score in evaluate.py
def f1_score(prediction, ground_truth):
    # Both empty = perfect match
    if not prediction.strip() and not ground_truth.strip():
        return 1.0
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def evaluate_dataset(predictions: list, references: list):
    """
    predictions: list of predicted answer strings
    references: list of ground truth answer strings
    """
    em_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match_score(pred, ref))
        f1_scores.append(f1_score(pred, ref))

    return {
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
        "f1": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "num_samples": len(predictions)
    }

