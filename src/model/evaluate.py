import json
from collections import Counter
import string

def normalize_answer(s):
    """Lower text, remove punctuation, articles, extra whitespace"""
    def remove_articles(text):
        return text.replace('a ', '').replace('an ', '').replace('the ', '')

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    pred = normalize_answer(prediction)
    gt = normalize_answer(ground_truth)
    
    if not pred or not gt:
        return 0
    
    # Character level matching for Gujarati
    from collections import Counter
    pred_chars = Counter(pred)
    gt_chars = Counter(gt)
    
    common = sum((pred_chars & gt_chars).values())
    
    if common == 0:
        return 0
    
    precision = common / len(pred)
    recall = common / len(gt)
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