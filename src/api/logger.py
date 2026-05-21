import json
import os
from datetime import datetime, UTC

LOG_FILE = "logs/predictions.jsonl"

def log_prediction(question: str, context: str, answer: str, confidence: float):
    """Append prediction to JSONL log file for monitoring"""
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "question": question,
        "context_length": len(context),
        "answer": answer,
        "confidence": confidence,
        "question_length": len(question),
        "answer_length": len(answer)
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')