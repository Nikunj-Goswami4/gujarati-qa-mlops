from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class GujaratiQAModel:
    def __init__(self, model_path: str = "models/gujarati-qa-best"):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()
        print("Model loaded.")

    def answer(self, question: str, context: str) -> dict:
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[answer_start:answer_end]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Confidence score (softmax of logits)
        start_score = torch.softmax(outputs.start_logits, dim=1)[0][answer_start].item()
        end_score = torch.softmax(outputs.end_logits, dim=1)[0][answer_end - 1].item()
        confidence = round((start_score + end_score) / 2, 4)

        return {
            "answer": answer,
            "confidence": confidence,
            "answer_start": answer_start.item(),
            "answer_end": (answer_end - 1).item()
        }