import yaml
import mlflow
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import load_dataset, Dataset
import json

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_jsonl_as_dataset(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def preprocess_function(examples, tokenizer, max_length, doc_stride):
    """Tokenize questions and contexts, find answer positions"""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map start/end positions of answers in tokenized inputs
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find start/end of context tokens
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If answer not in context window, label as (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def train():
    params = load_params()
    model_name = params['model']['name']
    max_length = params['model']['max_length']
    doc_stride = params['model']['doc_stride']
    lr = params['training']['learning_rate']
    batch_size = params['training']['batch_size']
    num_epochs = params['training']['num_epochs']

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    print("Loading datasets...")
    train_dataset = load_jsonl_as_dataset("data/processed/train.jsonl")
    val_dataset = load_jsonl_as_dataset("data/processed/val.jsonl")

    print("Tokenizing...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="models/gujarati-qa",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=params['training']['weight_decay'],
        warmup_steps=params['training']['warmup_steps'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,           # halves VRAM usage
        logging_dir="logs/training",
        report_to="none",  # We handle MLflow manually
        seed=params['training']['seed']
    )

    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start MLflow run
    mlflow.set_experiment("gujarati-qa")
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # Log params
        mlflow.log_params({
            "model_name": model_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_length": max_length
        })

        print("Training...")
        trainer.train()

        # Log metrics
        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": eval_results['eval_loss'],
        })

        # Save model + tokenizer
        trainer.save_model("models/gujarati-qa-best")
        tokenizer.save_pretrained("models/gujarati-qa-best")

        # Log model to MLflow
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="gujarati-qa-model",
            registered_model_name="gujarati-qa"
        )

        print(f"Training done. Model saved. Eval loss: {eval_results['eval_loss']:.4f}")
        return run.info.run_id

if __name__ == "__main__":
    train()