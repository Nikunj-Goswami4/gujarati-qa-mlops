# Script to run inside Kaggle on a T4 GPU.
# DON'T run this on personal laptop.

import subprocess
import os
import sys

print("=" * 50)
print("Gujarati QA - Kaggle T4 GPU Training")
print("=" * 50)

# ── Step 1: Install dependencies ──────────────────
print("\n[1/6] Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "mlflow",
    "indicnlp", "huggingface_hub", "torch",
    "scikit-learn", "pandas", "pyyaml"
], check=True)
print("Dependencies installed.")

# ── Step 2: Clone your GitHub repo ────────────────
print("\n[2/6] Cloning project repo...")

# # For private repos
# from kaggle_secrets import UserSecretsClient
# secrets = UserSecretsClient()

# try:
#     github_token = secrets.get_secret("GITHUB_TOKEN")
#     repo_url = f"https://{github_token}@github.com/YOUR_USERNAME/gujarati-qa-mlops.git"
#     print("Using GitHub token for private repo.")
# except Exception:
#     # If no token found, assume repo is public
#     repo_url = "https://github.com/YOUR_USERNAME/gujarati-qa-mlops.git"
#     print("No GitHub token found. Assuming public repo.")

repo_url = "https://github.com/Nikunj-Goswami4/gujarati-qa-mlops.git"

subprocess.run(["git", "clone", repo_url, "project"], check=True)
os.chdir("project")
print("Repo cloned.")

# ── Step 3: Set up data ───────────────────────────
print("\n[3/6] Copying data from Kaggle dataset...")
os.makedirs("data/processed", exist_ok=True)
subprocess.run([
    "cp", "-r",
    "/kaggle/input/gujarati-qa-data/.",
    "data/processed/"
], check=True)
print("Data ready at data/processed/")

# Dynamic Drift Adaptation Injector:
# Checks if the automated 2 AM GitHub Workflow packaged any fresh real-time drift logs
drift_logs_path = "../drifted_logs.jsonl"
if os.path.exists(drift_logs_path):
    print("⚠️ Fresh production data drift logs detected! Merging with baseline training datasets...")
    try:
        # Append new real-world drift examples straight into train.jsonl
        with open(drift_logs_path, "r", encoding="utf-8") as f_drift:
            drift_lines = f_drift.readlines()
        
        with open("data/processed/train.jsonl", "a", encoding="utf-8") as f_train:
            f_train.writelines(drift_lines)
            
        print(f"✅ Successfully blended {len(drift_lines)} drifted production samples into training stream.")
    except Exception as e:
        print(f"CRITICAL - Failed data blending combination: {e}. Falling back to default data.")
else:
    print("📋 No execution-level drift logs found. Training with baseline snapshot assets.")

# MLflow Tracker Injection mapping:
# Bypasses the empty dashboard bug by linking Kaggle directly to DagsHub
try:
    os.environ["MLFLOW_TRACKING_URI"] = secrets.get_secret("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret("MLFLOW_TRACKING_PASSWORD")
    print("MLflow tracking variables mapped securely to DagsHub.")
except Exception as e:
    print(f"Warning - MLflow server environment parameters omitted: {e}")

# Load HuggingFace token from Kaggle secrets
hf_token = secrets.get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token
print("HuggingFace token loaded.")

# ── Step 4: Run training ──────────────────────────
print("\n[4/6] Starting training...")
result = subprocess.run(
    [sys.executable, "src/model/train.py"],
    check=True
)
print("Training complete!")

# # ── Step 5: Push model to HuggingFace Hub ─────────
# print("\nPushing model to HuggingFace Hub...")
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path="models/gujarati-qa-best",
#     repo_id="YOUR_HF_USERNAME/gujarati-qa-model",
#     token=hf_token
# )
# print("Model pushed to HuggingFace Hub successfully!")
# print("=" * 50)
# print("All done! Training finished on Kaggle T4 GPU.")
# print("=" * 50)


# ── Step 5: Run evaluation and save metrics ─────────
print("\n[5/6] Running evaluation...")
subprocess.run(
    [sys.executable, "src/model/run_evaluate.py"],
    check=True
)
print("Evaluation complete. Metrics saved to reports/metrics.json")

# ── Step 6: Push model + metrics to STAGING repo (not production) ─────────
# GitHub Actions will ask you to approve before it goes to production
print("\n[6/6] Pushing model to HuggingFace STAGING repo...")
from huggingface_hub import HfApi
api = HfApi()

# Push the trained model files
api.upload_folder(
    folder_path="models/gujarati-qa-best",
    repo_id="Nikunj4/gujarati-qa-staging",
    token=hf_token
)

# Push metrics.json separately so GitHub Actions can read it
api.upload_file(
    path_or_fileobj="reports/metrics.json",
    path_in_repo="metrics.json",
    repo_id="Nikunj4/gujarati-qa-staging",
    token=hf_token
)

print("Model and metrics pushed to STAGING.")
print("GitHub Actions will now show you metrics and ask for approval.")
print("=" * 50)
print("All done! Training finished on Kaggle T4 GPU.")
print("=" * 50)