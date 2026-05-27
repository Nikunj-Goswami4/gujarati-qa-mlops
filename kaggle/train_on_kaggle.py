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
print("\n[3/6] Setting up data from Kaggle dataset...", flush=True)
os.makedirs("data/processed", exist_ok=True)

import shutil

# Define potential root mounting paths
candidate_roots = [
    "/kaggle/input/gujarati-qa-data",
    "/kaggle/input/datasets/nikunjgoswami/gujarati-qa-data"
]

target_dir = None

# Look for the actual directory containing train.jsonl dynamically
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.jsonl" in files:
            target_dir = root
            print(f"📦 Successfully located dataset files inside: {target_dir}", flush=True)
            break

# Fallback checking if the recursive walk fails
if not target_dir:
    for path in candidate_roots:
        # Check if the nested 'processed' folder exists explicitly
        nested_processed = os.path.join(path, "processed")
        if os.path.exists(nested_processed):
            target_dir = nested_processed
            break
        elif os.path.exists(path):
            target_dir = path
            break

# Copy the flat file contents directly into data/processed/
if target_dir:
    copied_count = 0
    for filename in os.listdir(target_dir):
        src_path = os.path.join(target_dir, filename)
        dest_path = os.path.join("data/processed", filename)
        
        # Only copy files to avoid creating nested folders
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
            print(f"  -> Copied: {filename}", flush=True)
            copied_count += 1
            
    print(f"✅ Data copy complete! Loaded {copied_count} files directly into data/processed/", flush=True)
else:
    print("❌ CRITICAL ERROR: Could not locate train.jsonl anywhere under /kaggle/input!", flush=True)
    sys.exit(1)

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
# Secrets injected by GitHub Actions before this script is pushed to Kaggle
os.environ["MLFLOW_TRACKING_URI"]      = "MLFLOW_URI_PLACEHOLDER"
os.environ["MLFLOW_TRACKING_USERNAME"] = "MLFLOW_USERNAME_PLACEHOLDER"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "MLFLOW_PASSWORD_PLACEHOLDER"

hf_token = "HF_TOKEN_PLACEHOLDER"
os.environ["HF_TOKEN"] = hf_token
print("All credentials loaded.")

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

# Wipe all secrets from memory when done
hf_token = "CLEARED"
os.environ["HF_TOKEN"]                = "CLEARED"
os.environ["MLFLOW_TRACKING_URI"]     = "CLEARED"
os.environ["MLFLOW_TRACKING_USERNAME"]= "CLEARED"
os.environ["MLFLOW_TRACKING_PASSWORD"]= "CLEARED"
print("All secrets cleared from memory.")