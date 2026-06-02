<div align="center">

# 🇮🇳 Gujarati QA — Production MLOps System

### Production-Grade Extractive Question Answering for the Gujarati Language
### with Full End-to-End Automated MLOps Infrastructure

[![Model on HF](https://img.shields.io/badge/🤗_HuggingFace-Production_Model-yellow)](https://huggingface.co/Nikunj4/gujarati-qa-model)
[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-orange)](https://huggingface.co/spaces/Nikunj4/gujarati-qa-demo)
[![MLflow on DagsHub](https://img.shields.io/badge/MLflow-DagsHub_Tracking-blue)](https://dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops.mlflow)
[![DVC](https://img.shields.io/badge/Data_Versioning-DVC-green)](https://dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20beff)](https://www.kaggle.com/datasets/nikunjgoswami/gujarati-qa-data)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**[🚀 Try the Live Demo](https://huggingface.co/spaces/Nikunj4/gujarati-qa-demo) · [📦 Production Model](https://huggingface.co/Nikunj4/gujarati-qa-model) · [📊 MLflow Experiments](https://dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops.mlflow) · [🗄️ Kaggle Dataset](https://www.kaggle.com/datasets/nikunjgoswami/gujarati-qa-data)**

</div>

<!-- `psswd: admin123` -->

<br/>

<div align="center">
  <video src="assets/Demo.mp4" width="90%" autoplay loop muted controls style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);"></video>
  <p align="center">
    <sub><i>Quick walk-through of the live Gujarati QA Interface and MLOps Telemetry Dashboard.</i></sub>
  </p>
</div>

<br/>

---

## 📌 What Is This Project?

This is an **enterprise-grade, fully automated MLOps pipeline** for an **extractive Question Answering system** tailored for the **Gujarati language**, one of India's 22 official scheduled languages, spoken by over 55 million people.

The system moves entirely away from typical local-notebook workflows. It establishes a **headless cloud infrastructure** using distributed GPU training, dataset versioning, enterprise CI/CD gates with human-in-the-loop approval, automated data drift detection, and a persistent telemetry feedback loop, all running without requiring your laptop to be on during training or deployment.

**What the system does end-to-end:**
- Fine-tunes `google/muril-base-cased` (MuRIL) for Gujarati extractive QA on 81,807 training samples
- Versions datasets with **DVC** and stores them on **Kaggle**
- Trains automatically on a **free Kaggle NVIDIA T4 GPU** triggered by GitHub Actions, no local GPU needed
- Tracks every experiment (hyperparameters, metrics, artifacts) in **MLflow on DagsHub**
- Pushes validated model weights to a **staging repository** on Hugging Face Hub first
- Uses a **human-in-the-loop production gate** where a reviewer inspects metrics before manual approval
- Deploys the approved model to a **production Hugging Face model repository**
- Serves it via a fully **custom-themed Gradio UI** with built-in live monitoring and drift analytics deployed on **Hugging Face Spaces**
- Logs every real-world inference to a **private telemetry dataset** on Hugging Face Hub
- Runs a **weekly Evidently AI drift check** and triggers automatic retraining when statistical drift is detected

---

## 📈 Production Metrics Baseline

Evaluated against a **clean, hand-verified validation benchmark of 247 samples** consisting exclusively of **native Gujarati script** to ensure real-world generalization, not synthetic proxies:

| Metric | Score | Target Threshold | Status |
|---|---|---|---|
| **Exact Match (EM)** | **51.82%** | > 40% | ✅ Passed |
| **F1 Score** | **73.02%** | > 60% | ✅ Passed |
| **Eval Loss** | **1.1189** | Stable convergence | ✅ Stable |
| **Average Inference Latency** | **< 180ms** | Low-latency | ✅ Optimized |

> **Why a strict 247-sample benchmark?** The validation partition was isolated entirely from the `ai4bharat/IndicQA` dataset before any training began. It was never touched during fine-tuning or synthetic generation, making it a clean held-out benchmark that reflects real-world Gujarati QA generalization.

---

## 🏗️ End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER & VERSIONING                             │
│                                                                             │
│  [Local Pipeline]  ──►  [DVC Metadata Tracking]  ──►  [Kaggle Dataset Hub]  │
│  generate_synthetic_qa.py                                                   │
│  preprocess.py                                                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ git push (dvc.lock or params.yaml or trai.py)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CI/CD ORCHESTRATION LAYER                           │
│                                                                             │
│  [GitHub Actions]  ──► [Inject Secrets]  ──►  [Kaggle API: Push Kernel]     │
│  train_and_deploy.yml       HF_TOKEN                                        │
│  retrain_on_drift.yml       MLFLOW_URI        [Kaggle T4 GPU Executes]      │
│                                                       ↓                     │
│                                                [MuRIL Fine-Tuning]          │
│                                                       ↓                     │
│                                     ┌─────────────────┴────────────────┐    │
│                                     ▼                                  ▼    │
│                        [DagsHub MLflow Server]            [HF Staging Repo] │
│                        (Logs: metrics, params,             Nikunj4/gujarati-│
│                          artifacts, run IDs)                  qa-staging    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  Polling: every 60s (up to 240 min)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HUMAN-IN-THE-LOOP PRODUCTION GATE                        │
│                                                                             │
│   GitHub sends email notification → Reviewer inspects metrics in workflow   │
│   logs (F1, EM, Eval Loss) → Manual approval in GitHub "production" env     │
│                                    │                                        │  
│                                    │  (on approval)                         │  
│                                    ▼                                        │
│                        [HF Production Model Repo]                           │
│                         Nikunj4/gujarati-qa-model                           │
│                       (model.safetensors — 954 MB)                          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVING & MONITORING LAYER                           │
│                                                                             │
│   [HF Spaces — Gradio UI]  ◄────────────────── [Production Model]           │
│    Custom dark-navy theme                                                   │
│    Built-in live dashboard                                                  │
│    Drift analysis on-demand                                                 │
│         │                                                                   │
│         │  Every prediction logged                                          │
│         ▼                                                                   │
│  [Private Telemetry Hub: Nikunj4/gujarati-qa-logs]  ◄── cloud_log_prediction│
│         │                                                                   │
│         ▼  Every Monday 2:00 AM UTC (retrain_on_drift.yml)                  │
│  [Evidently AI Drift Engine]  ─► drift detected? ─► [Kaggle T4 Retrain]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Engineering & Scaled Synthesis

The production model trains on **81,807 samples**, paired with a strict native 247-row evaluation benchmark. The dataset integrates three sources through a structured multi-stage processing engine:

### Source 1: Native Seed Dataset (`ai4bharat/IndicQA`)
Native Gujarati extractive QA pairs from `ai4bharat/IndicQA`. The **validation partition was isolated before training** and is used exclusively as the production evaluation benchmark. Only the training split participates in fine-tuning.

### Source 2: Cross-Lingual Adaptation Layer (`l3cube-pune/indic-squad`)
A massive translated SQuAD corpora filtered via **strict Gujarati Unicode range lookups** (U+0A80 to U+0AFF) to extract syntactically valid Gujarati samples. This cross-lingual adaptation layer adds broad structural diversity to the training set.

### Source 3: Automated Multi-Key Synthetic Generation Pipeline
A custom script (`src/data/generate_synthetic_qa.py`) generates synthetic QA pairs by targeting:
- **Wikipedia** articles in Gujarati
- The **`ai4bharat/sangraha` verified subset** (OCR documents, websites, transcripts)

To bypass free API rate-limiting thresholds, it uses an **automated token rotation architecture** across **7 Gemini API keys** and three models (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.5-flash`). Context was optimized to `text[:1500]` characters for maximum richness, producing **3 exact-substring verified QA pairs per document**.

---

## ⚠️ The Critical Preprocessing Breakthrough

Early pipeline versions suffered low EM/F1 scores due to a subtle but catastrophic bug. Standard text-cleaning routines (`clean_gujarati_text`) removed URLs, HTML tags, and irregular whitespace, but **character offset positions in the original dataset pointed to the un-cleaned text**. After cleaning, those offsets pointed to wrong positions, causing the model to train on misaligned answer spans.

**The fix** was to dynamically recalculate token bounds *post-normalization* using explicit substring searches in `src/data/preprocess.py`:

```python
# Systemic index alignment fix: recalculate answer_start after text cleaning
actual_start_idx = context_cleaned.find(answer_text_cleaned)
if actual_start_idx != -1:
    record['answers']['answer_start'] = [actual_start_idx]
```

This single adjustment, recalculating character matrices against the cleaned context, realigned the entire training target and allowed the model to hit its metric targets.

---

## ⚡ Hardware Acceleration & Model Fine-Tuning

### Model: `google/muril-base-cased` (MuRIL)

MuRIL (Multilingual Representations for Indian Languages) was selected over alternatives like IndicBERT due to its:
- Larger Indian-language pre-training corpus
- Better tokenized representation of complex Indic script structures
- Superior performance on downstream Gujarati extractive QA tasks

The fine-tuned model weighs **~954 MB** stored in SafeTensors format on Hugging Face Hub.

### Optimization Matrix

**VRAM Bottleneck Remediation - Mixed Precision Training**

Fine-tuning a 950 MB transformer with long context windows quickly saturates standard consumer GPUs. The pipeline applies `fp16=True` (Mixed-Precision Training), leveraging Tensor Cores on the NVIDIA T4 for a ~3× execution acceleration with a reduced memory footprint.

**Context Window & Hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| `max_length` | 384 | Maximizes context comprehension, minimizes attention padding |
| `num_train_epochs` | 4 | Reduced from 8 after dataset scaled to 81,807 to prevent memorizing synthetic artifacts |
| `warmup_steps` | 500 | Smooth acceleration slope covering ~5% of the total 10,544 training steps |
| `learning_rate` | 3e-5 | Standard QA fine-tuning rate |
| `fp16` | True | Mixed-precision for T4 Tensor Core acceleration |

**Overfitting Safeguard**

With the training dataset scaled to 81,807 samples, the total epochs were deliberately reduced to prevent the model from memorizing synthetic generation patterns while preserving foundational multilingual representations.

---

## 🔄 MLOps Pipeline & CI/CD Architecture

### Trigger Conditions

The `train_and_deploy.yml` workflow fires automatically when any of these files change on `main`:
- `params.yaml` - hyperparameter changes
- `src/model/train.py` - training code changes
- `dvc.lock` - new data version committed
- Manual trigger via `workflow_dispatch` at any time

### Kaggle GPU Execution

The pipeline **never runs heavy training on GitHub Action runners** (which have no GPU). Instead:

1. GitHub Actions installs the Kaggle CLI and sets up credentials from GitHub Encrypted Secrets
2. It pushes `kaggle/train_on_kaggle.py` (the remote training script) to the Kaggle kernel via `kaggle kernels push`
3. Inside Kaggle, the script: installs dependencies, clone Github repo(`Nikunj-Goswami4/gujarati-qa-mlops`), loads data from the Kaggle dataset (`nikunjgoswami/gujarati-qa-data`), runs `src/model/train.py`, logs metrics to DagsHub MLflow, and pushes trained weights directly to the **staging HF repository**.

### Automated Polling Engine

GitHub Actions monitors the remote Kaggle kernel using `KaggleApiExtended`, checking status every **60 seconds** for up to **240 minutes**:

```python
api = KaggleApi()
api.authenticate()

username = "nikunjgoswami"
kernel_name = "gujarati-qa-train"
kernel_slug = f"{username}/{kernel_name}"

print(f"Watching kernel: {kernel_slug}", flush=True)
print("Checking every 60 seconds (up to 240 minutes).\n",flush=True)

    for i in range(240):
        result = api.kernels_status(kernel_slug)
        status = str(getattr(result, "status", "unknown")).lower()
        print(f"[{i+1}/240] Status: {status}", flush=True)

        if "complete" in status:
            sys.exit(0)
        elif "error" in status:
            sys.exit(1)

        time.sleep(60)
```

### Human-in-the-Loop Production Gate

To protect production deployments, the system uses a **dual-repository staging architecture**:

1. The trained model is first pushed to `Nikunj4/gujarati-qa-staging` (staging repository)
2. The pipeline generates a metrics report (`reports/metrics.json`) and writes it as a non-cached artifact
3. GitHub Actions **pauses** at the `production` environment gate and sends an **email notification** to the reviewer
4. The reviewer inspects exact F1, Exact Match, and Loss values directly in the workflow log
5. Upon manual approval, the validated model artifacts are transferred to the **production repository** (`Nikunj4/gujarati-qa-model`)

```yaml
# GitHub Actions production environment with deployment protection
environment:
  name: production
  url: https://huggingface.co/spaces/Nikunj4/gujarati-qa-demo
```

### Drift-Triggered Auto-Retraining (Weekly)

`retrain_on_drift.yml` runs every **Monday at 2:00 AM UTC** via a cron schedule. If drift is detected, it triggers a full retraining cycle on Kaggle T4 GPU, using the same polling and approval infrastructure.

---

## 🚀 FastAPI Serving Architecture

The model is served through a **high-performance FastAPI server** (`src/api/main.py`) with the following design choices:

**Async Lifespan Manager:** The 950 MB transformer is loaded into memory **once on startup** using an `asynccontextmanager` lifespan, preventing per-request model reload overhead:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = GujaratiQAModel()   # loads once
    yield
    model = None
```

**Pydantic Validation Layers:** All incoming JSON is validated through `QARequest` / `QAResponse` schemas with strict field constraints (min/max length for question and context).

**Prediction Logger:** Every inference writes to `logs/predictions.jsonl` with a full payload: `timestamp`, `question`, `context_length`, `answer`, `confidence`,`answer_start`, `question_length`, `answer_length`, this feed drives the drift detection pipeline.

### Unit Testing, CI-Safe Isolation

The test suite (`tests/test_api.py`) evaluates the API without loading the actual 950 MB model on GitHub's resource-constrained runners. It uses `pytest` fixtures and `unittest.mock.patch` to mock the model layer entirely:

```python
@pytest.fixture(autouse=True)
def mock_model_prediction():
    with patch("src.api.main.model") as mock_model:
        mock_model.answer.return_value = {
            "answer": "ગાંધીનગર",
            "confidence": 0.92,
            "answer_start": 24,
            "answer_end": 33
        }
        yield mock_model
```

> A custom root `pytest.ini` combined with `__init__.py` files across all runtime directories solves Python module lookup failures on the GitHub runner.

---

## 🎨 Custom UI / UX — HF Spaces Live Demo

The Gradio UI (`app.py` in the HF Spaces repo) is deployed directly to Hugging Face Spaces and bypasses default Gradio themes entirely in favor of a **fully custom UI layout**:

**Color Palette & Visual Design**
- Background canvas: Dark navy `#0b0f1a`
- Card surfaces: Charcoal `#131929`
- Gradient accent line: Blue-purple `#6384ff → #9b6dff`
- Font: Inter, with soft weight variations for readability

**Interactive Features**
- **Quick-start suggestion pills:** pre-loaded native Gujarati example contexts covering geography, history, science, and cultural topics, wired to `gr.Button` listeners for instant one-click prediction workflows
- **Real-time model statistics panel:** displays current model commit hash, last-modified timestamp pulled from the HF Hub API on startup
- **Built-in confidence score display:** shows the model's confidence for each answer alongside the extracted span
- **Inline live monitoring dashboard:** confidence distribution histogram and inference stability timeline, powered by Plotly, rendered directly within the Space
- **On-demand drift analysis:** clicking "Run Drift Check" executes the full Evidently AI drift pipeline and returns a status indicator (`🟢 STABLE` or `🔴 DRIFT DETECTED`) without leaving the interface

<!-- **Layout** — Uses targeted CSS overrides to remove default Gradio chrome and maximize screen area for the context and question input blocks. -->

---

## 📊 Drift Analytics & Ephemeral Storage Solution

### The Ephemeral Storage Problem

Hugging Face Spaces run in ephemeral containers, any locally written file (including `logs/predictions.jsonl`) is permanently lost whenever the container restarts or is recycled. Standard file-based logging is therefore unreliable for production telemetry.

### The Solution: Direct Cloud Synchronization

Every prediction is written locally and immediately uploaded to a **private Hugging Face Dataset repository** (`Nikunj4/gujarati-qa-logs`) via `hf_api.upload_file`. The Space never relies on local persistence:

```python
def sync_logs_to_huggingface_cloud():
    hf_api.upload_file(
        path_or_fileobj=LOCAL_LOG_FILE,
        path_in_repo="predictions.jsonl",
        repo_id="Nikunj4/gujarati-qa-logs",
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="telemetry: automatically record production inference logs"
    )
```

On Space restart, `fetch_latest_cloud_logs()` pulls the full log history back down before any dashboard calculations.

### Drift Detection Engine (Evidently AI)

The drift check compares the **first half vs second half** of the collected prediction log using Evidently AI's `Report` API:

**Features monitored:** `confidence`, `question_length`, `answer_length`, `context_length`

```python
schema = DataDefinition(numerical_columns=cols)
reference_dataset = Dataset.from_pandas(reference_df, data_definition=schema)
current_dataset = Dataset.from_pandas(current_df, data_definition=schema)

report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
my_eval = report.run(reference_data=reference_dataset, current_data=current_dataset)
```

### Drift-Triggered Data Extraction

If drift is detected, the engine automatically extracts real production queries as new training candidates, applying these quality filters before exporting:

```python
# Quality filtering for safe retraining candidates
if raw_conf < 0.40:              continue   # skip low-confidence/guessing
if len(clean_answer) > 120:      continue   # skip abnormally long answers
if len(clean_answer) == 0:       continue   # skip empty answers
if clean_question in clean_answer: continue  # skip malformed pairs
```

The filtered set is saved to `logs/latest_drifted_data.jsonl` and immediately uploaded to `Nikunj4/gujarati-qa-logs` for use in the next retraining cycle.

<!-- > **Note on Evidently HTML Reports:** The warning *"The diff for this file is too large to render"* on the Hugging Face Web UI is expected. Evidently dashboard output files contain minified Plotly engine modules on single lines, which triggers browser layout safety warnings. The HTML file is valid and opens correctly when downloaded. -->

---

## 🛠️ Complete Local Setup Guide

### Prerequisites

- Python 3.10+
- Git
- Kaggle account (free) with API key configured
- Hugging Face account with a write token
- DagsHub account (for MLflow tracking)

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/Nikunj-Goswami4/gujarati-qa-mlops.git
cd gujarati-qa-mlops

# Create and activate a clean virtual environment
python -m venv venv
venv\Scripts\Activate.ps1       # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Secrets

Create a `.env` file in the project root (never commit this):

```bash
HF_TOKEN=hf_your_huggingface_write_token
MLFLOW_TRACKING_URI=mlflow_tracking_url_from_dagshub
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### Step 3: Run the Test Suite

```bash
# Run the full mocked test suite — no GPU needed, runs in seconds
pytest tests/ -v --tb=short
```

All tests use the mocked model fixture and should pass on any machine.

### Step 4: Check DVC Pipeline Status

```bash
# Check which pipeline stages are out of date
dvc status

# Reproduce any changed pipeline stages
dvc repro preprocess
```

### Step 5: Start the FastAPI Server Locally

```bash
# Loads the model from HuggingFace Hub (requires internet)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "ગુજરાતની રાજધાની કઈ છે?", "context": "ગુજરાત ભારતનું એક રાજ્ય છે. ગાંધીનગર ગુજરાતની રાજધાની છે."}'
```

### Step 6: Launch the Gradio UI Locally

```bash
# Accessible at http://localhost:7860
python src/ui/app.py
```

### Step 7: Launch the Monitoring Dashboard Locally

```bash
# Accessible at http://localhost:7861
python src/monitoring/dashboard.py
```

---

## 🔁 Adding New Training Data - Full Workflow

Whenever you generate a new batch of synthetic data or add a new data source, follow this workflow to sync the remote training environment and trigger a new training run:

```bash
# Step 1: Generate new synthetic QA pairs or download new data
python src/data/generate_synthetic_qa.py  #or python src/data/download.py

# Step 2: Merge and preprocess all data sources into processed/ + Data versioning in DVC
dvc repro preprocess

# Step 3: Upload the new processed dataset to Kaggle
#         (so the next Kaggle training run uses the updated data)
kaggle datasets version -p data/processed/ -m "Added new synthetic data batch"

# Step 4: Version-control the new data state (commit the DVC lock file, not the data)
git add dvc.lock
git commit -m "data: added new synthetic batch + new data source"

# Step 5: Push to GitHub, this auto-triggers retraining on Kaggle T4 GPU
#         (because dvc.lock changed, which is a train_and_deploy.yml trigger)
git push
```

> After `git push`, GitHub Actions detects the `dvc.lock` change, starts the training workflow, pushes the Kaggle kernel, waits for it to complete, then sends an email to the reviewer for production approval.

---

## 📂 Project Directory Structure

```
gujarati-qa-mlops/
│
├── .github/
│   └── workflows/
│       ├── train_and_deploy.yml        # Main CI/CD: Kaggle T4 GPU training, staging push,
│       │                               # human-in-the-loop gate, production deploy
│       └── retrain_on_drift.yml        # Weekly drift check: runs every Monday 2AM UTC,
│                                       # triggers Kaggle retraining if drift detected
│
├── data/
│   ├── raw/                            # Raw seed datasets (tracked by DVC, not in Git)
│   ├── processed/                      # Preprocessed & merged training data (DVC tracked)
│   │   ├── train.jsonl                 # 81,807 training samples
│   │   └── val.jsonl                   # NOT used during training
│   └── knowledge_base/                 # Gujarati contextual knowledge repository
│
├── docker/
│   ├── Dockerfile                      # Production API container configuration
│   └── docker-compose.yml              # Multi-service local environment (API + MLflow)
│
├── kaggle/
│   ├── kernel-metadata.json            # Kaggle kernel config: GPU type, dataset sources,
│   │                                   # kernel ID, privacy settings
│   └── train_on_kaggle.py              # Remote training script executed on Kaggle T4 GPU;
│                                       # installs deps, loads data, trains, logs to
│                                       # MLflow, pushes model to HF staging repo
│
├── logs/
│   └── predictions.jsonl               # Production inference log, synced to
│                                       # Nikunj4/gujarati-qa-logs on HF Hub after each entry
│
├── reports/
│   └── metrics.json                    # Non-cached CI artifact: eval_loss, f1, exact_match
│                                       # Written after each training run for gate review
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── logger.py                   # Prediction logger: writes to predictions.jsonl
│   │   ├── main.py                     # FastAPI app with async lifespan model loader,
│   │   │                               # /predict endpoint, /health endpoint
│   │   └── schemas.py                  # Pydantic QARequest / QAResponse / HealthResponse
│   │
│   ├── data/
│   │   ├── download.py                 # Downloads raw datasets from HuggingFace Hub
│   │   ├── generate_synthetic_qa.py    # Synthetic QA generator: rotates 7 Gemini API keys
│   │   │                               # across 3 Gemini models, produces 3 QA pairs/doc
│   │   └── preprocess.py               # Cleans data for training model
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # Exact Match + Token-level F1 calculation
│   │   ├── predict.py                  # GujaratiQAModel class: OOP inference wrapper
│   │   │                               # with confidence scoring (softmax of logits)
│   │   ├── run_evaluate.py             # Evaluation pipeline execution wrapper
│   │   └── train.py                    # MuRIL fine-tuning: Trainer API, MLflow logging,
│   │                                   # fp16, warmup_steps, evaluation_strategy
│   │
│   ├── monitoring/
│   │   ├── dashboard.py                # Local Gradio monitoring dashboard (port 7861)
│   │   │                               # Confidence distribution + inference timeline plots
│   │   └── drift_detector.py           # Evidently AI drift engine: DataDriftPreset +
│   │                                   # DataSummaryPreset, quality-filtered export
│   └──ui/
│       └── app.py                      # Local Gradio UI (port 7860); mirrors the HF Spaces
│
├── tests/
│   ├── test_api.py                     # Mocked FastAPI tests (no GPU load on CI runner)
│   │                                   # Uses pytest fixture + unittest.mock.patch
│   └── test_model.py                   # Core model unit tests: load, predict, EM, F1
│
├── dvc.yaml                            # DVC pipeline: preprocess → train stages with
│                                       # explicit deps, outs, params, and metrics
├── params.yaml                         # Hyperparameters: model name, max_length,
│                                       # learning_rate, batch_size, epochs, warmup_steps
├── pytest.ini                          # Root pytest config with testpaths and sys path fix
└── requirements.txt                    # Full dependency manifest
```

---

## 🧰 Complete Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Base Model** | `google/muril-base-cased` (MuRIL) | Multilingual Indian language pre-training |
| **Training Framework** | Hugging Face `Transformers` + `Trainer` API | Fine-tuning loop, evaluation, checkpointing |
| **Mixed Precision** | `fp16=True` | T4 Tensor Core acceleration, VRAM reduction |
| **Data Versioning** | DVC | Dataset pipeline versioning without Git bloat |
| **Experiment Tracking** | MLflow on DagsHub | Metrics, params, artifacts, run history |
| **CI/CD** | GitHub Actions | Automated trigger, orchestration, gate control |
| **Remote GPU** | Kaggle NVIDIA T4 (free) | Headless cloud training, no local GPU needed |
| **Model Registry** | Hugging Face Hub | Staging + production model repositories |
| **Serving API** | FastAPI + Uvicorn | High-performance async inference server |
| **API Validation** | Pydantic v2 | Request/response schema enforcement |
| **Containerization** | Docker + docker-compose | Portable API + MLflow local stack |
| **UI** | Gradio (custom CSS) | Live demo with built-in monitoring |
| **Live Deployment** | Hugging Face Spaces | Public demo hosting |
| **Drift Detection** | Evidently AI | Statistical feature drift analysis |
| **Telemetry Storage** | HF Hub Dataset | Persistent ephemeral-safe log store |
| **Synthetic Generation** | Gemini API (7-key rotation) | Scalable Gujarati QA pair creation |
| **Testing** | pytest + unittest.mock | CI-safe mocked API and model tests |
| **Language** | Python 3.10+ | — |

<!-- ---

## 🔗 All Project Resources

| Resource | Link |
|---|---|
| 🐙 **GitHub Repository** | [github.com/Nikunj-Goswami4/gujarati-qa-mlops](https://github.com/Nikunj-Goswami4/gujarati-qa-mlops) |
| 🤗 **Production Model** | [huggingface.co/Nikunj4/gujarati-qa-model](https://huggingface.co/Nikunj4/gujarati-qa-model) |
| 🤗 **Staging Model** | [huggingface.co/Nikunj4/gujarati-qa-staging](https://huggingface.co/Nikunj4/gujarati-qa-staging) |
| 🤗 **Live Demo (HF Spaces)** | [huggingface.co/spaces/Nikunj4/gujarati-qa-demo](https://huggingface.co/spaces/Nikunj4/gujarati-qa-demo) |
| 🤗 **Telemetry Dataset** | [huggingface.co/datasets/Nikunj4/gujarati-qa-logs](https://huggingface.co/datasets/Nikunj4/gujarati-qa-logs) |
| 📊 **MLflow Experiments** | [dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops.mlflow](https://dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops.mlflow) |
| 🔄 **DagsHub Repo** | [dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops](https://dagshub.com/Nikunj-Goswami4/gujarati-qa-mlops) |
| 🗄️ **Kaggle Dataset** | [kaggle.com/datasets/nikunjgoswami/gujarati-qa-data](https://www.kaggle.com/datasets/nikunjgoswami/gujarati-qa-data) | -->

<!-- ---

 ## ❓ Common Issues & Fixes

**Training fails on Kaggle with `error` status**
Go to `kaggle.com/code/YOUR_USERNAME/gujarati-qa-train` and read the kernel output logs. Most common causes: wrong dataset path in `kernel-metadata.json`, or a Python package installation failure.

**`kaggle kernels push` fails with "kernel not found" on first push**
This is expected on the very first push — the kernel doesn't exist yet. Run it once manually from your terminal (`cd kaggle && kaggle kernels push -p .`) to create it. Subsequent pushes from GitHub Actions will work correctly.

**Model not appearing on HuggingFace after training completes**
Verify that `HF_TOKEN` is added as a **Kaggle Secret** (not just a GitHub Secret). The model push to HF Hub happens from *inside* Kaggle — it needs the token there separately.

**GitHub Actions times out after 90 minutes**
Reduce `num_epochs` in `params.yaml` (try `2`), or check that your Kaggle dataset is not larger than expected.

**Out of memory during local training**
Reduce `batch_size` to 4 and `max_length` to 256 in `params.yaml`. For full training, always use the Kaggle T4 path.

**Gujarati text rendering as `\u0a97` escape sequences**
Ensure `ensure_ascii=False` in all `json.dumps()` calls and `encoding='utf-8'` in all file opens throughout the pipeline.

**Evidently `"The diff for this file is too large to render"` warning on HF**
Expected behavior — Evidently HTML reports embed minified Plotly JS on single lines, triggering GitHub/HF browser rendering limits. Download the file to view it.

**`pytest` failing with module import errors**
Confirm `pytest.ini` is at the project root with `testpaths = tests` and that `src/__init__.py` and all sub-package `__init__.py` files exist. -->

---

<div align="center">

<!-- Built with ❤️ for the open-source Indian AI ecosystem. -->

*Advancing NLP for Gujarati — one of India's most spoken languages.*

</div>
