import os

# Define absolute paths based on your system layout
BASE_DIR = r"D:\04 Nikunj (D Drive)\Personal Projects\06 Gujarati QA System (MLOps)\gujarati-qa-mlops"
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
EXP_1_DIR = os.path.join(MLRUNS_DIR, "1")
RUN_ID = "34496a874ce64205a369693b5651882c"
RUN_DIR = os.path.join(EXP_1_DIR, RUN_ID)

print("⚡ Initializing MLflow FileStore Database Repair Sequence...")

# Step 1: Clean up any accidental root-level metadata files causing UI corruption
root_meta = os.path.join(MLRUNS_DIR, "meta.yaml")
if os.path.exists(root_meta):
    os.remove(root_meta)
    print("🧹 Cleared corrupted root-level meta.yaml file.")

# Step 2: Ensure default experiment '0' exists with valid Docker configurations
os.makedirs(os.path.join(MLRUNS_DIR, "0"), exist_ok=True)
with open(os.path.join(MLRUNS_DIR, "0", "meta.yaml"), "w", encoding="utf-8") as f:
    f.write(
        "artifact_location: file:///mlruns/0\n"
        "creation_time: 1779647400000\n"
        "experiment_id: '0'\n"
        "last_update_time: 1779647400000\n"
        "lifecycle_stage: active\n"
        "name: Default\n"
    )
print("✅ Re-indexed Default Experiment '0' paths.")

# Step 3: Ensure experiment '1' exists with valid Docker configurations
os.makedirs(EXP_1_DIR, exist_ok=True)
with open(os.path.join(EXP_1_DIR, "meta.yaml"), "w", encoding="utf-8") as f:
    f.write(
        "artifact_location: file:///mlruns/1\n"
        "creation_time: 1779647400000\n"
        "experiment_id: '1'\n"
        "last_update_time: 1779647400000\n"
        "lifecycle_stage: active\n"
        "name: gujarati-qa\n"
    )
print("✅ Re-indexed Gujarati-QA Experiment '1' paths.")

# Step 4: Rebuild the mandatory sub-folders inside your training run directory
if os.path.exists(RUN_DIR):
    for subfolder in ["metrics", "params", "tags"]:
        os.makedirs(os.path.join(RUN_DIR, subfolder), exist_ok=True)
    print("📁 Rebuilt missing metadata directories (metrics, params, tags).")

    # Step 5: Inject vital tracking parameters to satisfy table rendering loops
    params_dict = {
        "model_name": "google/muril-base-cased",
        "num_epochs": "8",
        "learning_rate": "3e-5",
        "batch_size": "16"
    }
    for p_name, p_val in params_dict.items():
        with open(os.path.join(RUN_DIR, "params", p_name), "w", encoding="utf-8") as f:
            f.write(str(p_val))

    # Step 6: Inject essential environment tags to prevent UI array lookup crashes
    tags_dict = {
        "mlflow.user": "root",
        "mlflow.source.name": "src/model/train.py",
        "mlflow.source.type": "LOCAL",
        "mlflow.runName": "colab-training-run"
    }
    for t_name, t_val in tags_dict.items():
        with open(os.path.join(RUN_DIR, "tags", t_name), "w", encoding="utf-8") as f:
            f.write(str(t_val))

    # Step 7: Overwrite run-level meta.yaml with clean paths mapped to the container volume
    with open(os.path.join(RUN_DIR, "meta.yaml"), "w", encoding="utf-8") as f:
        f.write(
            f"artifact_uri: file:///mlruns/1/{RUN_ID}/artifacts\n"
            f"end_time: 1779648300000\n"
            f"entry_point_name: ''\n"
            f"experiment_id: '1'\n"
            f"lifecycle_stage: active\n"
            f"run_id: {RUN_ID}\n"
            f"run_name: colab-training-run\n"
            f"run_uuid: {RUN_ID}\n"
            f"start_time: 1779647400000\n"
            f"status: 3\n"
            f"user_id: root\n"
        )
    print("📝 Rewrote run metadata mapping file with container-safe pointers.")
    print("🚀 Repair complete! Run your docker-compose reboot commands next.")
else:
    print(f"❌ Error: Could not locate run directory at: {RUN_DIR}")