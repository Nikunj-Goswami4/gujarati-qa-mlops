import pandas as pd
import json
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from datetime import datetime, timedelta
import os

def load_prediction_logs(log_file: str = "logs/predictions.jsonl") -> pd.DataFrame:
    records = []
    with open(log_file, 'r', encoding='utf-8') as f: # Explicitly enforce UTF-8 stream handling
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    return df

def run_drift_check(log_file: str = "logs/predictions.jsonl"):
    """
    Compare last 7 days of predictions vs. previous 7 days.
    Returns drift report and whether drift was detected.
    """
    df = load_prediction_logs(log_file)

    if len(df) < 20:
        print("Not enough data for drift analysis (need at least 20 records)")
        return None, False

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)

    reference_df = df[(df['timestamp'] >= two_weeks_ago) & (df['timestamp'] < week_ago)]
    current_df = df[df['timestamp'] >= week_ago]

    if len(reference_df) < 5 or len(current_df) < 5:
        print("Not enough data in time windows. Splitting 50/50 instead.")
        mid = len(df) // 2
        reference_df = df.iloc[:mid]
        current_df = df.iloc[mid:]

    # Features to monitor
    numeric_features = ["confidence", "question_length", "answer_length", "context_length"]

    # Keep only relevant columns
    cols = [c for c in numeric_features if c in df.columns]
    reference_df = reference_df[cols]
    current_df = current_df[cols]

    # column_mapping = ColumnMapping(numerical_features=cols)
    schema = DataDefinition(
        numerical_columns=cols
    )

    # Convert pandas structures into Evidently wrapper datasets
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=schema)
    current_dataset = Dataset.from_pandas(current_df, data_definition=schema)

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    my_eval = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset
    )

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    my_eval.save_html(report_path)

    # Check if drift detected
    try:
        if hasattr(my_eval, 'as_dict'):
            report_dict = my_eval.as_dict()
        elif hasattr(my_eval, 'dict'):
            report_dict = my_eval.dict()
        else:
            report_dict = json.loads(my_eval.json())
            
        drift_detected = False
        for metric in report_dict.get('metrics', []):
            if 'dataset_drift' in str(metric.get('metric', '')):
                drift_detected = metric.get('result', {}).get('dataset_drift', False)
                break

        # Save fresh drift logs block:
        if drift_detected:  
            print("Drift detected! Exporting raw production text logs for retraining...")
            # Grab the unstripped rows matching our current drift window from the original df
            raw_drift_rows = df[df.index.isin(current_df.index)]
            
            # Save them directly to logs folder
            output_path = "logs/latest_drifted_data.jsonl"
            
            # Convert the dataframe rows back into JSON lines format
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, row in raw_drift_rows.iterrows():

                    # Extract fields
                    raw_q = row.get("question")
                    raw_c = row.get("context")
                    raw_a = row.get("answer")
                    raw_s = row.get("answer_start")
                    raw_conf = row.get("confidence", 0.0)

                    clean_question = str(raw_q) if pd.notna(raw_q) else ""
                    clean_context = str(raw_c) if pd.notna(raw_c) else ""
                    clean_answer = str(raw_a) if pd.notna(raw_a) else ""
                    # Ensure answer_start is an integer and defaults to 0 if NaN/None
                    clean_start = int(raw_s) if pd.notna(raw_s) and raw_s is not None else 0

                    # Quality  Filtering
                    # Skip rows where the model was guessing blindly (Low Confidence)
                    if clean_context == "" or clean_question == "":
                        continue
                    if raw_conf < 0.40:
                        continue
                    # Skip rows where the model hallucinated a massive block of text as an answer
                    if len(clean_answer) > 120 or len(clean_answer) == 0:
                        continue
                    # Skip rows where the model glitched and copied the question into the answer
                    if clean_question in clean_answer:
                        continue

                    # Reconstruct fields to match train/val schema exactly
                    record = {
                        "id": str(row.get("id", f"drift_{idx}")),
                        "context": clean_context,
                        "question": clean_question,
                        "answers": {
                            "text": [clean_answer],
                            "answer_start": [clean_start]
                        }
                    }
                    # Write as clean, unescaped JSON lines to preserve Gujarati text glyphs
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Successfully exported drifted samples to: {output_path}")

    except Exception as e:
        print(f"Error parsing drift results: {e}")
        drift_detected = False

    print(f"Drift detected: {drift_detected}")
    print(f"Report saved to: {report_path}")

    return report_path, drift_detected

if __name__ == "__main__":
    run_drift_check()