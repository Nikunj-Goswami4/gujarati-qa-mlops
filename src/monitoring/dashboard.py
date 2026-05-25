import gradio as gr
import pandas as pd
import plotly.express as px
import os
import json
from datetime import datetime, timedelta
from .drift_detector import run_drift_check, load_prediction_logs

LOG_FILE = "logs/predictions.jsonl"

def get_dashboard_metrics():
    """Calculate baseline overview statistics from live prediction log streams."""
    if not os.path.exists(LOG_FILE):
        return "0", "N/A", "0", "0", gr.update(visible=False), gr.update(visible=False)
    
    try:
        df = load_prediction_logs(LOG_FILE)
        if df.empty:
            return "0", "N/A", "0", "0", gr.update(visible=False), gr.update(visible=False)
        
        total_preds = len(df)
        avg_conf = f"{df['confidence'].mean() * 100:.1f}%" if "confidence" in df.columns else "N/A"
        anomalies = len(df[df['confidence'] < 0.35]) if "confidence" in df.columns else 0
        
        # Calculate Last 24h Inferences smoothly
        try:
            if df['timestamp'].dt.tz is None:
                last_24h = len(df[df['timestamp'] >= (datetime.utcnow() - timedelta(hours=24))])
            else:
                last_24h = len(df[df['timestamp'] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=24))])
        except Exception:
            last_24h = 0
        
        # 📊 Chart 1: Interactive Confidence Distribution Histogram
        fig_dist = px.histogram(
            df, x="confidence", nbins=20,
            title="Statistical Confidence Distribution Space",
            labels={"confidence": "Model Confidence Limit", "count": "Prediction Volumetric Frequency"},
            color_discrete_sequence=["#6384ff"]
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dde3f0", title_font_family="Inter"
        )
        
        # 📈 Chart 2: Rolling Confidence Trends Over Time
        df_sorted = df.sort_values("timestamp")
        fig_time = px.line(
            df_sorted, x="timestamp", y="confidence",
            title="System Inference Stability Timeline",
            labels={"timestamp": "Execution Timeline", "confidence": "Confidence Margin"},
            color_discrete_sequence=["#9b6dff"]
        )
        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dde3f0", title_font_family="Inter"
        )
        
        return str(total_preds), avg_conf, str(anomalies), str(last_24h), fig_dist, fig_time
    except Exception as e:
        print(f"Error compiling dashboard visualization: {e}")
        return "Error", "Error", "Error", "Error", gr.update(visible=False), gr.update(visible=False)

def trigger_drift_analysis():
    """Calculate statistical population drift arrays live using Evidently AI."""
    try:
        report_path, drift_detected = run_drift_check(LOG_FILE)
        if not report_path:
            return "⚠️ Insufficient historical document records logged to calculate drift targets."
        
        status = "🔴 DRIFT DETECTED: Statistical feature patterns are drifting. Retraining recommended!!" if drift_detected else "🟢 STABLE: Data distributions match profile metrics."
        
        return f"### Validation State\n## {status}\n\n*Interactive log report compiled inside project folder location:*\n`{report_path}`"
    except Exception as e:
        return f"❌ Monitoring execution cluster pipeline failure: {e}"

# Custom Dark Observability Palette with layout spacing classes
CSS = """
body, .gradio-container { background-color: #0b0f1a !important; color: #dde3f0 !important; font-family: 'Inter', sans-serif !important; }
.stat-box { background: #131929 !important; border: 1px solid rgba(99,130,255,0.12) !important; border-radius: 12px !important; padding: 15px !important; text-align: center; }
.btn-drift { background: linear-gradient(135deg, #6384ff 0%, #9b6dff 100%) !important; border: none !important; color: white !important; font-weight: 500 !important; border-radius: 8px !important; padding: 12px !important; cursor: pointer !important; }
.chart-row { margin-top: 25px !important; gap: 20px !important; }
.drift-row { margin-top: 35px !important; border-top: 1px solid rgba(99,130,255,0.12) !important; padding-top: 25px !important; }
"""

with gr.Blocks(title="Gujarati QA Observability Suite") as dashboard:
    gr.HTML(
        "<div style='padding: 20px 0; border-bottom: 1px solid rgba(99,130,255,0.12); margin-bottom: 25px;'>"
        "<h1 style='font-size: 1.8rem; color: #dde3f0; margin: 0;'>📊 System Observability & Drift Telemetry Dashboard</h1>"
        "<p style='color: #7a87a8; margin: 5px 0 0 0;'>Evidently AI + Plotly pipeline checking feature space alignment for regional language assets.</p>"
        "</div>"
    )
    
    with gr.Row():
        with gr.Column(elem_classes="stat-box"):
            gr.Markdown("<p style='color: #7a87a8; margin:0; text-transform: uppercase; font-size:0.7rem; letter-spacing:0.08em;'>Total Request Inferences</p>")
            stat_total = gr.Markdown("<h2 style='font-size: 2.2rem; color: #dde3f0; margin:5px 0;'>0</h2>")
        with gr.Column(elem_classes="stat-box"):
            gr.Markdown("<p style='color: #7a87a8; margin:0; text-transform: uppercase; font-size:0.7rem; letter-spacing:0.08em;'>Mean Model Confidence</p>")
            stat_avg = gr.Markdown("<h2 style='font-size: 2.2rem; color: #3dd68c; margin:5px 0;'>N/A</h2>")
        with gr.Column(elem_classes="stat-box"):
            gr.Markdown("<p style='color: #7a87a8; margin:0; text-transform: uppercase; font-size:0.7rem; letter-spacing:0.08em;'>Low Confidence Anomalies</p>")
            stat_anom = gr.Markdown("<h2 style='font-size: 2.2rem; color: #ff6b6b; margin:5px 0;'>0</h2>")
        with gr.Column(elem_classes="stat-box"):
            gr.Markdown("<p style='color: #7a87a8; margin:0; text-transform: uppercase; font-size:0.7rem; letter-spacing:0.08em;'>Last 24h Predictions</p>")
            stat_24h = gr.Markdown("<h2 style='font-size: 2.2rem; color: #ff9f43; margin:5px 0;'>0</h2>")

    with gr.Row(elem_classes="chart-row"):
        chart_dist = gr.Plot(label="Confidence Level Distribution Histogram")
        chart_time = gr.Plot(label="Inference Timeline Rolling Stability")

    with gr.Row(elem_classes="drift-row"):
        with gr.Column():
            gr.Markdown("### ⚙️ Deep Pipeline Population Stability Verification")
            gr.Markdown("Triggers a live multi-parametric population drift evaluation check comparing sample window distributions via Kolmogorov-Smirnov statistical testing frameworks.")
            drift_btn = gr.Button("Execute Data Drift Check Verification →", elem_classes="btn-drift")
        with gr.Column():
            drift_output = gr.Markdown("*Awaiting verification loop signal trigger...*")

    # Connect lifecycle events
    dashboard.load(
        fn=get_dashboard_metrics,
        outputs=[stat_total, stat_avg, stat_anom, stat_24h, chart_dist, chart_time]
    )
    drift_btn.click(
        fn=trigger_drift_analysis,
        outputs=drift_output
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        theme=gr.themes.Soft(),
        css=CSS
    )