import gradio as gr
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

EXAMPLE_CONTEXTS = {
    "🏙️ Gujarat Geography": "ગુજરાત ભારતના પશ્ચિમ ભાગમાં આવેલું છે. ગાંધીનગર ગુજરાતની રાજધાની છે. અમદાવાદ ગુજરાતનું સૌથી મોટું શહેર છે. ગુજરાતની સ્થાપના ૧ મે ૧૯૬૦ ના રોજ થઈ હતી.",
    "🕊️ Mahatma Gandhi": "મહાત્મા ગાંધીનો જન્મ ૨ ઓક્ટોબર ૧૮૬૯ ના રોજ પોરબંદર, ગુજરાત, ભારતમાં થયો હતો. તેઓ ભારતીય સ્વતંત્રતા ચળવળના નેતા હતા. ગાંધીજી અહિંસાના પૂજારી હતા.",
    "🌊 Narmada River": "નર્મદા નદી ગુજરાતની સૌથી મોટી નદી છે. તેને ગુજરાતની જીવાદોરી કહેવામાં આવે છે. સરદાર સરોવર બંધ નર્મદા નદી પર બાંધવામાં આવ્યો છે.",
    "🔬 Science": "ઓક્સિજન એ એક રાસાયણિક તત્વ છે. તેનો રાસાયણિક પ્રતીક O છે. ઓક્સિજન વિના જીવસૃષ્ટિ અશક્ય છે. પૃથ્વીના વાતાવરણમાં ૨૧% ઓક્સિજન છે.",
    "🌍 World Facts": "માઉન્ટ એવરેસ્ટ વિશ્વનો સૌથી ઊંચો પર્વત છે. તે હિમાલય પર્વતમાળામાં આવેલો છે. તેની ઊંચાઈ ૮૮૪૮.૮૬ મીટર છે. તે નેપાળ અને ચીનની સરહદ પર આવેલો છે.",
}

EXAMPLE_QUESTIONS = {
    "🏙️ Gujarat Geography": "ગુજરાતની રાજધાની કઈ છે?",
    "🕊️ Mahatma Gandhi": "ગાંધીજીનો જન્મ ક્યાં થયો હતો?",
    "🌊 Narmada River": "ગુજરાતની સૌથી મોટી નદી કઈ છે?",
    "🔬 Science": "ઓક્સિજનનો રાસાયણિક પ્રતીક શું છે?",
    "🌍 World Facts": "વિશ્વનો સૌથી ઊંચો પર્વત કયો છે?",
}

def answer_question(question, context):
    if not question.strip() or not context.strip():
        return "Please enter both a question and context.", 0.0
    try:
        r = requests.post(API_URL, json={"question": question, "context": context}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            ans = d.get("answer", "") or "No answer found."
            return ans, round(d.get("confidence", 0.0), 4)
        return f"API Error {r.status_code}", 0.0
    except requests.exceptions.ConnectionError:
        return "Cannot connect to API. Ensure FastAPI is running on port 8000.", 0.0
    except Exception as e:
        return f"Error: {e}", 0.0

def load_example(k):
    if not k:
        return "", ""
    return EXAMPLE_CONTEXTS.get(k, ""), EXAMPLE_QUESTIONS.get(k, "")

# Ultimate override css matrix
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --bg2:       #0f1422;
    --card:      #131929;
    --card2:     #182035;
    --border:    rgba(99,130,255,0.12);
    --border-hi: rgba(99,130,255,0.28);
    --txt:       #dde3f0;
    --txt2:      #7a87a8;
    --txt3:      #8fa8ff;
    --acc:       #6384ff;
    --acc2:      #9b6dff;
    --ok:        #3dd68c;
    --r:         12px;
}

*,*::before,*::after { box-sizing:border-box; }

body, .gradio-container {
    background-color: var(--bg) !important;
    background-image: 
        radial-gradient(circle at 50% -10%, rgba(80,100,255,0.18), transparent 55%),
        radial-gradient(circle at 5%, 75%, rgba(130,80,255,0.08), transparent 45%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--txt) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(99,130,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,130,255,0.02) 1px, transparent 1px);
    background-size: 48px 48px;
}

footer, .svelte-1ipelgc { display: none !important; }
.block { background: transparent !important; border: none !important; padding: 0 !important; }
label > span, .label-wrap span { display: none !important; }

/* ── NAVIGATION BAR ── */
.nav {
    border-bottom: 1px solid var(--border);
    background: rgba(11,15,26,0.85);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    position: relative; z-index: 10; width: 100%;
}
.nav-i {
    max-width: 1400px; margin: 0 auto; padding: 0 40px;
    height: 52px; display: flex; align-items: center; justify-content: space-between;
}
.nav-logo {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--txt); letter-spacing: 0.02em;
}
.nav-logo span { color: var(--acc); }
.nav-pill {
    display: flex; align-items: center; gap: 7px; font-size: 0.68rem; color: var(--txt2);
    border: 1px solid var(--border); border-radius: 100px; padding: 5px 14px; background: rgba(99,130,255,0.05);
}
.dot-live {
    width: 6px; height: 6px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px var(--ok);
    animation: blink 2.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── HERO HEADER ── */
.hero {
    width: 100%; max-width: 1400px; margin: 0 auto; padding: 60px 40px 40px 40px;
    border-bottom: 1px solid var(--border); position: relative; z-index: 5;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--txt2); margin-bottom: 18px; border: 1px solid var(--border); border-radius: 100px;
    padding: 5px 14px; background: rgba(99,130,255,0.06);
}
.hero-tag::before {
    content:''; width:5px; height:5px; border-radius:50%; background: var(--acc); box-shadow: 0 0 8px var(--acc);
}
.hero-h {
    font-size: clamp(2rem, 3.2vw, 2.8rem); font-weight: 500; line-height: 1.2; letter-spacing: -0.03em;
    color: var(--txt); margin-bottom: 12px;
}
.hero-h em {
    font-style: normal; background: linear-gradient(125deg, #7c9fff 0%, #b07dff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-p {
    font-size: 0.9rem; color: var(--txt2); font-weight: 300; line-height: 1.6; max-width: 520px; margin-bottom: 28px;
}

.mbar {
    display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: var(--r);
    overflow: hidden; background: var(--card);
}
.mc {
    padding: 10px 22px; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 3px;
}
.mc:last-child { border-right: none; }
.mc-n { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 500; color: var(--txt); }
.mc-l { font-size: 0.58rem; color: var(--txt2); text-transform: uppercase; letter-spacing: 0.08em; }
.mc-ok { font-size: 0.56rem; color: var(--ok); }

/* ── CORE WORKSPACE WRAPPER ── */
.workspace-grid {
    width: 100%; max-width: 1400px; margin: 0 auto; padding: 32px 40px 72px 40px;
    position: relative; z-index: 5;
}

/* ── 🛠️ FAIL-SAFE STYLING FOR CONTAINER CORES ── */
/* Forces every nested child structural div inside cards to use the dark-blue theme color */
.card, 
.card div, 
.card form,
.gradio-container .card,
.gradio-container .gr-group-inner {
    background-color: #131929 !important; 
    background: #131929 !important;
    border-color: var(--border) !important;
}

.card {
    border: 1px solid var(--border) !important; 
    border-radius: var(--r) !important;
    overflow: hidden; 
    box-shadow: 0 2px 0 rgba(99,130,255,0.08) inset, 0 20px 60px rgba(0,0,0,0.35) !important;
    transition: border-color 0.2s, box-shadow 0.2s; 
    width: 100%;
}
.card:hover {
    border-color: var(--border-hi) !important;
    box-shadow: 0 2px 0 rgba(99,130,255,0.12) inset, 0 0 0 1px rgba(99,130,255,0.08), 0 24px 64px rgba(0,0,0,0.4) !important;
}

/* Header strip background definition override */
.ch, .card .ch {
    padding: 12px 20px; border-bottom: 1px solid var(--border) !important; display: flex; align-items: center; gap: 10px; background: var(--card2) !important;
}
.c-num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--txt2);
    border: 1px solid rgba(99,130,255,0.15); border-radius: 4px; padding: 2px 7px;
}
.c-ttl { font-size: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--txt2); }
.cb { padding: 20px; display: flex; flex-direction: column; gap: 14px; background: transparent !important; }

.fl { font-size: 0.62rem; font-weight: 500; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }

/* ── RE-PROTECT INPUT CORES ── */
/* Re-applies the navy layout rules exclusively onto inputs so they stay beautifully defined */
.card textarea, 
.card input[type="text"], 
.card input[type="number"], 
.card select,
.card .wrap-inner {
    background-color: #161e33 !important; 
    background: #161e33 !important;
    border: 1px solid rgba(99,130,255,0.18) !important; 
    border-radius: 8px !important;
    color: #dde3f0 !important; 
    font-family: 'Inter', sans-serif !important; 
    font-size: 0.88rem !important;
    font-weight: 300 !important; 
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.card textarea:focus, .card input:focus {
    border-color: rgba(99,130,255,0.55) !important; box-shadow: 0 0 0 3px rgba(99,130,255,0.1) !important; background: #1c2744 !important;
}
textarea::placeholder, input::placeholder { color: #4a587a !important; font-size: 0.82rem !important; }

/* Workspace Quick Selection Row Buttons */
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; }
.pill-row button, .card .pill-row button {
    background: rgba(99,130,255,0.06) !important; border: 1px solid rgba(99,130,255,0.18) !important;
    border-radius: 100px !important; color: var(--txt2) !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important; padding: 5px 12px !important; min-width: unset !important; height: auto !important;
    transition: all 0.15s ease !important;
}
.pill-row button:hover {
    background: rgba(99,130,255,0.16) !important; border-color: rgba(99,130,255,0.35) !important; color: var(--txt) !important;
}

/* Monochromatic High Contrast Action Trigger */
.btn-go, .card .btn-go {
    width: 100% !important; background: linear-gradient(135deg, var(--acc) 0%, var(--acc2) 100%) !important;
    border: none !important; border-radius: 8px !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important; font-weight: 500 !important;
    letter-spacing: 0.03em !important; padding: 13px !important; cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(99,130,255,0.25) !important; transition: filter 0.15s, box-shadow 0.2s !important;
}
.btn-go:hover { filter: brightness(1.12) !important; box-shadow: 0 6px 24px rgba(99,130,255,0.4) !important; }

.ans-box textarea { border-color: rgba(99,130,255,0.22) !important; background: rgba(99,130,255,0.04) !important; color: var(--txt) !important; }

/* ── PIPELINE ARCHITECTURE LOGS DIAGNOSTIC SHEET ── */
.atab, .gradio-container .atab {
    background-color: var(--card) !important;
    background: var(--card) !important; 
    border: 1px solid var(--border) !important; 
    border-radius: var(--r) !important;
    overflow: hidden; margin-top: 18px; box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important; width: 100%;
}
.arow {
    display: flex; align-items: center; justify-content: space-between; padding: 11px 20px;
    border-bottom: 1px solid var(--border); font-size: 0.8rem;
}
.arow:last-child { border-bottom: none; }
.ak { color: var(--txt2); }
.av {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; background: rgba(99,130,255,0.1);
    border: 1px solid rgba(99,130,255,0.22); color: #8fa8ff; padding: 2px 9px; border-radius: 5px;
}
.avp { color: var(--txt); font-size: 0.8rem; }

.sstrip, .gradio-container .sstrip { display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px solid var(--border); background: var(--card2) !important; }
.sc { padding: 12px 20px; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 2px; }
.sc:last-child { border-right: none; }
.sn { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 500; color: var(--txt); }
.sl { font-size: 0.58rem; color: var(--txt2); text-transform: uppercase; letter-spacing: 0.06em; }
.st { font-size: 0.56rem; color: var(--ok); }

/* ── COMPACT STICKY FOOTER ── */
.ft { border-top: 1px solid var(--border); background: var(--bg2); width: 100%; position: relative; z-index: 5; margin-top: 40px; }
.ft-i {
    max-width: 1400px; margin: 0 auto; padding: 16px 40px;
    display: flex; align-items: center; justify-content: space-between; font-size: 0.68rem; color: #4a587a;
}
.tt { display: flex; gap: 6px; flex-wrap: wrap; }
.tg {
    padding: 3px 10px; border: 1px solid var(--border); border-radius: 100px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: var(--txt2); background: rgba(99,130,255,0.04);
}

@keyframes up { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
.hero { animation: up 0.4s ease both; }
.workspace-grid { animation: up 0.4s 0.06s ease both; }
"""

NAV_HTML = """
<div class="nav"><div class="nav-i">
  <div class="nav-logo">gujarati<span>.</span>qa</div>
  <div class="nav-pill"><div class="dot-live"></div>API Node Online · localhost:8000</div>
</div></div>
"""

HERO_HTML = """
<div class="hero">
  <div class="hero-tag">NLP · Regional Language Parsing · MLOps</div>
  <h1 class="hero-h">Extract answers from<br><em>Gujarati text, precisely.</em></h1>
  <p class="hero-p">Fine-tuned regional language transformer locating target span arrays inside corpus documents — built on IndicQA and Gemini synthetic samples.</p>
  <div class="mbar">
    <div class="mc"><div class="mc-n">48.99%</div><div class="mc-l">Exact Match</div><div class="mc-ok">✓ target &gt;40%</div></div>
    <div class="mc"><div class="mc-n">71.16%</div><div class="mc-l">F1 Score</div><div class="mc-ok">✓ target &gt;60%</div></div>
    <div class="mc"><div class="mc-n">~3,000</div><div class="mc-l">Train Samples</div><div class="mc-ok">IndicQA + Synthetic</div></div>
    <div class="mc"><div class="mc-n">MuRIL</div><div class="mc-l">Backbone</div><div class="mc-ok">google/muril-base-cased</div></div>
  </div>
</div>
"""

ARCH_HTML = """
<div class="atab">
  <div class="ch"><span class="c-num">INFO</span><span class="c-ttl">Model Architecture Specs</span></div>
  <div class="arow"><span class="ak">Backbone Layer</span><span class="av">google/muril-base-cased</span></div>
  <div class="arow"><span class="ak">Task Objective</span><span class="avp">Extractive Question Answering</span></div>
  <div class="arow"><span class="ak">Dataset Profile</span><span class="avp">IndicQA + Gemini Synthetic Data</span></div>
  <div class="arow"><span class="ak">Serving Core</span><span class="avp">FastAPI · Uvicorn · Docker Isolation</span></div>
  <div class="arow"><span class="ak">Registry Layer</span><span class="avp">MLflow Server · DVC Pointer Files</span></div>
  <div class="sstrip">
    <div class="sc"><div class="sn">48.99%</div><div class="sl">Exact Match</div><div class="st">✓ &gt;40%</div></div>
    <div class="sc"><div class="sn">71.16%</div><div class="sl">F1 Score</div><div class="st">✓ &gt;60%</div></div>
    <div class="sc"><div class="sn">~3K+</div><div class="sl">Volume</div><div class="st">Balanced</div></div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="ft"><div class="ft-i">
  <span>Gujarati QA Interface · Regional Language Processing Deployment</span>
  <div class="tt">
    <span class="tg">MuRIL</span><span class="tg">FastAPI</span>
    <span class="tg">Gradio</span><span class="tg">MLflow</span>
    <span class="tg">Docker</span>
  </div>
</div></div>
"""

with gr.Blocks(css=CSS, title="Gujarati QA Engine") as demo:

    gr.HTML(NAV_HTML)
    gr.HTML(HERO_HTML)

    with gr.Row(elem_classes="workspace-grid"):
        
        # Column 01: Interface Inputs
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="ch"><span class="c-num">01</span><span class="c-ttl">Workspace Parameters</span></div>')
                with gr.Column(elem_classes="cb"):
                    gr.HTML('<div class="fl">Quick Select Sample Profile</div>')
                    with gr.Row(elem_classes="pill-row"):
                        pill_geo    = gr.Button("Geography 🏙️", size="sm")
                        pill_gandhi = gr.Button("Gandhi 🕊️", size="sm")
                        pill_river  = gr.Button("Narmada 🌊", size="sm")
                        pill_sci    = gr.Button("Science 🔬", size="sm")
                        pill_world  = gr.Button("World 🌍", size="sm")
                    
                    example_selector = gr.Dropdown(
                        choices=list(EXAMPLE_CONTEXTS.keys()),
                        label="", container=False, visible=False
                    )
                    
                    gr.HTML('<div class="fl">Gujarati Reference Context</div>')
                    context_input = gr.Textbox(
                        label="", container=False, lines=7, show_label=False,
                        placeholder="અહીં ગુજરાતીમાં ફકરો લખો..."
                    )
                    
                    gr.HTML('<div class="fl">Input Question</div>')
                    question_input = gr.Textbox(
                        label="", container=False, lines=2, show_label=False,
                        placeholder="તમારો પ્રશ્ન અહીં પૂછો..."
                    )
                    
                    submit_btn = gr.Button(
                        "Extract Answer →",
                        variant="primary", elem_classes="btn-go"
                    )

        # Column 02: Output Telemetry
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="ch"><span class="c-num">02</span><span class="c-ttl">Performance Metrics</span></div>')
                with gr.Column(elem_classes="cb"):
                    gr.HTML('<div class="fl">Extracted Answer</div>')
                    answer_output = gr.Textbox(
                        label="", container=False, lines=3, show_label=False,
                        placeholder="Awaiting pipeline activation signal...",
                        interactive=False, elem_classes="ans-box"
                    )
                    
                    gr.HTML('<div class="fl">Confidence Score</div>')
                    confidence_output = gr.Number(
                        label="", container=False, show_label=False,
                        precision=4, interactive=False
                    )
            
            gr.HTML(ARCH_HTML)

    gr.HTML(FOOTER_HTML)

    # Routing interface events logic flows
    def load_geo():    return load_example("🏙️ Gujarat Geography")
    def load_gandhi(): return load_example("🕊️ Mahatma Gandhi")
    def load_river():  return load_example("🌊 Narmada River")
    def load_sci():    return load_example("🔬 Science")
    def load_world():  return load_example("🌍 World Facts")

    pill_geo.click(fn=load_geo,    outputs=[context_input, question_input])
    pill_gandhi.click(fn=load_gandhi, outputs=[context_input, question_input])
    pill_river.click(fn=load_river,  outputs=[context_input, question_input])
    pill_sci.click(fn=load_sci,    outputs=[context_input, question_input])
    pill_world.click(fn=load_world,  outputs=[context_input, question_input])

    example_selector.change(
        fn=load_example, 
        inputs=example_selector,
        outputs=[context_input, question_input]
    )
    submit_btn.click(
        fn=answer_question, 
        inputs=[question_input, context_input],
        outputs=[answer_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)