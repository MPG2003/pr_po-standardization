"""
PR/PO Intelligence Dashboard
- Server-side Groq API key (hidden from users)
- ML model /api/classify endpoint
- Upgraded to llama-3.3-70b-versatile
"""

import os
import requests as http_requests
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# ── Groq config ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# ── ML model (loaded once at startup) ────────────────────────
nlp_pipeline = None

def load_model():
    global nlp_pipeline
    model_path = os.path.join(os.path.dirname(__file__), "nlp_model_final.pkl")
    if os.path.exists(model_path):
        import pickle
        with open(model_path, "rb") as f:
            nlp_pipeline = pickle.load(f)
        print("✓ ML model loaded from nlp_model_final.pkl")
    else:
        print("⚠ nlp_model_final.pkl not found — /api/classify will use keyword fallback")

load_model()

# ── Page ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("templates/dashboard.html")

# ── API: Chat via Groq (server-side key — hidden from users) ──

@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    body = request.get_json(force=True)
    messages   = body.get("messages", [])
    system     = body.get("system", "You are a helpful SAP procurement analyst.")
    max_tokens = body.get("max_tokens", 800)

    payload = {
        "model":       GROQ_MODEL,
        "max_tokens":  max_tokens,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            *messages[-14:]   # last 14 messages to stay within token limits
        ]
    }

    try:
        resp = http_requests.post(
            GROQ_URL,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=60,
        )
        data = resp.json()
        if resp.status_code == 200 and "choices" in data:
            text = data["choices"][0]["message"]["content"]
            return jsonify({"text": text}), 200
        else:
            err = data.get("error", {}).get("message", str(data))
            return jsonify({"error": err}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: ML classify descriptions ────────────────────────────

KEYWORD_MAP = [
    ("bearing","MECH"),("hydraulic pump","MECH"),("gear box","MECH"),("v belt","MECH"),
    ("shaft","MECH"),("pulley","MECH"),("sprocket","MECH"),("valve","MECH"),
    ("motor","ELEC"),("circuit breaker","ELEC"),("contactor","ELEC"),("led","ELEC"),
    ("fuse","ELEC"),("transformer","ELEC"),("vfd","ELEC"),("plc","ELEC"),("relay","ELEC"),
    ("laptop","ITEQ"),("keyboard","ITEQ"),("monitor","ITEQ"),("ups","ITEQ"),("ssd","ITEQ"),
    ("safety helmet","SAFE"),("safety gloves","SAFE"),("safety shoes","SAFE"),
    ("fire extinguisher","SAFE"),("safety goggles","SAFE"),("dust mask","SAFE"),
    ("a4 paper","OFFC"),("pen","OFFC"),("stapler","OFFC"),("toner","OFFC"),("notepad","OFFC"),
    ("steel","ROH"),("aluminium","ROH"),("copper","ROH"),("pipe","ROH"),("rubber","ROH"),
    ("welding","MTNC"),("nut bolt","MTNC"),("drill","MTNC"),("o ring","MTNC"),("grease","MTNC"),
    ("engine oil","CHEM"),("hydraulic oil","CHEM"),("coolant","CHEM"),("acetone","CHEM"),
    ("cardboard","PACK"),("bubble wrap","PACK"),("pallet","PACK"),("stretch wrap","PACK"),
    ("coffee","FOOD"),("tea","FOOD"),("drinking water","FOOD"),("biscuit","FOOD"),("milk","FOOD"),
]

MATERIAL_GROUPS = {
    "ROH":"Raw Materials","MECH":"Mechanical Parts","ELEC":"Electrical Components",
    "OFFC":"Office Supplies","SAFE":"Safety Equipment","ITEQ":"IT Equipment",
    "MTNC":"Maintenance & Repair","PACK":"Packaging Materials",
    "CHEM":"Chemicals & Lubricants","FOOD":"Canteen & Food Supplies",
}

def keyword_classify(desc):
    tl = str(desc).lower()
    for kw, grp in KEYWORD_MAP:
        if kw in tl:
            return grp, 0.72
    return "OFFC", 0.30

@app.route("/api/classify", methods=["POST"])
def classify():
    body  = request.get_json(force=True)
    descs = body.get("descriptions", [])
    if not descs:
        return jsonify({"error": "No descriptions provided"}), 400

    results = []
    if nlp_pipeline:
        import numpy as np
        lowers = [str(d).lower() for d in descs]
        preds  = nlp_pipeline.predict(lowers)
        probs  = nlp_pipeline.predict_proba(lowers)
        for desc, pred, prob in zip(descs, preds, probs):
            conf = float(prob.max())
            top3_idx = prob.argsort()[-3:][::-1]
            top3 = [{"group": nlp_pipeline.classes_[j], "pct": round(float(prob[j])*100,1)} for j in top3_idx]
            results.append({
                "description": desc,
                "predicted":   pred,
                "group_name":  MATERIAL_GROUPS.get(pred, pred),
                "confidence":  round(conf*100, 1),
                "decision":    "AUTO APPLY" if conf >= 0.85 else "REVIEW",
                "top3":        top3,
            })
    else:
        for desc in descs:
            grp, conf = keyword_classify(desc)
            results.append({
                "description": desc,
                "predicted":   grp,
                "group_name":  MATERIAL_GROUPS.get(grp, grp),
                "confidence":  round(conf*100, 1),
                "decision":    "AUTO APPLY" if conf >= 0.85 else "REVIEW",
                "top3":        [{"group": grp, "pct": round(conf*100,1)}],
                "note":        "keyword-fallback (upload nlp_model_final.pkl for full ML)"
            })

    return jsonify({"results": results, "model": "TF-IDF+LogReg" if nlp_pipeline else "keyword-fallback"})


# ── API: health check ─────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":    "ok",
        "model":     GROQ_MODEL,
        "ml_loaded": nlp_pipeline is not None,
        "groq_key":  "configured" if GROQ_API_KEY else "missing — set GROQ_API_KEY env var",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
