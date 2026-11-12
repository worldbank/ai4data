"""PII detection — server-side CPU inference.

LiquidAI/LFM2.5-Encoder-350M-PII-Detector (LFM2.5 encoder + token-classification head + hybrid regex
decode) runs on the Space's CPU. The browser POSTs {text} to /api/detect and
gets back char-anchored PII spans (type/domain/label) + a redacted string, then
highlights them. The model repo is public, so no token is needed to pull it.
"""

import os
import time

import torch
from flask import Flask, request, send_from_directory

import pii as P

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Match threads to the container's cgroup CPU quota, not the host core count.
_CPUS = P.effective_cpus()
torch.set_num_threads(_CPUS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

app = Flask(__name__)

print(f"[boot] loading the PII detector on {_CPUS} CPU thread(s), f32…", flush=True)
MODEL = P.PiiDetector()
MODEL.detect("Warm up: email jane@example.com, SSN 123-45-6789.")   # warm the graph
print("[boot] detector ready", flush=True)


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/<path:name>")
def static_file(name):
    if name in ("style.css",):
        return send_from_directory(APP_DIR, name)
    return ("not found", 404)


@app.route("/api/ready")
def ready():
    return {"ready": MODEL is not None}


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").replace("\r", "")
    if not text.strip():
        return {"spans": [], "redacted": "", "count": 0, "ms": 0}
    t0 = time.perf_counter()
    out = MODEL.detect(text)
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), threaded=True)
