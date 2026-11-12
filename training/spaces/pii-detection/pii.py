"""pii.py — server-side CPU PII detection with LiquidAI/LFM2.5-Encoder-350M-PII-Detector.

A bidirectional LFM2.5 phase-2 encoder (354M) with a token-classification head,
loaded from the Hub via its custom modeling code (trust_remote_code), plus the
model's own inference-time **hybrid regex decode** (pii_hybrid_decode.py +
context_cued.py, fetched from the same repo) — the decode the model is meant to
use. It covers 40 PII types across 11 domains and 16 languages.

f32 on CPU (not fp16: x86 has no fast half path; not int8: token-classification
boundaries are precision-sensitive) — the same reasoning as the spellchecker /
policy-linting Spaces. One forward per request. The model repo is public, so
HF_TOKEN is optional.
"""

import os
import sys
import threading

from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = os.environ.get("PII_MODEL", "LiquidAI/LFM2.5-Encoder-350M-PII-Detector")
HF_TOKEN = os.environ.get("HF_TOKEN")


def effective_cpus():
    """Cores actually available to this process (cgroup quota, not the host)."""
    try:  # cgroup v2
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw and raw[0] != "max":
            return max(1, round(int(raw[0]) / int(raw[1])))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


# --- label -> {domain (color), pretty label, redaction tag} ------------------
_ACRONYM = {"ssn", "iban", "id", "ip", "url", "api", "mac", "gps", "bic", "dob",
            "pin", "vat", "cvv", "swift", "tax", "jwt", "imei"}
# Types are `domain.subtype` (label_schema.json), so the color bucket is just the
# prefix. `developer.*` is the exception: login_credentials -> credential, device_id
# -> device.
_DOMAINS = {"identity", "contact", "financial", "credential", "online", "device",
            "location", "healthcare", "org", "special", "legal"}


def domain_for(raw):
    pre = raw.split(".")[0].lower()
    if pre in _DOMAINS:
        return pre
    if pre == "developer":
        return "device" if "device" in raw.lower() else "credential"
    return "other"


def pretty(raw):
    base = raw.split(".")[-1]
    words = base.replace("-", " ").replace("_", " ").split()
    out = [w.upper() if w.lower() in _ACRONYM else w.capitalize() for w in words]
    return " ".join(out) or raw


def tag_for(raw):
    base = raw.split(".")[-1]
    return "_".join(base.replace("-", " ").replace("_", " ").split()).upper() or "PII"


class PiiDetector:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, trust_remote_code=True, token=HF_TOKEN).float().eval()
        # the model's shipped hybrid decode (+ its context_cued dependency); load
        # both from the repo snapshot and import off that dir.
        hd_path = hf_hub_download(MODEL_ID, "pii_hybrid_decode.py", token=HF_TOKEN)
        hf_hub_download(MODEL_ID, "context_cued.py", token=HF_TOKEN)
        sys.path.insert(0, os.path.dirname(hd_path))
        import pii_hybrid_decode as hd
        self.hd = hd

    def detect(self, text):
        """Return {spans, redacted, count}. spans: char-anchored, enriched, sorted,
        non-overlapping. When a model span and a regex span cover the same text, the
        regex (AUTH) type wins — it's the precise format (e.g. api_key/jwt over the
        model's generic login_credentials)."""
        with self.lock:
            raw = self.hd.predict(text, self.tok, self.model)   # [{start,end,type,text}]

        auth = getattr(self.hd, "_AUTH_TYPES", set())
        raw.sort(key=lambda s: (s["start"], 0 if s["type"] in auth else 1, -(s["end"] - s["start"])))
        spans, last_end = [], -1
        for sp in raw:
            if sp["start"] < last_end or sp["end"] <= sp["start"]:
                continue                                        # drop overlaps
            typ = sp["type"]
            spans.append({
                "start": sp["start"], "end": sp["end"], "text": text[sp["start"]:sp["end"]],
                "type": typ, "domain": domain_for(typ), "label": pretty(typ), "tag": tag_for(typ),
            })
            last_end = sp["end"]

        parts, prev = [], 0
        for sp in spans:
            parts.append(text[prev:sp["start"]])
            parts.append(f"[{sp['tag']}]")
            prev = sp["end"]
        parts.append(text[prev:])
        return {"spans": spans, "redacted": "".join(parts), "count": len(spans)}
