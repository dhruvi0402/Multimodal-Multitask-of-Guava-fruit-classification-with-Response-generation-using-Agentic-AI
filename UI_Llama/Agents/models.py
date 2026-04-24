# ═══════════════════════════════════════════════════════════════════════════════
# models.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# Description:
#   This file contains the data models, domain constants, and the Llama LLM
#   interface layer used by the LangGraph advisory agent for GuavaXAI.
#
#   Contents:
#
#   1. STATE DEFINITION — `AdvisoryState` (TypedDict) defines the shared state
#      that flows through every node in the LangGraph advisory graph.  Fields
#      include classification inputs (maturity, disease, confidences), routing
#      flags, intermediate triage notes, and the final formatted response.
#
#   2. DOMAIN CONSTANTS — Agronomic lookup tables that parameterise the agent's
#      deterministic logic:
#        • CONFIDENCE_THRESHOLD — minimum confidence to trust a prediction
#        • HARVEST_WINDOWS      — maturity → estimated days-to-harvest
#        • DISEASE_SEVERITY     — disease  → severity tier
#        • DISEASE_DESCRIPTIONS — disease  → short pathology description
#        • RULE_BASED_TEMPLATES — (maturity, disease) → fallback advice string
#
#   3. LLAMA MODEL INTERFACE — Ollama-based local LLM interface.
#      The Ollama server is expected to be running at localhost:11434.
#        • LLAMA_MODEL_NAME             — model identifier string
#        • OLLAMA_BASE_URL              — Ollama server endpoint
#        • _llama_available             — whether Ollama is reachable
#        • _call_llama(system, user)    — single-call helper that returns text
#
# Usage:
#   from models import (
#       AdvisoryState,
#       CONFIDENCE_THRESHOLD, HARVEST_WINDOWS, DISEASE_SEVERITY,
#       DISEASE_DESCRIPTIONS, RULE_BASED_TEMPLATES,
#       _call_llama, _llama_available,
#   )
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import TypedDict


# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════
class AdvisoryState(TypedDict):
    """
    Shared state dictionary that is passed through every node in the
    LangGraph advisory graph.

    Fields
    ------
    maturity         : Predicted maturity stage (e.g. "Mature")
    disease          : Predicted disease label  (e.g. "Anthracnose")
    m_conf           : Maturity prediction confidence [0, 1]
    d_conf           : Disease prediction confidence  [0, 1]
    mode             : Routing mode set by router nodes ("harvest" / "treatment" / "uncertain")
    confidence_ok    : Whether both confidences exceed CONFIDENCE_THRESHOLD
    severity         : Disease severity tier ("none" / "low" / "moderate" / "high")
    harvest_window   : Human-readable harvest window string
    triage_notes     : Free-text triage context assembled by triage_node
    raw_llm_response : Raw text returned by the Llama LLM (before formatting)
    response         : Final formatted advisory text
    used_llm         : Whether the LLM was successfully called for this request
    """
    maturity:         str
    disease:          str
    m_conf:           float
    d_conf:           float
    mode:             str
    confidence_ok:    bool
    severity:         str
    harvest_window:   str
    triage_notes:     str
    raw_llm_response: str
    response:         str
    used_llm:         bool


# ═══════════════════════════════════════════════════════════════
# DOMAIN CONSTANTS
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.50

HARVEST_WINDOWS = {
    "Immature":    "7–10 days before harvest-ready",
    "Semi-Mature": "2–4 days to optimal harvest",
    "Mature":      "Ready for immediate harvest",
}

DISEASE_SEVERITY = {
    "Healthy":    "none",
    "Scab":       "low",
    "Anthracnose":"high",
    "Styler Rot": "moderate",
}

DISEASE_DESCRIPTIONS = {
    "Healthy":     "No disease detected. Fruit is healthy.",
    "Scab":        "Scab causes rough, corky lesions on the fruit surface.",
    "Anthracnose": "Anthracnose is a fungal disease causing dark, sunken lesions. "
                   "It spreads rapidly in warm, humid conditions.",
    "Styler Rot":  "Styler Rot (stylar-end rot) is caused by calcium deficiency "
                   "or fungal infection at the blossom end of the fruit.",
}

RULE_BASED_TEMPLATES = {
    ("Mature",      "Healthy"):     "✅ Your fruit has reached optimal maturity and looks healthy — this is exactly the window you want. Harvest within the next 24–48 hours and move it to cool, dry storage (10–15°C) to preserve quality.",

    ("Semi-Mature", "Healthy"):     "🟡 Looking good — the fruit is developing well with no disease signs. I'd recommend waiting another 2–4 days before harvesting. Keep irrigation consistent and check back daily as it approaches peak ripeness.",

    ("Immature",    "Healthy"):     "🌱 The fruit is healthy but still needs time. Give it 7–10 more days and ensure the crop is getting adequate water and nutrition. Avoid any stress to the plant during this critical growth window.",

    ("Mature",      "Anthracnose"): "⚠️ Anthracnose has been detected on mature fruit — act today. Harvest immediately and keep affected fruit completely separate from healthy stock. Apply an approved post-harvest fungicide (e.g., Prochloraz) and do not store these together under any circumstances.",

    ("Semi-Mature", "Anthracnose"): "⚠️ This is a concern — Anthracnose on semi-mature fruit can spread quickly if conditions stay humid. Apply a copper-based fungicide within 24 hours and prune dense canopy areas to improve airflow. Avoid overhead irrigation until the situation is under control.",

    ("Immature",    "Anthracnose"): "🚨 Immediate action required. Anthracnose on immature fruit rarely recovers — remove and destroy all visibly infected fruit now to prevent spore spread. Follow up with a systemic fungicide across the entire tree and scout surrounding plants for early signs.",

    ("Mature",      "Scab"):        "🟠 Scab at this stage is mainly a cosmetic issue — the fruit is still safe to harvest. Handle carefully during picking and grading to minimise surface damage. Affected fruit may be better suited for processing markets. Clear fallen debris post-harvest to reduce next season's risk.",

    ("Semi-Mature", "Scab"):        "🟠 There's still time to reduce the impact. Apply a sulfur-based fungicide (e.g., Wettable Sulphur) within 48 hours and switch to drip irrigation if possible — wet foliage is what drives Scab spread. Reassess in 5–7 days.",

    ("Immature",    "Scab"):        "🟠 Good that we're catching this early. Prune affected shoots or clusters to reduce fungal load, then apply a protectant fungicide to cover new growth. Keep a close eye on humidity — Scab thrives when leaf surfaces stay wet for extended periods.",

    ("Mature",      "Styler Rot"):  "⚠️ Styler Rot (Blossom End Rot) is a calcium deficiency issue, not an infectious disease, so it won't spread — but affected fruit deteriorates fast. Harvest immediately and discard symptomatic fruit. Get a soil test done to check calcium availability and review your irrigation consistency.",

    ("Semi-Mature", "Styler Rot"):  "⚠️ You can still limit the damage here. Apply foliar calcium spray (Calcium Chloride 0.5%) without delay and switch to shorter, more frequent irrigation cycles — erratic moisture is the main driver of calcium uptake failure. Avoid high-nitrogen fertilizers during this period.",

    ("Immature",    "Styler Rot"):  "⚠️ Catching this early gives you the best chance of a clean harvest. Address the calcium deficiency now with a soil amendment based on test results, and stabilise your irrigation schedule. Consistent soil moisture is the single most important factor — the fruit still has time to recover.",
}


# ═══════════════════════════════════════════════════════════════
# LLAMA MODEL INTERFACE (via Ollama)
# ═══════════════════════════════════════════════════════════════

# ── LLAMA MODEL NAME ──────────────────────────────────────────
# Change this to any model available in your Ollama installation,
# e.g. "llama3.1:8b", "llama3.2:3b", "mistral", etc.
LLAMA_MODEL_NAME = "llama3.2:3b"

# ── OLLAMA SERVER ENDPOINT ────────────────────────────────────
# Default Ollama local server address. Override with OLLAMA_BASE_URL env var.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _check_ollama_available() -> bool:
    """
    Check if the Ollama server is reachable by hitting the /api/tags endpoint.
    Returns True if the server responds, False otherwise.
    """
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# Check availability once at import time
_llama_available = _check_ollama_available()


def _call_llama(system_prompt: str, user_prompt: str, max_output_tokens: int = 1000) -> str:
    """
    Single helper that calls Llama via Ollama and returns the response text.

    Parameters
    ----------
    system_prompt    : The system-level persona prompt
    user_prompt      : The full user-facing prompt text
    max_output_tokens: Maximum tokens in the generated response

    Returns
    -------
    str — Stripped response text from Llama.

    Raises
    ------
    Any network or Ollama error — callers are expected to handle exceptions.
    """
    payload = json.dumps({
        "model": LLAMA_MODEL_NAME,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": max_output_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body.get("response", "").strip()
