from __future__ import annotations

import requests
from typing import TypedDict

# ═══════════════════════════════════════════════════════════════════════════════
# models.py
#
# Local LLM interface layer for the GuavaXAI advisory agent.
# Supports two offline models served via Ollama:
#   • Llama  (llama3)
#   • Qwen   (qwen2.5:7b)
#
# Requirements:
#   ollama pull llama3
#   ollama pull qwen2.5:7b
#   ollama serve
# ═══════════════════════════════════════════════════════════════════════════════


# ==========================================================
# STATE DEFINITION
# ==========================================================
class AdvisoryState(TypedDict):
    maturity: str
    disease: str
    m_conf: float
    d_conf: float
    mode: str
    confidence_ok: bool
    severity: str
    harvest_window: str
    triage_notes: str
    raw_llm_response: str
    response: str
    used_llm: bool


# ==========================================================
# DOMAIN CONSTANTS
# ==========================================================
CONFIDENCE_THRESHOLD = 0.50

HARVEST_WINDOWS = {
    "Immature": "Not ready for harvest",
    "Semi-Mature": "Approaching harvest stage",
    "Mature": "Ready for harvest",
}

DISEASE_SEVERITY = {
    "Healthy": "none",
    "Anthracnose": "high",
    "Scab": "moderate",
    "Styler-End-Rot": "high",
}

DISEASE_DESCRIPTIONS = {
    "Healthy": "No visible disease symptoms detected.",
    "Anthracnose": "Fungal disease causing dark sunken lesions.",
    "Scab": "Surface scab lesions affecting fruit appearance.",
    "Styler-End-Rot": "Decay symptoms visible at the styler end of fruit.",
}


# ==========================================================
# OLLAMA CONFIG
# ==========================================================
_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_TAGS = "http://localhost:11434/api/tags"

_TIMEOUT = 120
_TEMPERATURE = 0.7
_TOP_P = 0.9

_LLAMA_MODEL = "llama3"
_QWEN_MODEL = "qwen2.5:7b"


# ==========================================================
# AVAILABILITY CHECK
# ==========================================================
def _ollama_available() -> bool:
    try:
        response = requests.get(_OLLAMA_TAGS, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


_llama_available = _ollama_available()


# ==========================================================
# SHARED MODEL CALLER
# ==========================================================
def _call_ollama(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000
) -> str:

    prompt = f"""
System:
{system_prompt}

User:
{user_prompt}
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
            "num_predict": max_tokens
        }
    }

    response = requests.post(
        _OLLAMA_URL,
        json=payload,
        timeout=_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()
    return result["response"].strip()


# ==========================================================
# PUBLIC MODEL FUNCTIONS
# ==========================================================
def _call_llama(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000
) -> str:
    return _call_ollama(
        _LLAMA_MODEL,
        system_prompt,
        user_prompt,
        max_tokens
    )


def _call_qwen(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000
) -> str:
    return _call_ollama(
        _QWEN_MODEL,
        system_prompt,
        user_prompt,
        max_tokens
    )