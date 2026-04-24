from __future__ import annotations
import re
from typing import Dict

# ── Hallucination guard ───────────────────────────────────────────────────────

_SUSPICIOUS_PATTERNS = re.compile(
    r"\d+\s*(?:°c|%|brix|ppm|n\b|days?)",
    re.IGNORECASE,
)

def contains_suspicious_numbers(text: str) -> bool:
    """Return True if the text contains hallucinated numeric claims."""
    return bool(_SUSPICIOUS_PATTERNS.search(text))

# ── Response scorer ───────────────────────────────────────────────────────────

_USEFUL_WORDS = frozenset({
    "harvest", "monitor", "inspect", "remove",
    "store", "treat", "observe", "separate", "clean",
})

def score_response(response: str, state: Dict) -> int:
    """
    Score a model response against the current prediction state.

    Points breakdown:
      +3  maturity label present
      +3  disease label present
      +1  at least one actionable word
      +2  confidence/uncertainty acknowledged
      +2  response is substantive (≥ 30 words)
      -3  suspicious hallucinated numbers detected
      -1  excessive repetition of "guava" (> 5 times)
    """
    text = response.lower()
    score = 0

    # Prediction alignment
    if state.get("maturity", "").lower() in text:
        score += 3
    if state.get("disease", "").lower() in text:
        score += 3

    # Actionability — reward presence of any useful word
    if _USEFUL_WORDS & set(text.split()):
        score += 1

    # Confidence awareness
    if "confidence" in text or "uncertain" in text:
        score += 2

    # Minimum substance check
    if len(response.split()) >= 30:
        score += 2

    # Penalties
    if contains_suspicious_numbers(text):
        score -= 3
    if text.count("guava") > 5:
        score -= 1

    return score

# ── Model arbitrator ──────────────────────────────────────────────────────────

def judge_outputs(llama_response: str, qwen_response: str, state: Dict) -> str:
    """Return whichever response scores higher; prefer Llama on a tie."""
    llama_score = score_response(llama_response, state)
    qwen_score  = score_response(qwen_response,  state)
    return llama_response if llama_score >= qwen_score else qwen_response