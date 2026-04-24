from __future__ import annotations
from typing import Dict

# ── Advisory templates ────────────────────────────────────────────────────────
# Keyed by (disease, maturity); disease "healthy" uses maturity only.
# Diseases : anthracnose | scab | styler-end-rot | healthy
# Maturity : immature | semi-mature | mature

_TEMPLATES: dict[tuple[str, str], str] = {
    # Healthy
    ("healthy", "immature"): (
        "The guava appears healthy but is not yet ready for harvest. "
        "Continue monitoring colour development and recheck in a few days. "
        "Maintain regular irrigation and orchard hygiene."
    ),
    ("healthy", "semi-mature"): (
        "The guava is healthy and approaching harvest readiness. "
        "Monitor skin colour, aroma, and firmness over the coming days. "
        "Handle carefully to avoid surface damage."
    ),
    ("healthy", "mature"): (
        "The guava is healthy and ready to harvest. "
        "Pick gently to prevent bruising, then sort and store in cool, clean conditions. "
        "Market or consume promptly for best quality."
    ),
    # Anthracnose
    ("anthracnose", "immature"): (
        "Anthracnose detected on an immature fruit. "
        "Isolate affected fruits and remove severely infected ones. "
        "Keep the orchard dry and clean; monitor nearby fruits regularly."
    ),
    ("anthracnose", "semi-mature"): (
        "Anthracnose detected on a semi-mature fruit. "
        "Separate affected fruits, improve air circulation, and inspect neighbours for spreading symptoms. "
        "Prioritise healthy fruits in harvest planning."
    ),
    ("anthracnose", "mature"): (
        "Anthracnose detected on a mature fruit. "
        "Sort and harvest healthy fruits separately. "
        "Do not mix diseased fruits during storage; sanitise crates and tools after handling."
    ),
    # Scab
    ("scab", "immature"): (
        "Scab detected on an immature fruit. "
        "Monitor surface damage, remove heavily affected fruits if needed, "
        "and reduce branch overcrowding to improve airflow."
    ),
    ("scab", "semi-mature"): (
        "Scab detected on a semi-mature fruit. "
        "Separate affected fruits, inspect neighbours for similar lesions, "
        "and improve sunlight exposure and airflow."
    ),
    ("scab", "mature"): (
        "Scab detected on a mature fruit. "
        "Sort by market quality, separate damaged fruits from the healthy harvest, "
        "and maintain post-harvest hygiene during packaging."
    ),
    # Styler-End-Rot
    ("styler-end-rot", "immature"): (
        "Styler-End-Rot detected on an immature fruit. "
        "Remove affected fruits where necessary, monitor orchard moisture balance, "
        "and keep the field clean."
    ),
    ("styler-end-rot", "semi-mature"): (
        "Styler-End-Rot detected on a semi-mature fruit. "
        "Separate symptomatic fruits, improve drainage and sanitation, "
        "and reassess harvest quality before picking."
    ),
    ("styler-end-rot", "mature"): (
        "Styler-End-Rot detected on a mature fruit. "
        "Harvest healthy fruits separately and avoid storing damaged fruits with healthy stock. "
        "Sanitise storage trays and handling surfaces."
    ),
}

# Canonical aliases so variant spellings resolve to the same key
_DISEASE_ALIASES: dict[str, str] = {
    "styler end rot":  "styler-end-rot",
    "styler_end_rot":  "styler-end-rot",
}

_LOW_CONF_THRESHOLD = 0.50


def generate_template_response(state: Dict) -> str:
    """
    Return a rule-based advisory string when both LLM models are unavailable.
    Covers all combinations of maturity × disease, plus a low-confidence guard
    and a final catch-all fallback.
    """
    maturity = state.get("maturity", "Unknown").strip()
    disease  = state.get("disease",  "Unknown").strip()
    m_conf   = state.get("m_conf", 0.0)
    d_conf   = state.get("d_conf", 0.0)

    # ── Low-confidence guard ──────────────────────────────────────────────────
    if m_conf < _LOW_CONF_THRESHOLD or d_conf < _LOW_CONF_THRESHOLD:
        return (
            f"Prediction confidence is low (maturity: {m_conf:.0%}, disease: {d_conf:.0%}). "
            "Upload a clearer image with proper lighting and a single fruit in frame. "
            "Inspect the fruit manually for colour, firmness, and lesions, then re-run the prediction."
        )

    # ── Normalise keys ────────────────────────────────────────────────────────
    disease_key  = _DISEASE_ALIASES.get(disease.lower(), disease.lower())
    maturity_key = maturity.lower()

    # ── Template lookup ───────────────────────────────────────────────────────
    advisory = _TEMPLATES.get((disease_key, maturity_key))
    if advisory:
        return advisory

    # ── Catch-all fallback ────────────────────────────────────────────────────
    return (
        f"Analysis complete — maturity: {maturity}, disease: {disease}. "
        "Inspect the fruit manually, separate any visibly unhealthy fruits, "
        "and re-run the prediction if results seem unclear."
    )