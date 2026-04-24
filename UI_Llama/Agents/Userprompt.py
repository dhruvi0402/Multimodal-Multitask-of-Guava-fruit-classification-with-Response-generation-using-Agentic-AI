# ═══════════════════════════════════════════════════════════════════════════════
# Userprompt.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# Description:
#   This file contains all user-facing prompt components used by the LangGraph
#   advisory agent for GuavaXAI.  It includes:
#
#   1. FEW-SHOT EXAMPLES — Gold-standard output examples embedded in prompts to
#      show the Llama LLM exactly what "specific, farm-actionable" advice looks
#      like.  There is one example per advisory mode:
#        • _FEWSHOT_HARVEST     — Example for healthy-fruit harvest advisory
#        • _FEWSHOT_TREATMENT   — Example for disease treatment advisory
#        • _FEWSHOT_UNCERTAINTY — Example for low-confidence uncertainty advisory
#
#   2. CONFIDENCE TIER HELPER — `_conf_tier(conf)` maps a numeric confidence
#      score to a human-readable tier label (HIGH / MODERATE / LOW) that is
#      injected into every user prompt so the LLM can calibrate its hedging.
#
#   3. PROMPT BUILDER FUNCTIONS — One function per advisory mode that assembles
#      the full user prompt from classification results, few-shot example, and
#      task instructions:
#        • build_harvest_prompt(mat, hw, m_tier, d_tier)
#        • build_treatment_prompt(mat, dis, sev, hw, notes, m_tier, d_tier)
#        • build_uncertainty_prompt(mat, dis, m_tier, d_tier)
#
# Usage:
#   from Userprompt import (
#       _conf_tier,
#       build_harvest_prompt,
#       build_treatment_prompt,
#       build_uncertainty_prompt,
#   )
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE TIER THRESHOLDS
# ═══════════════════════════════════════════════════════════════
_HIGH_CONF   = 0.80
_MEDIUM_CONF = 0.60


# ═══════════════════════════════════════════════════════════════
# HELPER — confidence tier label for prompt context
# ═══════════════════════════════════════════════════════════════
def _conf_tier(conf: float) -> str:
    """
    Convert a numeric confidence score (0–1) into a descriptive tier string
    that tells the LLM how hedged or direct its language should be.
    """
    if conf >= _HIGH_CONF:
        return f"HIGH ({conf*100:.1f}%) — be direct and unhedged."
    elif conf >= _MEDIUM_CONF:
        return f"MODERATE ({conf*100:.1f}%) — include one brief note suggesting visual confirmation."
    else:
        return f"LOW ({conf*100:.1f}%) — frame advice cautiously throughout."


# ═══════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES — show the model what "specific" looks like
# ═══════════════════════════════════════════════════════════════

_FEWSHOT_HARVEST = """
## Example of the output quality and specificity expected
(This example is for Semi-Mature / Healthy — do NOT copy it; produce the equivalent
quality for the actual stage given above.)

HARVEST READINESS
Do not harvest today. Semi-Mature guava flesh is still starchy (Brix 6–8°) and the
skin has not begun its colour break from dark green to yellow-green. Harvesting now
cuts soluble-solid accumulation short and reduces final shelf-life by roughly 30–40%.

QUALITY INDICATORS
1. Skin colour: dark green with no yellow blush on the shoulder — confirms Semi-Mature.
2. Firmness: fruit resists at >12 N when pressed at the equator; thumbnail pressure
   leaves no dent.
3. Aroma: hold the stem end to your nose — no detectable guava fragrance yet.
4. Size: fruit has reached ~75% of its expected final diameter for the variety.

HARVEST TECHNIQUE
At Semi-Mature stage do not harvest; instead flag fruit with coloured tape for
monitoring. When the 2–4 day window arrives, use a clean, sharp stainless-steel
snipping shear to cut the peduncle 1–2 cm above the calyx. Cradle each fruit — never
drop into a collection bin. Work in the early morning (below 28°C) to minimise
transpiration stress.

POST-HARVEST HANDLING
Semi-Mature guava held after the colour break: store at 8–10°C, 90–95% RH in single-
layer ventilated corrugated fibreboard trays lined with perforated polyethylene film
(30–40 µm). Expected shelf-life at this temperature: 14–18 days for table-grade fruit.

NEXT CYCLE TIPS
Apply a split potassium fertilisation — 60% of the seasonal K₂O dose at fruit set and
40% at the Semi-Mature stage — to lift final Brix by 1–2° and improve peel colour
uniformity.
"""

_FEWSHOT_TREATMENT = """
## Example of the output quality and specificity expected
(This example is for Immature / Anthracnose — do NOT copy it; produce the equivalent
quality for the actual combination given above.)

HARVEST READINESS
Do not harvest. Immature fruit infected with Anthracnose cannot ripen normally; the
fungus (Colletotrichum gloeosporioides) will continue colonising through the quiescent
phase and cause complete post-harvest breakdown within 3–5 days of any attempted
ripening. Remove and destroy all symptomatic fruit instead.

DISEASE EXPLANATION
Anthracnose on guava presents as small, water-soaked circular spots (2–5 mm) that
enlarge into sunken, dark-brown to black lesions with salmon-pink spore masses at the
centre in humid conditions. On Immature fruit the lesions remain latent (no visible
sign) until ripening begins — this makes early AI detection especially important.

CAUSE & SPREAD
Primary pathogen: Colletotrichum gloeosporioides (asexual stage; teleomorph
Glomerella cingulata). Spread accelerates under two conditions: (1) sustained leaf
wetness of ≥6 hours at 25–30°C, and (2) mechanical wounds from overhead irrigation
droplets or hail that create infection courts on the immature skin.

IMMEDIATE ACTION
1. Remove all visibly symptomatic fruit from the tree and seal in black polythene bags
   before disposal — do not compost.
2. Sterilise pruning tools with 70% ethanol between every tree.
3. Apply a curative copper oxychloride spray (see Treatment Protocol) within 24 hours
   across all immature fruit on affected trees.
4. Suspend overhead irrigation for 5 days and switch to drip if possible.

TREATMENT PROTOCOL
Curative spray: Copper oxychloride 50% WP at 3 g/L water; spray to run-off, covering
fruit surface and calyx end. Repeat every 10 days for 3 applications.
Systemic follow-up: Carbendazim 50% WP at 1 g/L or Thiophanate-methyl 70% WP at
1.5 g/L, alternated with the copper spray to reduce resistance risk. Do not apply
within 14 days of anticipated harvest on any nearby mature fruit.

PREVENTION & MONITORING
1. Canopy management: prune to maintain 30–40% light penetration into the canopy
   interior, targeting a branch density that allows leaf surfaces to dry within 2 hours
   of rain or irrigation.
2. Orchard floor hygiene: collect and destroy all fallen fruit weekly throughout the
   season; Colletotrichum overwinters in mummified fruit and re-inoculates the crop
   at flower-set each cycle.
"""

_FEWSHOT_UNCERTAINTY = """
## Example of the output quality and specificity expected
(This example is for Semi-Mature / Scab at low confidence — do NOT copy it; produce
the equivalent quality for the actual combination given above.)

UNCERTAINTY ADVISORY
The model's confidence for both maturity and disease is below the reliable threshold,
meaning the predicted Semi-Mature stage and Scab diagnosis carry a meaningful error
rate. Acting on a misdiagnosis here carries two specific risks: premature harvest of
fruit that needs more time, or applying a sulphur-based fungicide to fruit that may
not need it, which can cause phytotoxic russeting on young guava skin.

VISUAL INSPECTION GUIDE
1. Maturity check — skin colour: Semi-Mature guava should show the first faint
   yellow-green blush on the shoulder; if the whole fruit is uniformly dark green,
   it is likely still Immature.
2. Maturity check — firmness: press the equator firmly with your thumb; Semi-Mature
   fruit resists without yielding. Immature fruit feels almost woody.
3. Maturity check — size: Semi-Mature fruit should be at 70–80% of the final expected
   diameter for your variety.
4. Disease check — Scab lesions: look for raised, rough, corky or warty outgrowths on
   the fruit surface, typically 2–6 mm across with a greyish-brown centre. Scab does
   not produce sunken lesions — if you see sunken dark spots, suspect Anthracnose.
5. Disease check — lesion pattern: Scab lesions cluster near the calyx end and on the
   shoulders; random distribution across the whole fruit surface is more consistent
   with Anthracnose or physical damage.

RECOMMENDED NEXT STEPS
1. Re-capture the image in open shade (not direct noon sun) with the fruit held against
   a plain white background at 20–30 cm from the camera lens — this resolves the
   majority of low-confidence predictions caused by harsh shadow or motion blur.
2. Perform the five-point visual inspection above and note your findings before taking
   any spray or harvest action.
3. If lesions are confirmed after inspection, consult your local agricultural extension
   officer or a certified crop protection adviser before applying any fungicide, as
   misidentification between Scab and Anthracnose requires entirely different active
   ingredients.
"""


# ═══════════════════════════════════════════════════════════════
# PROMPT BUILDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_harvest_prompt(mat: str, hw: str, m_tier: str, d_tier: str) -> str:
    """
    Build the full user prompt for the harvest advisory node.

    Parameters
    ----------
    mat    : Maturity stage label (e.g. "Mature", "Semi-Mature", "Immature")
    hw     : Human-readable harvest window string
    m_tier : Confidence tier string for maturity (from _conf_tier)
    d_tier : Confidence tier string for disease  (from _conf_tier)

    Returns
    -------
    str — The assembled prompt ready to send to Llama.
    """
    return f"""## Classification Results
- Maturity Stage          : {mat}
  Confidence              : {m_tier}
- Disease Status          : Healthy — no disease detected
  Confidence              : {d_tier}
- Estimated harvest window: {hw}

{_FEWSHOT_HARVEST}

## Your Task
Now write the advisory for the ACTUAL case above: maturity = "{mat}", disease = Healthy.
Use EXACTLY the same five-section structure shown in the example.
Plain text only — no markdown headings, no bullet symbols.
Separate sections with one blank line.
Every sentence must be specific to "{mat}" guava — numbers, thresholds, and actions
must differ from the example where the stages differ.

1. HARVEST READINESS
2. QUALITY INDICATORS
3. HARVEST TECHNIQUE
4. POST-HARVEST HANDLING
5. NEXT CYCLE TIPS
"""


def build_treatment_prompt(
    mat: str, dis: str, sev: str, hw: str, notes: str, m_tier: str, d_tier: str
) -> str:
    """
    Build the full user prompt for the disease treatment advisory node.

    Parameters
    ----------
    mat    : Maturity stage label
    dis    : Disease label (e.g. "Anthracnose", "Scab", "Styler Rot")
    sev    : Severity level string (e.g. "high", "moderate", "low")
    hw     : Human-readable harvest window string
    notes  : Triage notes assembled by the triage node
    m_tier : Confidence tier string for maturity
    d_tier : Confidence tier string for disease

    Returns
    -------
    str — The assembled prompt ready to send to Llama.
    """
    return f"""## Classification Results
- Maturity Stage          : {mat}
  Confidence              : {m_tier}
- Disease Detected        : {dis}
  Confidence              : {d_tier}
- Disease severity        : {sev}
- Estimated harvest window: {hw}

## Triage Context
{notes}

{_FEWSHOT_TREATMENT}

## Your Task
Now write the advisory for the ACTUAL case above: maturity = "{mat}", disease = "{dis}".
Use EXACTLY the same six-section structure shown in the example.
Plain text only — no markdown headings, no bullet symbols.
Separate sections with one blank line.
Every sentence must be specific to "{dis}" on "{mat}" guava.

1. HARVEST READINESS
2. DISEASE EXPLANATION
3. CAUSE & SPREAD
4. IMMEDIATE ACTION
5. TREATMENT PROTOCOL
6. PREVENTION & MONITORING
"""


def build_uncertainty_prompt(mat: str, dis: str, m_tier: str, d_tier: str) -> str:
    """
    Build the full user prompt for the low-confidence uncertainty advisory node.

    Parameters
    ----------
    mat    : Predicted maturity stage label
    dis    : Predicted disease label
    m_tier : Confidence tier string for maturity
    d_tier : Confidence tier string for disease

    Returns
    -------
    str — The assembled prompt ready to send to Llama.
    """
    return f"""## Low-Confidence Predictions
- Predicted Maturity: {mat}
  Confidence        : {m_tier}
- Predicted Disease : {dis}
  Confidence        : {d_tier}

{_FEWSHOT_UNCERTAINTY}

## Your Task
Now write the cautious advisory for the ACTUAL case above:
maturity = "{mat}", disease = "{dis}".
Use EXACTLY the same three-section structure shown in the example.
Plain text only — no markdown headings, no bullet symbols.
Separate sections with one blank line.

1. UNCERTAINTY ADVISORY
2. VISUAL INSPECTION GUIDE
3. RECOMMENDED NEXT STEPS
"""
