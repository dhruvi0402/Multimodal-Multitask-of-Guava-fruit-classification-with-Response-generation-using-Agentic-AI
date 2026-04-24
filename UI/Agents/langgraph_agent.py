"""
Agents/langgraph_agent.py
─────────────────────────
Advanced LangGraph-based advisory agent for GuavaXAI.

Graph topology:
    [input_validator]
         │
    [confidence_router] ──low_conf──▶ [uncertainty_node] ──▶ [formatter]
         │                                                          ▲
      high_conf                                                     │
         │                                                          │
    [decision_router]  ──healthy──▶ [harvest_node] ────────────────┤
         │                                                          │
      diseased                                                      │
         │                                                          │
    [triage_node] ──▶ [treatment_node] ─────────────────────────── ┘

Public API:
    generate_advisory(maturity, disease, m_conf, d_conf) → (str, bool)
    Returns (formatted_advice, used_llm_flag)

Module dependencies:
    - SystemPrompt.py  → expert persona prompts (_SYSTEM_*)
    - Userprompt.py    → few-shot examples, confidence tiers, prompt builders
    - models.py        → AdvisoryState, domain constants, multi-LLM local interface (Llama + Qwen)
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Literal

from langgraph.graph import StateGraph, END

# ── Ensure the Agents package directory is importable ────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT   = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from Agents.SystemPrompt import _SYSTEM_HARVEST, _SYSTEM_TREATMENT, _SYSTEM_UNCERTAINTY
from Agents.Userprompt import (
    _conf_tier,
    build_harvest_prompt,
    build_treatment_prompt,
    build_uncertainty_prompt,
)
from Agents.models import (
    AdvisoryState,
    CONFIDENCE_THRESHOLD,
    HARVEST_WINDOWS,
    DISEASE_SEVERITY,
    DISEASE_DESCRIPTIONS,
)
from Agents.llm_router import generate_best_response
from Agents.templates import generate_template_response


# ═══════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════

def input_validator(state: AdvisoryState) -> AdvisoryState:
    state["m_conf"]   = max(0.0, min(float(state["m_conf"]), 1.0))
    state["d_conf"]   = max(0.0, min(float(state["d_conf"]), 1.0))
    state["maturity"] = state["maturity"].strip().title()
    state["disease"]  = state["disease"].strip().title()
    state["confidence_ok"]    = (state["m_conf"] >= CONFIDENCE_THRESHOLD and
                                  state["d_conf"] >= CONFIDENCE_THRESHOLD)
    state["severity"]         = DISEASE_SEVERITY.get(state["disease"], "unknown")
    state["harvest_window"]   = HARVEST_WINDOWS.get(state["maturity"], "unknown")
    state["triage_notes"]     = ""
    state["raw_llm_response"] = ""
    state["response"]         = ""
    state["used_llm"]         = False
    state["mode"]             = ""
    return state


def confidence_router_node(state: AdvisoryState) -> AdvisoryState:
    if not state["confidence_ok"]:
        state["mode"] = "uncertain"
    return state


def decision_router_node(state: AdvisoryState) -> AdvisoryState:
    if state["disease"].lower() == "healthy":
        state["mode"] = "harvest"
    else:
        state["mode"] = "treatment"
    return state


def triage_node(state: AdvisoryState) -> AdvisoryState:
    sev = state["severity"]
    mat = state["maturity"]
    dis = state["disease"]

    if sev == "high" and mat == "Mature":
        urgency = "URGENT: harvest immediately and quarantine."
    elif sev == "high":
        urgency = "High severity disease on non-mature fruit — treatment is the top priority."
    elif sev == "moderate":
        urgency = "Moderate severity — monitor closely and apply targeted treatment."
    else:
        urgency = "Low severity — preventive measures are sufficient."

    state["triage_notes"] = (
        f"Disease severity: {sev}. "
        f"Harvest window: {state['harvest_window']}. "
        f"Urgency: {urgency} "
        f"Disease profile: {DISEASE_DESCRIPTIONS.get(dis, '')}"
    )
    return state


# ═══════════════════════════════════════════════════════════════
# LLM NODES
# ═══════════════════════════════════════════════════════════════

def harvest_node(state: AdvisoryState) -> AdvisoryState:
    """Harvest-focused advisory for healthy fruit."""
    prompt = build_harvest_prompt(
        state["maturity"],
        state["harvest_window"],
        _conf_tier(state["m_conf"]),
        _conf_tier(state["d_conf"]),
    )
    try:
        state["raw_llm_response"] = generate_best_response(_SYSTEM_HARVEST, prompt, state)
        state["used_llm"] = True
    except Exception:
        state["raw_llm_response"] = ""
        state["used_llm"] = False
    return state


def treatment_node(state: AdvisoryState) -> AdvisoryState:
    """Disease-treatment advisory with triage context."""
    prompt = build_treatment_prompt(
        state["maturity"],
        state["disease"],
        state["severity"],
        state["harvest_window"],
        state["triage_notes"],
        _conf_tier(state["m_conf"]),
        _conf_tier(state["d_conf"]),
    )
    try:
        state["raw_llm_response"] = generate_best_response(_SYSTEM_TREATMENT, prompt, state)
        state["used_llm"] = True
    except Exception:
        state["raw_llm_response"] = ""
        state["used_llm"] = False
    return state


def uncertainty_node(state: AdvisoryState) -> AdvisoryState:
    """Cautious advisory for low-confidence predictions."""
    prompt = build_uncertainty_prompt(
        state["maturity"],
        state["disease"],
        _conf_tier(state["m_conf"]),
        _conf_tier(state["d_conf"]),
    )
    try:
        state["raw_llm_response"] = generate_best_response(_SYSTEM_UNCERTAINTY, prompt, state)
        state["used_llm"] = True
    except Exception:
        state["raw_llm_response"] = ""
        state["used_llm"] = False
    return state


# ═══════════════════════════════════════════════════════════════
# FORMATTER NODE
# ═══════════════════════════════════════════════════════════════

def formatter_node(state: AdvisoryState) -> AdvisoryState:
    if state["used_llm"] and state["raw_llm_response"]:
        lines = state["raw_llm_response"].splitlines()
        cleaned, prev_blank = [], False
        for line in lines:
            is_blank = (line.strip() == "")
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank
        state["response"] = "\n".join(cleaned).strip()
    else:
        state["response"] = generate_template_response(state)
        state["used_llm"] = False
    return state


# ═══════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def route_after_confidence(state: AdvisoryState) -> Literal["uncertain", "decision"]:
    return "uncertain" if state["mode"] == "uncertain" else "decision"


def route_after_decision(state: AdvisoryState) -> Literal["harvest", "triage"]:
    return "harvest" if state["mode"] == "harvest" else "triage"


# ═══════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════

def build_advisory_graph():
    g = StateGraph(AdvisoryState)

    g.add_node("input_validator",   input_validator)
    g.add_node("confidence_router", confidence_router_node)
    g.add_node("decision_router",   decision_router_node)
    g.add_node("triage",            triage_node)
    g.add_node("harvest_node",      harvest_node)
    g.add_node("treatment_node",    treatment_node)
    g.add_node("uncertainty_node",  uncertainty_node)
    g.add_node("formatter",         formatter_node)

    g.set_entry_point("input_validator")
    g.add_edge("input_validator", "confidence_router")

    g.add_conditional_edges(
        "confidence_router",
        route_after_confidence,
        {"uncertain": "uncertainty_node", "decision": "decision_router"},
    )
    g.add_conditional_edges(
        "decision_router",
        route_after_decision,
        {"harvest": "harvest_node", "triage": "triage"},
    )

    g.add_edge("triage",           "treatment_node")
    g.add_edge("harvest_node",     "formatter")
    g.add_edge("treatment_node",   "formatter")
    g.add_edge("uncertainty_node", "formatter")
    g.add_edge("formatter",        END)

    return g.compile()


_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_advisory_graph()
    return _graph


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def generate_advisory(
    maturity: str,
    disease:  str,
    m_conf:   float,
    d_conf:   float,
) -> tuple[str, bool]:
    """
    Run the LangGraph advisory pipeline.

    Parameters
    ----------
    maturity : str   — e.g. "Mature", "Semi-Mature", "Immature"
    disease  : str   — e.g. "Healthy", "Anthracnose", "Scab", "Styler-End-Rot"
    m_conf   : float — maturity confidence  [0, 1]
    d_conf   : float — disease  confidence  [0, 1]

    Returns
    -------
    (advice: str, used_llm: bool)
    """
    try:
        result = _get_graph().invoke({
            "maturity":         maturity,
            "disease":          disease,
            "m_conf":           m_conf,
            "d_conf":           d_conf,
            "mode":             "",
            "confidence_ok":    True,
            "severity":         "",
            "harvest_window":   "",
            "triage_notes":     "",
            "raw_llm_response": "",
            "response":         "",
            "used_llm":         False,
        })
        return result["response"], result["used_llm"]

    except Exception:
        traceback.print_exc()
        return generate_template_response({
            "maturity": maturity,
            "disease":  disease,
            "m_conf":   m_conf,
            "d_conf":   d_conf,
        }), False


if __name__ == "__main__":
    response, used_llm = generate_advisory(
        maturity="Mature",
        disease="Healthy",
        m_conf=0.9,
        d_conf=0.95,
    )
    print("\n==============================")
    print("USED LLM:", used_llm)
    print("==============================\n")
    print(response)