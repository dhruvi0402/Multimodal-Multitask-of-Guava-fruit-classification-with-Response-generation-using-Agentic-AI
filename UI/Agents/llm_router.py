from __future__ import annotations
from typing import Dict

from Agents.models import _call_llama, _call_qwen
from Agents.judge import judge_outputs


def generate_best_response(system_prompt: str, user_prompt: str, state: Dict) -> str:
    """
    Call Llama and Qwen in parallel, return the higher-scoring response.
    Falls back to whichever model succeeds if one fails.
    Raises RuntimeError if both models fail.
    """
    llama_response, qwen_response = None, None

    try:
        llama_response = _call_llama(system_prompt, user_prompt)
    except Exception:
        pass

    try:
        qwen_response = _call_qwen(system_prompt, user_prompt)
    except Exception:
        pass

    if llama_response and qwen_response:
        return judge_outputs(llama_response, qwen_response, state)

    result = llama_response or qwen_response
    if result:
        return result

    raise RuntimeError("Both Llama and Qwen failed to produce a response.")