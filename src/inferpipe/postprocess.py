# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

try:
    from src.normalize import extract_json_from_text_lenient
except ModuleNotFoundError:
    from normalize import extract_json_from_text_lenient


_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)",
    flags=re.DOTALL,
)
_HARMONY_SPECIAL_RE = re.compile(r"<\|(?:start|end|message|channel|return|call|constrain|endoftext)\|>")


def extract_final_channel(text: str) -> tuple[str, bool]:
    """Split a harmony-format (gpt-oss) generation into the final-channel message.

    Returns (final_text, found). When no final channel is present (e.g. the generation
    stopped inside the analysis channel), returns ("", False) so the prediction becomes []
    and the failure can be counted; the caller keeps the full raw text separately.
    """
    text = text or ""
    matches = _FINAL_CHANNEL_RE.findall(text)
    if not matches:
        return "", False
    final = matches[-1]
    final = _HARMONY_SPECIAL_RE.sub("", final).strip()
    return final, True


def harmony_prediction_text(raw_text: str) -> tuple[str, bool]:
    """Prediction text from a harmony-format generation.

    Primary: the final channel. Fallback (found=False): the model ended the generation inside
    the analysis channel; strip the harmony special tokens and let the lenient JSON extractor
    find the answer in the full text (the same treatment as non-harmony SLM outputs).
    """
    final, found = extract_final_channel(raw_text)
    if found:
        return final, True
    return _HARMONY_SPECIAL_RE.sub(" ", raw_text or "").strip(), False


def canonicalize_prediction_json(pred_text: str, max_events: int) -> str:
    text = (pred_text or "").strip()
    if not text:
        return "[]"
    # Qwen-family may emit explicit reasoning tags before JSON.
    # Remove them to stabilize downstream JSON extraction.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    if not text:
        return "[]"

    parsed: Any
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = extract_json_from_text_lenient(text)
        except Exception:
            parsed = []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        parsed = []

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for ev in parsed:
        if not isinstance(ev, dict):
            continue
        key = json.dumps(ev, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
        if max_events > 0 and len(deduped) >= max_events:
            break

    return json.dumps(deduped, ensure_ascii=False)
