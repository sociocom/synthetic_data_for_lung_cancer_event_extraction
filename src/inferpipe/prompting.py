# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Tuple

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


STRUCTURED_TEMPLATE = """\
<guideline>

### Input:
<text>

### Output:
"""


def read_text_file(path: str) -> str:
    if not path:
        return ""
    if not os.path.exists(path):
        raise FileNotFoundError(f"guideline_path not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompts_toml(path: str) -> Tuple[str, str]:
    if not path:
        raise ValueError("prompt_file is empty")
    if tomllib is None:
        raise RuntimeError("tomllib is not available. Use Python 3.11+.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt_file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    system = data.get("system")
    user = data.get("user")
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"Missing or empty 'system' in {path}")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"Missing or empty 'user' in {path}")
    return system.rstrip(), user.rstrip()


def render_user_prompt(user_template: str, input_text: str, context_text: str | None = None) -> str:
    s = user_template
    if context_text is not None:
        if "{{CONTEXT}}" in s:
            s = s.replace("{{CONTEXT}}", context_text)
        else:
            s = f"{context_text}\n\n{s}"
    if "{{INPUT}}" in s:
        return s.replace("{{INPUT}}", input_text.strip())
    return f"{s}\n\n### 入力:\n{input_text.strip()}"


def build_chat_text(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "",
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        if getattr(tokenizer, "chat_template", None):
            kwargs = dict(
                tokenize=False,
                add_generation_prompt=True,
            )
            template = str(tokenizer.chat_template)
            # Qwen3 templates support `enable_thinking`; forcing False avoids
            # long reasoning traces that can consume generation budget.
            if "enable_thinking" in template:
                kwargs["enable_thinking"] = False
            # gpt-oss harmony templates take `reasoning_effort` (low/medium/high; default medium).
            if reasoning_effort and "reasoning_effort" in template:
                kwargs["reasoning_effort"] = reasoning_effort
            return tokenizer.apply_chat_template(
                messages,
                **kwargs,
            )
    except Exception:
        pass

    return f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}\n\n[Assistant]\n"


def build_structured_prompt(guideline: str, record_text: str) -> str:
    template = STRUCTURED_TEMPLATE.replace("<guideline>", guideline or "")
    return template.replace("<text>", record_text.strip())
