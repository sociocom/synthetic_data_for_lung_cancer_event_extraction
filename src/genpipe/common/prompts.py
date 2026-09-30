# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

try:
    import tomllib  # Python 3.11+
except Exception as e:  # pragma: no cover
    tomllib = None  # type: ignore


@dataclass(frozen=True)
class Prompts:
    system: str
    user: str


def load_prompts_toml(prompt_path: str) -> Prompts:
    """
    Load prompts from TOML file. Required keys: system, user
    """
    if tomllib is None:
        raise RuntimeError("tomllib is not available. Use Python 3.11+.")

    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    data = tomllib.loads(p.read_text(encoding="utf-8"))
    system = data.get("system")
    user = data.get("user")

    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"Missing or empty 'system' in {prompt_path}")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"Missing or empty 'user' in {prompt_path}")

    return Prompts(system=system.rstrip(), user=user.rstrip())


def render_user_prompt(user_template: str, input_text: str, context_text: str | None = None) -> str:
    """
    Replace placeholders:
      - {{INPUT}}   required
      - {{CONTEXT}} optional
    """
    s = user_template

    if context_text is not None:
        if "{{CONTEXT}}" in s:
            s = s.replace("{{CONTEXT}}", context_text)
        else:
            # If template doesn't include it, prepend for safety.
            s = f"{context_text}\n\n{s}"

    if "{{INPUT}}" in s:
        s = s.replace("{{INPUT}}", input_text)
    else:
        # fallback
        s = f"{s}\n\n### 入力:\n{input_text}"

    return s
