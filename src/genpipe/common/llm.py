# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    max_retry: int = 2


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def call_once(
    client: OpenAI,
    cfg: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    cid_for_log: str,
) -> str:
    """
    One request with retry/backoff for transient errors.
    """
    backoff = 1.0
    for attempt in range(1, cfg.max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            retryable = any(x in msg for x in ["429", "Rate limit", "timeout", "5", "Temporary", "temporarily"])
            if (not retryable) or (attempt == cfg.max_retry):
                raise RuntimeError(f"{cid_for_log}: request failed (attempt {attempt}/{cfg.max_retry}): {msg}") from e

            sleep_s = backoff + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 30)

    raise RuntimeError(f"{cid_for_log}: request failed after retries")
