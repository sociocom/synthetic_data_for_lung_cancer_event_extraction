#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


DEFAULT_LIST_VALUE_SCHEMA_PATH = Path(__file__).with_name("list_value_schema_noae_v2.json")
KEY_LINE_RE = re.compile(r"^\s*-\s*JSONキー:\s*(.+?)\s*$")


def load_prompt_user_text(prompt_file: Path) -> str:
    if tomllib is not None:
        with prompt_file.open("rb") as f:
            data = tomllib.load(f)
        user = data.get("user")
        if isinstance(user, str) and user.strip():
            return user

    text = prompt_file.read_text(encoding="utf-8")
    m = re.search(r'(?ms)^\s*user\s*=\s*"""\\?\n(.*?)\n"""', text)
    if not m:
        raise ValueError(f"'user' section is missing or could not be parsed: {prompt_file}")
    user = m.group(1)
    if not user.strip():
        raise ValueError(f"'user' section is empty: {prompt_file}")
    return user


def extract_allowed_values_from_user_text(user_text: str) -> dict[str, list[str]]:
    lines = user_text.splitlines()
    allowed: dict[str, list[str]] = {}
    current_key: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = KEY_LINE_RE.match(line)
        if m:
            current_key = m.group(1).strip()
            i += 1
            continue

        if current_key is not None and "候補値:" in line:
            list_text = line.split("候補値:", 1)[1].strip()
            bracket_balance = list_text.count("[") - list_text.count("]")
            j = i
            while bracket_balance > 0 and j + 1 < len(lines):
                j += 1
                nxt = lines[j].strip()
                list_text += nxt
                bracket_balance += nxt.count("[") - nxt.count("]")

            sanitized = (
                list_text.replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
                .replace("‘", "'")
            )
            try:
                parsed = ast.literal_eval(sanitized)
            except Exception:
                try:
                    parsed = json.loads(sanitized)
                except Exception:
                    parsed = None

            if isinstance(parsed, list):
                allowed[current_key] = [str(v).strip() for v in parsed if str(v).strip()]

            i = j + 1
            continue

        i += 1

    return allowed


def load_allowed_values_from_schema(schema_file: Path) -> dict[str, list[str]]:
    data = json.loads(schema_file.read_text(encoding="utf-8"))
    keys = data.get("keys")
    if not isinstance(keys, dict):
        raise ValueError(f"schema 'keys' must be an object: {schema_file}")

    allowed: dict[str, list[str]] = {}
    for key, meta in keys.items():
        if not isinstance(key, str) or not isinstance(meta, dict):
            continue
        vals = meta.get("allowed_values")
        if isinstance(vals, list):
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            if cleaned:
                allowed[key] = cleaned
    return allowed


def load_allowed_values(
    *,
    schema_file: Path | None = None,
    prompt_file: Path | None = None,
) -> tuple[dict[str, list[str]], str]:
    if schema_file is None:
        schema_file = DEFAULT_LIST_VALUE_SCHEMA_PATH if DEFAULT_LIST_VALUE_SCHEMA_PATH.exists() else None

    if schema_file is not None:
        if not schema_file.exists():
            raise FileNotFoundError(f"schema_file not found: {schema_file}")
        return load_allowed_values_from_schema(schema_file), str(schema_file)

    if prompt_file is None:
        raise ValueError("Either schema_file or prompt_file is required.")
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")
    user_text = load_prompt_user_text(prompt_file)
    return extract_allowed_values_from_user_text(user_text), str(prompt_file)
