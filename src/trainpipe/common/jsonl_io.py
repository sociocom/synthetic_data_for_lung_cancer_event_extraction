# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, List, Optional


def load_text_file(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def iter_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e


def save_jsonl(objs: List[Dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_json(obj: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
