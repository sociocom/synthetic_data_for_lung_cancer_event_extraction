# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_ids(path: str, id_key: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _id = obj.get(id_key)
            if _id is None:
                continue
            out.add(str(_id))
    return out


def collect_rows(
    input_jsonl: str,
    id_key: str,
    input_key: str,
    resume: bool,
    done_ids: set[str],
) -> tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    missing = 0
    for obj in iter_jsonl(input_jsonl):
        if id_key not in obj or input_key not in obj:
            missing += 1
            continue

        rid = str(obj[id_key])
        if resume and rid in done_ids:
            continue

        txt = obj.get(input_key)
        if txt is None:
            missing += 1
            continue
        txt = str(txt)
        if not txt.strip():
            missing += 1
            continue

        rows.append({"id": rid, "text": txt})

    return rows, missing


def slice_range(n: int, start: int, end: int) -> Tuple[int, int]:
    s = max(0, int(start))
    e = int(end)
    if e <= 0 or e > n:
        e = n
    if s > e:
        s = e
    return s, e


def ensure_output_parent(output_jsonl: str) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
