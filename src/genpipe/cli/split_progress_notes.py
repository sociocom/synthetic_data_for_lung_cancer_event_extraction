#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from typing import Iterator, List, Dict, Any

from genpipe.common.jsonl_io import iter_jsonl, write_jsonl


_SPLIT_RE = re.compile(r"\n\n(?=Progress note\s*\()", flags=re.IGNORECASE)


def _split_notes(text: str) -> List[str]:
    """Split a single progress_note blob into sub-notes.

    Minimal heuristics:
    - Split on blank-line before 'Progress note (' which appears in the synthetic data.
    - If no delimiter is found, return the original text as one note.
    """
    if not text:
        return []

    parts = _SPLIT_RE.split(text.strip())
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def iter_split_records(input_jsonl: str) -> Iterator[Dict[str, Any]]:
    for obj in iter_jsonl(input_jsonl):
        if "id" not in obj or "progress_note" not in obj:
            continue

        patient_id = str(obj["id"])
        raw = obj.get("progress_note")
        if raw is None:
            continue

        notes = _split_notes(str(raw))
        for i, note in enumerate(notes, start=1):
            yield {
                "patient_id": patient_id,
                "subnote_id": f"{patient_id}_{i}",
                "note_index": i,
                "note_text": note,
            }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split progress_notes.jsonl into progress_notes_split.jsonl")
    p.add_argument(
        "--input_jsonl",
        type=str,
        default="../../../data/processed/synthetic_data/progress_notes.jsonl",
        help="Input progress_notes.jsonl (expects keys: id, progress_note)",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="../../../data/processed/synthetic_data/progress_notes_split.jsonl",
        help="Output jsonl (keys: patient_id, subnote_id, note_index, note_text)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    write_jsonl(args.output_jsonl, iter_split_records(args.input_jsonl))


if __name__ == "__main__":
    main()
