#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
from typing import Any, Dict, Iterator, List, Tuple

try:
    from src.eval_schema import EVAL_KEY_TYPES, canonicalize_eval_key
except ModuleNotFoundError:
    from eval_schema import EVAL_KEY_TYPES, canonicalize_eval_key

def _looks_present(value: Any) -> bool:
    """
    「ファイル内にキーが1つでもあればOK」という条件向け.
    値が None, 空文字, 空配列, 空辞書 だけは absent 扱いにする.
    それ以外は present 扱いにする.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True  # number, bool, etc.

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def parse_annotation_field(v: Any) -> Any:
    """Parse synthetic_data 'annotation' field into a JSON object.

    Accepts:
      - list/dict (already parsed)
      - string containing JSON
      - string containing fenced JSON: ```json ... ```
      - empty list string: "[]"

    Returns a Python object (typically list[dict]).
    """
    if v is None:
        return []
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        s = _strip_code_fence(v)
        s = s.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # as a fallback, try to extract JSON substring (first '[' or '{' ... last matching)
            start_candidates = [s.find("["), s.find("{")]
            start_candidates = [x for x in start_candidates if x != -1]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end_candidates = [s.rfind("]"), s.rfind("}")]
            end_candidates = [x for x in end_candidates if x != -1]
            if not end_candidates:
                raise
            end = max(end_candidates)
            return json.loads(s[start : end + 1])
    # unknown type -> treat as absent
    return []

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {ln} is not an object")
            yield obj

def keys_present_in_doc(doc: Any, keys: List[str]) -> Dict[str, bool]:
    """
    doc は, list[dict] を想定.
    ただし壊れた形式でも落とさず, 可能な範囲で探索する.
    """
    present = {k: False for k in keys}

    def scan_obj(obj: Any) -> None:
        if isinstance(obj, dict):
            for raw_key, value in obj.items():
                eval_key = canonicalize_eval_key(raw_key)
                if eval_key in present and (not present[eval_key]) and _looks_present(value):
                    present[eval_key] = True
            # 入れ子も一応見る
            for v in obj.values():
                scan_obj(v)
        elif isinstance(obj, list):
            for it in obj:
                scan_obj(it)

    scan_obj(doc)
    return present

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input_glob",
        required=True,
        help="Input JSONL glob. Each line is treated as 1 document/sample.",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_failed", required=True)
    ap.add_argument(
        "--jsonl_annotation_key",
        default="annotation",
        help="Field name that contains the annotation payload (string or JSON).",
    )
    args = ap.parse_args()

    keys = list(EVAL_KEY_TYPES.keys())
    paths = sorted(glob.glob(args.input_glob))

    total_docs = 0
    failed: List[Tuple[str, str]] = []
    doc_count_with_key = {k: 0 for k in keys}

    for p in paths:
        try:
            for ln, rec in enumerate(iter_jsonl(p), start=1):
                total_docs += 1
                try:
                    payload = parse_annotation_field(rec.get(args.jsonl_annotation_key))
                    pres = keys_present_in_doc(payload, keys)
                    for k, ok in pres.items():
                        if ok:
                            doc_count_with_key[k] += 1
                except Exception as e:
                    failed.append((f"{p}:{ln}", f"{type(e).__name__}: {e}"))
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "docs_with_key", "total_docs", "coverage"])
        for k in keys:
            c = doc_count_with_key[k]
            cov = (c / total_docs) if total_docs > 0 else 0.0
            w.writerow([k, c, total_docs, f"{cov:.6f}"])

    with open(args.out_failed, "w", encoding="utf-8") as f:
        for loc, reason in failed:
            f.write(f"{loc}\t{reason}\n")

    print(
        f"done, total_docs={total_docs}, out_csv={args.out_csv}, failed={len(failed)}, out_failed={args.out_failed}"
    )


if __name__ == "__main__":
    main()
