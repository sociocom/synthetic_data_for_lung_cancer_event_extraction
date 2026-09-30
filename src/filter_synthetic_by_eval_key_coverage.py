#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# uv run python filter_synthetic_by_eval_key_coverage.py \
#     --eval_jsonl ../data/processed/simulation/simu_annotation_noae.jsonl \
#     --synthetic_annotations_jsonl ../data/processed/synthetic_data_clean/annotations_bias_noae_v2_clean.jsonl \
#     --out_annotations_jsonl ../data/processed/synthetic_data_clean/annotations_bias_noae_v2_clean_95_all9.jsonl \
#     --synthetic_progress_jsonl ../data/processed/synthetic_data_clean/progress_notes_bias_clean.jsonl \
#     --out_progress_jsonl ../data/processed/synthetic_data_clean/progress_notes_bias_clean_95_all9.jsonl \
#     --threshold 0.95 \
#     --save_report_json ../data/processed/synthetic_data_clean/filter_95_all9_report.json

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from src.schema import KEY_TYPES
except ModuleNotFoundError:
    from schema import KEY_TYPES


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield obj


def parse_annotation(value: Any) -> List[Dict[str, Any]]:
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict)]


def collect_present_keys(events: List[Dict[str, Any]], valid_keys: set[str]) -> set[str]:
    present: set[str] = set()
    for ev in events:
        for k in ev.keys():
            if k in valid_keys:
                present.add(k)
    return present


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_required_keys(
    eval_jsonl: str,
    annotation_key: str,
    threshold: float,
    forced_keys: List[str],
    dropped_keys: List[str],
) -> Tuple[List[str], Dict[str, float], int]:
    valid_keys = set(KEY_TYPES.keys())
    n_rows = 0
    counts = {k: 0 for k in valid_keys}

    for obj in iter_jsonl(eval_jsonl):
        n_rows += 1
        events = parse_annotation(obj.get(annotation_key))
        present = collect_present_keys(events, valid_keys)
        for k in present:
            counts[k] += 1

    if n_rows == 0:
        raise ValueError(f"No rows found in eval_jsonl: {eval_jsonl}")

    ratios = {k: (counts[k] / n_rows) for k in valid_keys}
    if forced_keys:
        required = [k for k in forced_keys if k in valid_keys]
        if len(required) != len(forced_keys):
            unknown = [k for k in forced_keys if k not in valid_keys]
            raise ValueError(f"Unknown forced keys: {unknown}")
    else:
        required = sorted([k for k, r in ratios.items() if r >= threshold])

    if dropped_keys:
        drop_set = set(dropped_keys)
        required = [k for k in required if k not in drop_set]

    if not required:
        raise ValueError(
            "required_keys is empty. Lower --threshold or pass --required_keys explicitly."
        )
    return required, ratios, n_rows


def filter_synthetic_annotations(
    synthetic_annotations_jsonl: str,
    annotation_key: str,
    id_key: str,
    required_keys: List[str],
    min_key_match_count: int,
) -> Tuple[List[Dict[str, Any]], set[str], Dict[str, Any]]:
    valid_keys = set(KEY_TYPES.keys())
    req_set = set(required_keys)
    out_rows: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    total_rows = 0
    hist: Dict[int, int] = {}

    for obj in iter_jsonl(synthetic_annotations_jsonl):
        total_rows += 1
        events = parse_annotation(obj.get(annotation_key))
        present = collect_present_keys(events, valid_keys)
        matched = len(req_set & present)
        hist[matched] = hist.get(matched, 0) + 1

        if matched >= min_key_match_count:
            out_rows.append(obj)
            rid = obj.get(id_key)
            if rid is not None:
                selected_ids.add(str(rid))

    kept_rows = len(out_rows)
    report = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
        "kept_ratio": (kept_rows / total_rows) if total_rows > 0 else 0.0,
        "match_count_histogram": {str(k): v for k, v in sorted(hist.items())},
    }
    return out_rows, selected_ids, report


def filter_progress_rows(
    synthetic_progress_jsonl: str,
    id_key: str,
    selected_ids: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    total_rows = 0
    missing_id_rows = 0

    for obj in iter_jsonl(synthetic_progress_jsonl):
        total_rows += 1
        rid = obj.get(id_key)
        if rid is None:
            missing_id_rows += 1
            continue
        if str(rid) in selected_ids:
            out_rows.append(obj)

    kept_rows = len(out_rows)
    report = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
        "missing_id_rows": missing_id_rows,
        "kept_ratio": (kept_rows / total_rows) if total_rows > 0 else 0.0,
    }
    return out_rows, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter synthetic annotation JSONL by requiring keys that appear in eval data "
            "at or above a threshold (document-level key presence)."
        )
    )
    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--synthetic_annotations_jsonl", type=str, required=True)
    p.add_argument("--out_annotations_jsonl", type=str, required=True)

    p.add_argument("--synthetic_progress_jsonl", type=str, default="")
    p.add_argument("--out_progress_jsonl", type=str, default="")

    p.add_argument("--annotation_key", type=str, default="annotation")
    p.add_argument("--id_key", type=str, default="id")

    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Eval document-level key appearance threshold (default: 0.95).",
    )
    p.add_argument(
        "--required_keys",
        type=str,
        default="",
        help="Optional comma-separated key list. If set, overrides --threshold auto-selection.",
    )
    p.add_argument(
        "--drop_keys",
        type=str,
        default="",
        help="Optional comma-separated keys to remove from required set.",
    )
    p.add_argument(
        "--min_key_match_count",
        type=int,
        default=-1,
        help=(
            "Minimum number of required keys that must appear in a document. "
            "Default: all required keys (i.e., 9/9 with current default eval+threshold)."
        ),
    )
    p.add_argument(
        "--save_report_json",
        type=str,
        default="",
        help="Optional path to save summary report JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    forced_keys = [x.strip() for x in args.required_keys.split(",") if x.strip()]
    dropped_keys = [x.strip() for x in args.drop_keys.split(",") if x.strip()]

    required_keys, eval_ratios, eval_rows = resolve_required_keys(
        eval_jsonl=args.eval_jsonl,
        annotation_key=args.annotation_key,
        threshold=float(args.threshold),
        forced_keys=forced_keys,
        dropped_keys=dropped_keys,
    )

    use_all_required_default = int(args.min_key_match_count) <= 0
    min_match = len(required_keys) if use_all_required_default else int(args.min_key_match_count)
    if min_match > len(required_keys):
        raise ValueError(
            f"min_key_match_count={min_match} cannot exceed required_keys count={len(required_keys)}"
        )

    filtered_ann, selected_ids, ann_report = filter_synthetic_annotations(
        synthetic_annotations_jsonl=args.synthetic_annotations_jsonl,
        annotation_key=args.annotation_key,
        id_key=args.id_key,
        required_keys=required_keys,
        min_key_match_count=min_match,
    )
    save_jsonl(args.out_annotations_jsonl, filtered_ann)

    prog_report: Dict[str, Any] | None = None
    if args.synthetic_progress_jsonl:
        if not args.out_progress_jsonl:
            raise ValueError(
                "--out_progress_jsonl is required when --synthetic_progress_jsonl is provided."
            )
        filtered_prog, prog_report = filter_progress_rows(
            synthetic_progress_jsonl=args.synthetic_progress_jsonl,
            id_key=args.id_key,
            selected_ids=selected_ids,
        )
        save_jsonl(args.out_progress_jsonl, filtered_prog)

    summary = {
        "inputs": {
            "eval_jsonl": args.eval_jsonl,
            "synthetic_annotations_jsonl": args.synthetic_annotations_jsonl,
            "synthetic_progress_jsonl": args.synthetic_progress_jsonl,
        },
        "outputs": {
            "out_annotations_jsonl": args.out_annotations_jsonl,
            "out_progress_jsonl": args.out_progress_jsonl,
        },
        "settings": {
            "annotation_key": args.annotation_key,
            "id_key": args.id_key,
            "threshold": float(args.threshold),
            "required_keys": required_keys,
            "required_keys_count": len(required_keys),
            "min_key_match_count": min_match,
            "default_match_policy": (
                "all_required_keys" if use_all_required_default else "at_least_min_key_match_count"
            ),
        },
        "eval": {
            "rows": eval_rows,
            "ratios_for_required_keys": {
                k: eval_ratios[k] for k in required_keys
            },
        },
        "annotations_filter": ann_report,
        "progress_filter": prog_report,
    }

    if args.save_report_json:
        out = Path(args.save_report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"required_keys({len(required_keys)}): {required_keys}")
    print(f"min_key_match_count: {min_match}")
    if use_all_required_default:
        print("match_policy: all_required_keys (default)")
    else:
        print("match_policy: at_least_min_key_match_count")
    print(
        "annotations: "
        f"kept={ann_report['kept_rows']}/{ann_report['total_rows']} "
        f"({ann_report['kept_ratio']:.4f})"
    )
    if prog_report is not None:
        print(
            "progress: "
            f"kept={prog_report['kept_rows']}/{prog_report['total_rows']} "
            f"({prog_report['kept_ratio']:.4f})"
        )
    if args.save_report_json:
        print(f"saved report: {args.save_report_json}")


if __name__ == "__main__":
    main()
