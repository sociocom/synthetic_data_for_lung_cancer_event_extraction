#!/usr/bin/env python3
"""Compute scores for inference results.

Input JSONL format
- gold file: each line is a JSON object with at least {"id": ..., "annotation": "{...}"}
- pred file: each line is a JSON object with at least {"id": ..., "pred_annotation": "{...}"}

Notes
- annotation/pred_annotation are JSON strings (not dicts) in the current dataset.
- IDs are used to align examples.

Usage
  python -m src.score --gold path/to/test.jsonl --pred path/to/pred_test.jsonl

This file is intended to be runnable from a thin bash wrapper (scripts/score.sh).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def _parse_label_dict(value: Any, *, where: str) -> Dict[str, int]:
    """Parse label mapping from either a dict or a JSON string."""
    if value is None:
        raise ValueError(f"Missing annotation at {where}")
    if isinstance(value, dict):
        obj = value
    elif isinstance(value, str):
        try:
            obj = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"annotation is not valid JSON string at {where}: {e}") from e
    else:
        raise ValueError(f"annotation must be dict or JSON string at {where}, got {type(value)}")

    out: Dict[str, int] = {}
    for k, v in obj.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = int(v)
        elif isinstance(v, str) and v.isdigit():
            out[k] = int(v)
        else:
            raise ValueError(f"Label value must be 0/1-like at {where} for key={k!r}, got {v!r}")
    return out


@dataclass
class LabelMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        denom = p + r
        return (2 * p * r / denom) if denom else 0.0


def _collect_labels(gold: Mapping[str, int], pred: Mapping[str, int], *, strict: bool) -> List[str]:
    keys = sorted(set(gold.keys()) | set(pred.keys()))
    if strict and set(gold.keys()) != set(pred.keys()):
        missing_in_pred = sorted(set(gold.keys()) - set(pred.keys()))
        extra_in_pred = sorted(set(pred.keys()) - set(gold.keys()))
        msg = []
        if missing_in_pred:
            msg.append(f"missing labels in pred: {missing_in_pred}")
        if extra_in_pred:
            msg.append(f"extra labels in pred: {extra_in_pred}")
        raise ValueError("Label set mismatch (strict mode): " + "; ".join(msg))
    return keys


def compute_metrics(
    pairs: Sequence[Tuple[Dict[str, int], Dict[str, int]]],
    *,
    strict_label_set: bool = False,
) -> Tuple[Dict[str, LabelMetrics], Dict[str, float]]:
    """Compute per-label confusion counts and aggregate scores."""
    if not pairs:
        raise ValueError("No aligned (gold, pred) pairs to score")

    # Determine union label set across corpus
    all_labels: List[str] = []
    seen = set()
    for g, p in pairs:
        for k in set(g.keys()) | set(p.keys()):
            if k not in seen:
                seen.add(k)
                all_labels.append(k)
    all_labels = sorted(all_labels)

    per_label: Dict[str, LabelMetrics] = {k: LabelMetrics() for k in all_labels}

    subset_correct = 0
    total = len(pairs)

    for gold, pred in pairs:
        labels = _collect_labels(gold, pred, strict=strict_label_set)

        # subset accuracy: all labels exact match
        if all(int(pred.get(k, 0)) == int(gold.get(k, 0)) for k in labels):
            subset_correct += 1

        for k in all_labels:
            y = int(gold.get(k, 0))
            yhat = int(pred.get(k, 0))
            m = per_label[k]
            if y == 1 and yhat == 1:
                m.tp += 1
            elif y == 0 and yhat == 1:
                m.fp += 1
            elif y == 1 and yhat == 0:
                m.fn += 1
            else:
                m.tn += 1

    # macro
    macro_p = sum(per_label[k].precision() for k in all_labels) / len(all_labels)
    macro_r = sum(per_label[k].recall() for k in all_labels) / len(all_labels)
    macro_f1 = sum(per_label[k].f1() for k in all_labels) / len(all_labels)

    # micro
    tp = sum(per_label[k].tp for k in all_labels)
    fp = sum(per_label[k].fp for k in all_labels)
    fn = sum(per_label[k].fn for k in all_labels)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    summary = {
        "num_examples": float(total),
        "num_labels": float(len(all_labels)),
        "subset_accuracy": subset_correct / total if total else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }

    return per_label, summary


def _align_by_id(
    gold_rows: Iterable[Dict[str, Any]],
    pred_rows: Iterable[Dict[str, Any]],
    *,
    gold_key: str,
    pred_key: str,
    strict_ids: bool,
) -> Tuple[List[str], List[Tuple[Dict[str, int], Dict[str, int]]]]:
    gold_map: Dict[str, Dict[str, int]] = {}
    for r in gold_rows:
        if "id" not in r:
            raise ValueError("gold row missing 'id'")
        _id = str(r["id"])
        gold_map[_id] = _parse_label_dict(r.get(gold_key), where=f"gold id={_id}")

    pred_map: Dict[str, Dict[str, int]] = {}
    for r in pred_rows:
        if "id" not in r:
            raise ValueError("pred row missing 'id'")
        _id = str(r["id"])
        pred_map[_id] = _parse_label_dict(r.get(pred_key), where=f"pred id={_id}")

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())

    missing_pred = sorted(gold_ids - pred_ids)
    extra_pred = sorted(pred_ids - gold_ids)

    if strict_ids and (missing_pred or extra_pred):
        parts = []
        if missing_pred:
            parts.append(f"missing predictions for {len(missing_pred)} ids")
        if extra_pred:
            parts.append(f"extra predictions for {len(extra_pred)} ids")
        raise ValueError("ID mismatch (strict mode): " + ", ".join(parts))

    common_ids = sorted(gold_ids & pred_ids)
    pairs = [(gold_map[i], pred_map[i]) for i in common_ids]
    return common_ids, pairs


def _print_report(per_label: Dict[str, LabelMetrics], summary: Dict[str, float]) -> None:
    # Summary
    print("# Score summary")
    for k in [
        "num_examples",
        "num_labels",
        "subset_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]:
        v = summary[k]
        if k.startswith("num_"):
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.6f}")

    # Per-label
    print("\n# Per-label")
    header = [
        "label",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    print("\t".join(header))
    for label in sorted(per_label.keys()):
        m = per_label[label]
        print(
            "\t".join(
                [
                    label,
                    f"{m.precision():.6f}",
                    f"{m.recall():.6f}",
                    f"{m.f1():.6f}",
                    str(m.tp),
                    str(m.fp),
                    str(m.fn),
                    str(m.tn),
                ]
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score predicted annotations against gold annotations (multi-label).")
    p.add_argument("--gold", type=Path, default=None, help="Path to gold JSONL (contains 'annotation').")
    p.add_argument("--pred", type=Path, default=None, help="Path to prediction JSONL (contains 'pred_annotation').")
    p.add_argument("--gold-key", type=str, default="annotation", help="Field name for gold label JSON.")
    p.add_argument("--pred-key", type=str, default="pred_annotation", help="Field name for predicted label JSON.")
    p.add_argument(
        "--strict-ids",
        action="store_true",
        help="Fail if gold/pred id sets differ (otherwise use intersection).",
    )
    p.add_argument(
        "--strict-label-set",
        action="store_true",
        help="Fail if a sample's gold/pred label key sets differ.",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )

    args = p.parse_args(argv)

    gold_path = args.gold or Path(os.environ.get("GOLD_JSONL", ""))
    pred_path = args.pred or Path(os.environ.get("PRED_JSONL", ""))

    if not gold_path or str(gold_path) == ".":
        raise SystemExit("--gold not provided and GOLD_JSONL not set")
    if not pred_path or str(pred_path) == ".":
        raise SystemExit("--pred not provided and PRED_JSONL not set")

    gold_path = gold_path.expanduser().resolve()
    pred_path = pred_path.expanduser().resolve()

    ids, pairs = _align_by_id(
        _read_jsonl(gold_path),
        _read_jsonl(pred_path),
        gold_key=args.gold_key,
        pred_key=args.pred_key,
        strict_ids=args.strict_ids,
    )

    per_label, summary = compute_metrics(pairs, strict_label_set=args.strict_label_set)

    _print_report(per_label, summary)

    if args.save_json is not None:
        out = {
            "summary": summary,
            "per_label": {
                k: {
                    "precision": per_label[k].precision(),
                    "recall": per_label[k].recall(),
                    "f1": per_label[k].f1(),
                    "tp": per_label[k].tp,
                    "fp": per_label[k].fp,
                    "fn": per_label[k].fn,
                    "tn": per_label[k].tn,
                }
                for k in sorted(per_label.keys())
            },
            "ids_scored": ids,
            "gold_path": str(gold_path),
            "pred_path": str(pred_path),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
