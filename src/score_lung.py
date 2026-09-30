#!/usr/bin/env python3
"""
Hungarian-based scoring for lung event extraction outputs.

Input JSONL format (score.py-like)
- gold file: each line has at least {"id": "...", "annotation": "{...}"}
- pred file: each line has at least {"id": "...", "pred_annotation": "{...}"}

Where annotation/pred_annotation are JSON strings representing either:
- a dict (single event), or
- a list of dicts (multiple events)

Scoring
- Each event dict is normalized into a multiset Counter of (key, atom) pairs
- Build score matrix between gold events and pred events: overlap count (multiset intersection size)
- Use Hungarian algorithm to find 1-1 assignment maximizing total overlap (tp_items)
- Compute item-level precision/recall/f1:
    precision = tp_items / pred_total_items
    recall    = tp_items / gold_total_items

Outputs
- Prints overall summary to stdout
- Optionally saves detailed JSON via --save-json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from src.eval_schema import EVAL_KEY_ALIASES, EVAL_KEY_TYPES, canonicalize_eval_key
from src.normalize import normalize_atoms, extract_json_from_text_lenient
from src.procedurediagnosis_hierarchy import (
    is_pred_child_or_equal_procedurediagnosis,
)


ProcedureDiagnosisMatchMode = Literal["strict", "pred_child_of_gold"]


# -----------------------------
# JSONL I/O (score.py-like)
# -----------------------------
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


def _parse_json_value(value: Any, *, where: str) -> Any:
    """Parse JSON value from either a Python object (dict/list) or a JSON string.

    If the value is a string but is not valid JSON, fall back to lenient extraction.
    If even lenient extraction fails (e.g., truncated output), return an empty list
    so scoring can continue (treated as 'no predictions').
    """
    if value is None:
        raise ValueError(f"Missing annotation at {where}")
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return extract_json_from_text_lenient(value)
            except Exception as e:
                print(
                    f"[warn] failed to parse annotation JSON at {where}; treating as empty. error={e}",
                    file=sys.stderr,
                )
                return []
    raise ValueError(f"annotation must be dict/list or JSON string at {where}, got {type(value)}")


def _ensure_list_of_dicts(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return [o for o in x if isinstance(o, dict)]
    return []


# -----------------------------
# Event representation
# -----------------------------
def _event_counter(obj: Mapping[str, Any]) -> Counter:
    """
    Represent one event dict as multiset of (key, atom).
    Only keys in EVAL_KEY_TYPES are considered.
    """
    c: Counter = Counter()
    if not isinstance(obj, Mapping):
        return c
    for key, raw_value in obj.items():
        eval_key = canonicalize_eval_key(key)
        if eval_key not in EVAL_KEY_TYPES:
            continue
        for atom in normalize_atoms(key, raw_value):
            c[(eval_key, atom)] += 1
    return c


def _total_item_count(event_counters: Sequence[Counter]) -> int:
    return sum(sum(ec.values()) for ec in event_counters)


def _counter_by_key(counter: Counter) -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for (key, atom), count in counter.items():
        out[key][atom] += count
    return out


def _procedurediagnosis_atom_to_label(atom: Any) -> str:
    if isinstance(atom, tuple) and len(atom) == 2 and atom[0] == "str":
        return str(atom[1])
    return str(atom)


def _procedurediagnosis_tp_count(
    gold_atoms: Counter,
    pred_atoms: Counter,
    *,
    mode: ProcedureDiagnosisMatchMode,
) -> int:
    if mode == "strict":
        tp = 0
        for atom, g_count in gold_atoms.items():
            tp += min(g_count, pred_atoms.get(atom, 0))
        return tp

    # Build unit-level bipartite graph and maximize matches with Hungarian.
    gold_units: List[str] = []
    pred_units: List[str] = []
    for atom, count in gold_atoms.items():
        gold_units.extend([_procedurediagnosis_atom_to_label(atom)] * int(count))
    for atom, count in pred_atoms.items():
        pred_units.extend([_procedurediagnosis_atom_to_label(atom)] * int(count))

    n = len(gold_units)
    m = len(pred_units)
    if n == 0 or m == 0:
        return 0

    score_matrix: List[List[int]] = []
    for g in gold_units:
        row = []
        for p in pred_units:
            row.append(1 if is_pred_child_or_equal_procedurediagnosis(p, g) else 0)
        score_matrix.append(row)

    _assign, tp_items = _hungarian_maximize(score_matrix)
    return int(tp_items)


def _key_tp_count(
    key: str,
    gold_atoms: Counter,
    pred_atoms: Counter,
    *,
    pd_match_mode: ProcedureDiagnosisMatchMode,
) -> int:
    if key == "ProcedureDiagnosis":
        return _procedurediagnosis_tp_count(gold_atoms, pred_atoms, mode=pd_match_mode)
    tp = 0
    for atom, g_count in gold_atoms.items():
        tp += min(g_count, pred_atoms.get(atom, 0))
    return tp


def _intersection_count(
    c1: Counter,
    c2: Counter,
    *,
    pd_match_mode: ProcedureDiagnosisMatchMode,
) -> int:
    by_key_1 = _counter_by_key(c1)
    by_key_2 = _counter_by_key(c2)
    tp = 0
    for key in (set(by_key_1.keys()) | set(by_key_2.keys())):
        tp += _key_tp_count(
            key,
            by_key_1.get(key, Counter()),
            by_key_2.get(key, Counter()),
            pd_match_mode=pd_match_mode,
        )
    return tp


# -----------------------------
# Hungarian (min-cost) for n <= m
# -----------------------------
def _hungarian_min_cost(cost: List[List[int]]) -> List[int]:
    """
    Solve min-cost assignment for cost matrix with shape n x m, requiring n <= m.
    Returns assignment list of length n, where assign[i] = chosen column index for row i.
    """
    n = len(cost)
    m = len(cost[0]) if n > 0 else 0
    if n == 0:
        return []
    if m == 0:
        return [-1] * n
    if n > m:
        raise ValueError("hungarian_min_cost requires n <= m")

    # 1-indexed arrays
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = row assigned to column j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [10**18] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**18
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        # augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = [-1] * n
    for j in range(1, m + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign


def _hungarian_maximize(score_matrix: List[List[int]]) -> Tuple[List[int], int]:
    """
    Maximize total score for score matrix n x m (any shape).
    Returns (assign, total_score).
    assign maps each gold row i to a pred column j (or dummy), length n.
    """
    n = len(score_matrix)
    m = len(score_matrix[0]) if n > 0 else 0
    if n == 0:
        return [], 0
    if m == 0:
        return [-1] * n, 0

    # Ensure rectangular
    for row in score_matrix:
        if len(row) != m:
            raise ValueError("score_matrix must be rectangular")

    # If preds < golds, pad columns with zeros (dummy preds) so m == n
    if m < n:
        pad = n - m
        score_matrix = [row + [0] * pad for row in score_matrix]
        m = n

    # Convert to min-cost
    max_score = 0
    for i in range(n):
        for j in range(m):
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
    cost = [[max_score - score_matrix[i][j] for j in range(m)] for i in range(n)]

    assign = _hungarian_min_cost(cost)

    total = 0
    for i, j in enumerate(assign):
        if 0 <= j < m:
            total += score_matrix[i][j]
    return assign, total


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class ExampleScore:
    id: str
    gold_items: int
    pred_items: int
    tp_items: int
    precision: float
    recall: float
    f1: float


def _prf(tp: int, pred_total: int, gold_total: int) -> Tuple[float, float, float]:
    prec = tp / pred_total if pred_total > 0 else 0.0
    rec = tp / gold_total if gold_total > 0 else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _acc_key_count(
    by_key: Dict[str, Dict[str, int]],
    key: str,
    *,
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
) -> None:
    rec = by_key.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
    rec["tp"] += tp
    rec["fp"] += fp
    rec["fn"] += fn


def _merge_key_counts(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]) -> None:
    for key, rec in src.items():
        _acc_key_count(dst, key, tp=rec.get("tp", 0), fp=rec.get("fp", 0), fn=rec.get("fn", 0))


def _compute_per_key_metrics(by_key: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    per_key: Dict[str, Dict[str, Any]] = {}
    for key in sorted(by_key.keys()):
        tp = int(by_key[key]["tp"])
        fp = int(by_key[key]["fp"])
        fn = int(by_key[key]["fn"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_key[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_items": tp,
            "fp_items": fp,
            "fn_items": fn,
            "gold_items": tp + fn,
            "pred_items": tp + fp,
        }
    return per_key


def _align_by_id_events(
    gold_rows: Iterable[Dict[str, Any]],
    pred_rows: Iterable[Dict[str, Any]],
    *,
    gold_key: str,
    pred_key: str,
    strict_ids: bool,
) -> Tuple[List[str], List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]]:
    gold_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in gold_rows:
        if "id" not in r:
            raise ValueError("gold row missing 'id'")
        _id = str(r["id"])
        obj = _parse_json_value(r.get(gold_key), where=f"gold id={_id}")
        gold_map[_id] = _ensure_list_of_dicts(obj)

    pred_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in pred_rows:
        if "id" not in r:
            raise ValueError("pred row missing 'id'")
        _id = str(r["id"])
        obj = _parse_json_value(r.get(pred_key), where=f"pred id={_id}")
        pred_map[_id] = _ensure_list_of_dicts(obj)

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


def score_one_with_key_counts(
    gold_events: List[Dict[str, Any]],
    pred_events: List[Dict[str, Any]],
    *,
    pd_match_mode: ProcedureDiagnosisMatchMode = "strict",
) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    gold_ec = [_event_counter(o) for o in gold_events]
    pred_ec = [_event_counter(o) for o in pred_events]

    gold_total = _total_item_count(gold_ec)
    pred_total = _total_item_count(pred_ec)
    by_key: Dict[str, Dict[str, int]] = {}

    n = len(gold_ec)
    m = len(pred_ec)

    if n == 0 and m == 0:
        return 0, 0, 0, by_key

    if n == 0 and m > 0:
        for pc in pred_ec:
            for (key, _atom), count in pc.items():
                _acc_key_count(by_key, key, fp=count)
        return 0, pred_total, 0, by_key

    if n > 0 and m == 0:
        for gc in gold_ec:
            for (key, _atom), count in gc.items():
                _acc_key_count(by_key, key, fn=count)
        return gold_total, 0, 0, by_key

    score_matrix = [
        [
            _intersection_count(gold_ec[i], pred_ec[j], pd_match_mode=pd_match_mode)
            for j in range(m)
        ]
        for i in range(n)
    ]
    assign, tp_items = _hungarian_maximize(score_matrix)
    used_pred = set()

    for i, gc in enumerate(gold_ec):
        j = assign[i] if i < len(assign) else -1
        pc = pred_ec[j] if (0 <= j < m) else Counter()
        if 0 <= j < m:
            used_pred.add(j)

        gc_by_key = _counter_by_key(gc)
        pc_by_key = _counter_by_key(pc)
        for key in (set(gc_by_key.keys()) | set(pc_by_key.keys())):
            gold_atoms = gc_by_key.get(key, Counter())
            pred_atoms = pc_by_key.get(key, Counter())
            gold_count = sum(gold_atoms.values())
            pred_count = sum(pred_atoms.values())
            tp = _key_tp_count(key, gold_atoms, pred_atoms, pd_match_mode=pd_match_mode)
            fn = int(gold_count) - int(tp)
            fp = int(pred_count) - int(tp)
            _acc_key_count(by_key, key, tp=tp, fp=fp, fn=fn)

    for j, pc in enumerate(pred_ec):
        if j in used_pred:
            continue
        for (key, _atom), count in pc.items():
            _acc_key_count(by_key, key, fp=count)

    return gold_total, pred_total, tp_items, by_key


def score_one(
    gold_events: List[Dict[str, Any]],
    pred_events: List[Dict[str, Any]],
    *,
    pd_match_mode: ProcedureDiagnosisMatchMode = "strict",
) -> Tuple[int, int, int]:
    gold_total, pred_total, tp_items, _ = score_one_with_key_counts(
        gold_events,
        pred_events,
        pd_match_mode=pd_match_mode,
    )
    return gold_total, pred_total, tp_items


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score lung events with Hungarian matching.")
    p.add_argument("--gold", type=Path, default=None, help="Path to gold JSONL (contains 'annotation').")
    p.add_argument("--pred", type=Path, default=None, help="Path to prediction JSONL (contains 'pred_annotation').")
    p.add_argument("--gold-key", type=str, default="annotation", help="Field name for gold event JSON.")
    p.add_argument("--pred-key", type=str, default="pred_annotation", help="Field name for predicted event JSON.")
    p.add_argument(
        "--strict-ids",
        action="store_true",
        help="Fail if gold/pred id sets differ (otherwise use intersection).",
    )
    p.add_argument("--save-json", type=Path, default=None, help="Optional path to save scores as JSON.")
    p.add_argument(
        "--save-per-key-json",
        type=Path,
        default=None,
        help="Optional path to save per-key item-level scores as JSON.",
    )
    p.add_argument(
        "--procedurediagnosis-match-mode",
        type=str,
        choices=["strict", "pred_child_of_gold"],
        default=(os.environ.get("PROCEDUREDIAGNOSIS_MATCH_MODE") or "strict"),
        help=(
            "Matching mode for ProcedureDiagnosis only. "
            "'strict': exact match. "
            "'pred_child_of_gold': pred can be a descendant of gold."
        ),
    )
    args = p.parse_args(argv)

    gold_path = (args.gold or Path(os.environ.get("GOLD_JSONL", ""))).expanduser().resolve()
    pred_path = (args.pred or Path(os.environ.get("PRED_JSONL", ""))).expanduser().resolve()

    if not str(gold_path) or str(gold_path) == ".":
        raise SystemExit("--gold not provided and GOLD_JSONL not set")
    if not str(pred_path) or str(pred_path) == ".":
        raise SystemExit("--pred not provided and PRED_JSONL not set")

    ids, pairs = _align_by_id_events(
        _read_jsonl(gold_path),
        _read_jsonl(pred_path),
        gold_key=args.gold_key,
        pred_key=args.pred_key,
        strict_ids=args.strict_ids,
    )

    per_example: List[ExampleScore] = []
    total_gold_items = 0
    total_pred_items = 0
    total_tp_items = 0
    total_key_counts: Dict[str, Dict[str, int]] = {}

    for _id, (gold_events, pred_events) in zip(ids, pairs):
        gold_total, pred_total, tp_items, key_counts = score_one_with_key_counts(
            gold_events,
            pred_events,
            pd_match_mode=args.procedurediagnosis_match_mode,
        )
        prec, rec, f1 = _prf(tp_items, pred_total, gold_total)
        per_example.append(
            ExampleScore(
                id=_id,
                gold_items=gold_total,
                pred_items=pred_total,
                tp_items=tp_items,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )
        total_gold_items += gold_total
        total_pred_items += pred_total
        total_tp_items += tp_items
        _merge_key_counts(total_key_counts, key_counts)

    prec_all, rec_all, f1_all = _prf(total_tp_items, total_pred_items, total_gold_items)

    # stdout summary (simple)
    print("# Score summary (lung/hungarian, item-level)")
    print(f"num_examples: {len(per_example)}")
    print(f"total_gold_items: {total_gold_items}")
    print(f"total_pred_items: {total_pred_items}")
    print(f"total_tp_items: {total_tp_items}")
    print(f"micro_precision: {prec_all:.6f}")
    print(f"micro_recall: {rec_all:.6f}")
    print(f"micro_f1: {f1_all:.6f}")

    if args.save_json is not None:
        out = {
            "summary": {
                "num_examples": len(per_example),
                "total_gold_items": total_gold_items,
                "total_pred_items": total_pred_items,
                "total_tp_items": total_tp_items,
                "micro_precision": prec_all,
                "micro_recall": rec_all,
                "micro_f1": f1_all,
            },
            "per_example": [
                {
                    "id": s.id,
                    "gold_items": s.gold_items,
                    "pred_items": s.pred_items,
                    "tp_items": s.tp_items,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                }
                for s in per_example
            ],
            "ids_scored": ids,
            "gold_path": str(gold_path),
            "pred_path": str(pred_path),
            "method": "hungarian_maximize(intersection_count over (key,atom) multiset)",
            "procedurediagnosis_match_mode": args.procedurediagnosis_match_mode,
            "evaluation_key_aliases": EVAL_KEY_ALIASES,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.save_per_key_json is not None:
        per_key_out = {
            "summary": {
                "num_keys": len(total_key_counts),
                "note": "Per-key item-level scores with Hungarian event alignment.",
                "evaluation_key_aliases": EVAL_KEY_ALIASES,
            },
            "per_key": _compute_per_key_metrics(total_key_counts),
        }
        args.save_per_key_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_per_key_json.write_text(
            json.dumps(per_key_out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
