# -*- coding: utf-8 -*-
"""Extract document x item tp/fp/fn counts for every condition and seed.

Output: data/paper/revision/calc/doc_key_counts.tsv
  columns: condition, seed, fold, doc_id, key, tp, fp, fn
Keys are canonical evaluation keys (TBB/TBLB merged). p63 etc. are kept in the
file but flagged via EXCLUDED_KEYS downstream so the 37-item protocol can be applied.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.score_lung import _align_by_id_events, score_one_with_key_counts  # noqa: E402
from src.paper_revision.runs import (  # noqa: E402
    EXP08_RUNS, GOLD_JSONL, REPO, pred_files, fold_of_doc, available_llm_conditions,
)

OUT = REPO / "data/paper/revision/calc/doc_key_counts.tsv"


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def score_condition(condition: str, seed: int | None):
    gold_rows = list(read_jsonl(GOLD_JSONL))
    pred_rows = []
    for f in pred_files(condition, seed):
        pred_rows += list(read_jsonl(f))
    ids, pairs = _align_by_id_events(gold_rows, pred_rows, gold_key="annotation",
                                     pred_key="pred_annotation", strict_ids=True)
    assert len(ids) == 96, (condition, seed, len(ids))
    folds = fold_of_doc(seed, condition) if condition in ("ft_mock", "ft_all") else {}
    rows = []
    for _id, (g, p) in zip(ids, pairs):
        gold_total, pred_total, tp_items, by_key = score_one_with_key_counts(g, p, pd_match_mode="strict")
        # consistency: doc-level totals equal sums over keys
        assert sum(v["tp"] for v in by_key.values()) == tp_items, (condition, seed, _id)
        assert sum(v["tp"] + v["fn"] for v in by_key.values()) == gold_total
        assert sum(v["tp"] + v["fp"] for v in by_key.values()) == pred_total
        for key, rec in sorted(by_key.items()):
            rows.append({"condition": condition, "seed": "" if seed is None else seed,
                         "fold": folds.get(_id, ""), "doc_id": _id, "key": key,
                         "tp": rec["tp"], "fp": rec["fp"], "fn": rec["fn"]})
    return rows


def main():
    all_rows = []
    all_rows += score_condition("zero_shot", None)
    all_rows += score_condition("few_shot", None)
    for cond in available_llm_conditions():
        all_rows += score_condition(cond, None)
        print(f"done {cond}", file=sys.stderr)
    for seed in EXP08_RUNS:
        for cond in ("ft_synthetic", "ft_mock", "ft_all"):
            all_rows += score_condition(cond, seed)
            print(f"done {cond} seed={seed}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "fold", "doc_id", "key", "tp", "fp", "fn"], delimiter="\t")
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
