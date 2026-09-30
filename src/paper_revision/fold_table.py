# -*- coding: utf-8 -*-
"""Per-fold micro-F1 and fold sizes for the CV conditions (FT-Manual, FT-ALL), 5 seeds x 5 folds."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paper_revision.runs import EXCLUDED_KEYS, CONDITION_LABELS, REPO
CALC = REPO / "data/paper/revision/calc"

df = pd.read_csv(CALC / "doc_key_counts.tsv", sep="\t", dtype={"seed": str, "fold": str}, keep_default_na=False)
df = df[~df.key.isin(EXCLUDED_KEYS) & df.condition.isin(["ft_mock", "ft_all"])]
rows = []
for (cond, seed, fold), g in df.groupby(["condition", "seed", "fold"]):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    n_eval = g.doc_id.nunique()
    rows.append({"condition": CONDITION_LABELS[cond], "seed": seed, "fold": int(fold), "n_train": 96 - n_eval, "n_eval": n_eval,
                 "gold_items": int(tp + fn), "micro_precision": p, "micro_recall": r, "micro_f1": 2 * p * r / (p + r) if p + r else 0})
out = pd.DataFrame(rows).sort_values(["condition", "seed", "fold"])
out.to_csv(CALC / "per_fold.tsv", sep="\t", index=False, float_format="%.6f")
# fold assignment identical across seeds?
assign = df[df.condition == "ft_all"].groupby(["seed", "doc_id"]).fold.first().unstack(0)
same = (assign.nunique(axis=1) == 1).all()
print("fold assignment identical across seeds:", same)
print(out.pivot_table(index=["condition", "seed"], columns="fold", values="micro_f1").round(3))
print(out.groupby("condition").micro_f1.agg(["mean", "std", "min", "max"]).round(4))
