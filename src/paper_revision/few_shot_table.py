# -*- coding: utf-8 -*-
"""Few-shot (SLM) reference baseline: overall and per-item F1 on the 37-item protocol."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paper_revision.runs import EXCLUDED_KEYS, REPO
CALC = REPO / "data/paper/revision/calc"
df = pd.read_csv(CALC / "doc_key_counts.tsv", sep="\t", dtype={"seed": str, "fold": str}, keep_default_na=False)
df = df[~df.key.isin(EXCLUDED_KEYS)]
def prf(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)
rows = []
for cond in ["zero_shot", "few_shot"]:
    g = df[df.condition == cond]
    p, r, f = prf(g)
    k = g.groupby("key").apply(lambda x: prf(x)[2])
    rows.append({"condition": cond, "micro_precision": p, "micro_recall": r, "micro_f1": f, "macro_f1": k.mean(), "n_keys": len(k)})
pd.DataFrame(rows).to_csv(CALC / "few_shot_overall.tsv", sep="\t", index=False, float_format="%.4f")
per = []
for key, g in df[df.condition == "few_shot"].groupby("key"):
    p, r, f = prf(g)
    per.append({"item": key, "reference_instances": int(g.tp.sum() + g.fn.sum()), "precision": p, "recall": r, "f1": f})
pd.DataFrame(per).to_csv(CALC / "few_shot_per_item.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(rows).round(4).to_string(index=False))
