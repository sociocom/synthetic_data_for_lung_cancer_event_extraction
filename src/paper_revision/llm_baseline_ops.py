# -*- coding: utf-8 -*-
"""Operational metrics of the exp09 prompted-LLM baselines (for Methods / supplementary table).

Reads each registered run's inference/pred_raw.jsonl and run_env.txt, writes
data/paper/revision/calc/llm_baselines_ops.tsv with: docs, final-channel parse failures,
empty predictions, max_new_tokens hits, generated tokens (mean/max), seconds per document (mean/median/min/max).
"""
from __future__ import annotations
import json
import statistics as st
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paper_revision.runs import LLM_BASELINE_PREDS, CONDITION_LABELS, REPO  # noqa: E402

CALC = REPO / "data/paper/revision/calc"
rows = []
for cond, pred in LLM_BASELINE_PREDS.items():
    if pred is None or not Path(pred).exists():
        continue
    recs = [json.loads(l) for l in open(pred, encoding="utf-8") if l.strip()]
    run_dir = Path(pred).parents[1]
    env = dict(l.strip().split("=", 1) for l in open(run_dir / "run_env.txt", encoding="utf-8") if "=" in l)
    secs = [r["gen_seconds"] for r in recs]
    toks = [r["n_gen_tokens"] for r in recs]
    rows.append({
        "condition": CONDITION_LABELS[cond], "run_dir": str(run_dir.relative_to(REPO)),
        "model_id": env.get("MODEL_ID", ""), "reasoning_effort": env.get("REASONING_EFFORT", ""),
        "gpus": env.get("CUDA_VISIBLE_DEVICES", ""), "docs": len(recs),
        "final_channel_missing": sum(1 for r in recs if not r.get("final_channel_found", True)),
        "empty_prediction": sum(1 for r in recs if json.loads(r["pred_annotation"]) == []),
        "hit_max_new_tokens": sum(1 for r in recs if r.get("hit_max_new_tokens")),
        "gen_tokens_mean": st.mean(toks), "gen_tokens_max": max(toks),
        "sec_per_doc_mean": st.mean(secs), "sec_per_doc_median": st.median(secs),
        "sec_per_doc_min": min(secs), "sec_per_doc_max": max(secs),
        "tok_per_sec": sum(toks) / sum(secs),
    })
df = pd.DataFrame(rows)
df.to_csv(CALC / "llm_baselines_ops.tsv", sep="\t", index=False, float_format="%.2f")
print(df.round(2).to_string(index=False) if len(df) else "no registered runs")
