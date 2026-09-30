# -*- coding: utf-8 -*-
"""Re-derive pred_annotation for an exp09 run from the stored raw harmony output.

Usage: python src/paper_revision/reparse_llm_preds.py <run_dir> [--strict]
Writes <run_dir>/inference/pred_reparsed.jsonl (default: final channel, else lenient JSON
extraction over the whole output = same as the current pipeline) or pred_strict.jsonl
(--strict: no final channel -> empty prediction). Prints how many documents used the fallback.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from inferpipe.postprocess import canonicalize_prediction_json, harmony_prediction_text, extract_final_channel  # noqa: E402

run_dir = Path(sys.argv[1]); strict = "--strict" in sys.argv
src = run_dir / "inference/pred_raw.jsonl"
out = run_dir / ("inference/pred_strict.jsonl" if strict else "inference/pred_reparsed.jsonl")
n = fb = empty = 0
with open(out, "w", encoding="utf-8") as w:
    for line in open(src, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); n += 1
        raw = r["raw_pred_annotation"]
        text, found = (extract_final_channel(raw) if strict else harmony_prediction_text(raw))
        fb += (not found)
        r["pred_annotation"] = canonicalize_prediction_json(text, max_events=64)
        r["final_channel_found"] = bool(found)
        empty += (r["pred_annotation"] == "[]")
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{out}: docs={n} no_final_channel={fb} empty_prediction={empty} mode={'strict' if strict else 'fallback'}")
