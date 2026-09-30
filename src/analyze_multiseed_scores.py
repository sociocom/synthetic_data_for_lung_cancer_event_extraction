#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = {
    "FT-Synthetic": {
        "score": "ft_synthetic/score.json",
        "per_key": "ft_synthetic/score_per_key.json",
    },
    "FT-Mock": {
        "score": "ft_mock/score_hungarian_cv.json",
        "per_key": "ft_mock/score_hungarian_cv_per_key.json",
    },
    "FT-ALL": {
        "score": "ft_all/score_hungarian_cv.json",
        "per_key": "ft_all/score_hungarian_cv_per_key.json",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)) if len(xs) >= 2 else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _macro_f1(per_key_obj: dict[str, Any]) -> float:
    vals = [float(v.get("f1", 0.0)) for v in per_key_obj.get("per_key", {}).values()]
    return float(statistics.mean(vals)) if vals else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multiseed score JSON files.")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(p for p in root.glob("seed=*") if p.is_dir())
    if not seed_dirs:
        raise RuntimeError(f"No seed=* directories found under: {root}")

    overall_rows = []
    per_key_values: dict[tuple[str, str], list[dict[str, float]]] = {}

    for condition, rels in CONDITIONS.items():
        for seed_dir in seed_dirs:
            seed = seed_dir.name.split("=", 1)[1]
            score_path = seed_dir / rels["score"]
            per_key_path = seed_dir / rels["per_key"]
            if not score_path.exists() or not per_key_path.exists():
                continue
            score = _load_json(score_path)
            per_key_obj = _load_json(per_key_path)
            summary = score.get("summary", {})
            macro_f1 = _macro_f1(per_key_obj)
            overall_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "micro_precision": float(summary.get("micro_precision", 0.0)),
                    "micro_recall": float(summary.get("micro_recall", 0.0)),
                    "micro_f1": float(summary.get("micro_f1", 0.0)),
                    "macro_f1": macro_f1,
                    "score_json": str(score_path),
                    "per_key_json": str(per_key_path),
                }
            )
            for key, rec in per_key_obj.get("per_key", {}).items():
                per_key_values.setdefault((condition, key), []).append(
                    {
                        "precision": float(rec.get("precision", 0.0)),
                        "recall": float(rec.get("recall", 0.0)),
                        "f1": float(rec.get("f1", 0.0)),
                        "gold_items": float(rec.get("gold_items", 0.0)),
                        "pred_items": float(rec.get("pred_items", 0.0)),
                    }
                )

    with (out_dir / "overall_by_seed.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "condition",
            "seed",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "score_json",
            "per_key_json",
        ]
        wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        wr.writerows(overall_rows)

    overall_summary = []
    for condition in CONDITIONS:
        rows = [r for r in overall_rows if r["condition"] == condition]
        if not rows:
            continue
        out = {"condition": condition, "num_seeds": len(rows)}
        for metric in ["micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
            st = _stats([float(r[metric]) for r in rows])
            for name, val in st.items():
                out[f"{metric}_{name}"] = val
        overall_summary.append(out)

    with (out_dir / "overall_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["condition", "num_seeds"]
        for metric in ["micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
            for name in ["mean", "std", "min", "max"]:
                fieldnames.append(f"{metric}_{name}")
        wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        wr.writerows(overall_summary)

    per_key_rows = []
    for (condition, key), vals in sorted(per_key_values.items()):
        row = {"condition": condition, "item": key, "num_seeds": len(vals)}
        for metric in ["precision", "recall", "f1", "gold_items", "pred_items"]:
            st = _stats([v[metric] for v in vals])
            for name, val in st.items():
                row[f"{metric}_{name}"] = val
        per_key_rows.append(row)

    with (out_dir / "per_key_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["condition", "item", "num_seeds"]
        for metric in ["precision", "recall", "f1", "gold_items", "pred_items"]:
            for name in ["mean", "std", "min", "max"]:
                fieldnames.append(f"{metric}_{name}")
        wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        wr.writerows(per_key_rows)

    print(f"[OK] wrote multiseed summaries to {out_dir}")


if __name__ == "__main__":
    main()
