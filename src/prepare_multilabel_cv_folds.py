#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.eval_schema import EVAL_KEY_TYPES, canonicalize_eval_key
from src.normalize import extract_json_from_text_lenient
from trainpipe.sft.data import build_pairs_by_id, load_annotation_dict, load_progress_dict


def _parse_annotation(value: str, *, where: str) -> list[dict[str, Any]]:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        obj = extract_json_from_text_lenient(value)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    raise ValueError(f"annotation must be dict/list at {where}")


def _keys_present(annotation: str, *, where: str) -> set[str]:
    out: set[str] = set()
    for event in _parse_annotation(annotation, where=where):
        for raw_key, raw_value in event.items():
            key = canonicalize_eval_key(str(raw_key))
            if key not in EVAL_KEY_TYPES:
                continue
            if raw_value is None:
                continue
            if isinstance(raw_value, str) and not raw_value.strip():
                continue
            if isinstance(raw_value, list) and len(raw_value) == 0:
                continue
            out.add(key)
    return out


def _make_folds(y: np.ndarray, *, n_splits: int, seed: int) -> list[list[int]]:
    x_dummy = np.zeros((y.shape[0], 1), dtype=int)
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds: list[list[int]] = []
    for _train_idx, test_idx in splitter.split(x_dummy, y):
        folds.append(sorted(int(i) for i in test_idx.tolist()))
    return folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create note-level CV folds with iterative multilabel stratification by item presence."
    )
    p.add_argument("--progress_jsonl", required=True)
    p.add_argument("--annotations_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id_key", default="id")
    p.add_argument("--progress_key", default="progress_note")
    p.add_argument("--annotation_key", default="annotation")
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    progress = load_progress_dict(args.progress_jsonl, args.id_key, args.progress_key)
    ann = load_annotation_dict(args.annotations_jsonl, args.id_key, args.annotation_key)
    pairs = build_pairs_by_id(progress, ann, max_samples=args.max_samples)
    keys = sorted(EVAL_KEY_TYPES.keys())

    rows = []
    y = np.zeros((len(pairs), len(keys)), dtype=int)
    for i, (_id, _note, annotation) in enumerate(pairs):
        present = _keys_present(annotation, where=f"id={_id}")
        for key in present:
            y[i, keys.index(key)] = 1
        rows.append({"id": _id, "keys_present": sorted(present), "num_keys_present": len(present)})

    folds_idx = _make_folds(y, n_splits=int(args.num_folds), seed=int(args.seed))
    ids = [_id for _id, _note, _ann in pairs]

    folds = []
    for fold, idxs in enumerate(folds_idx):
        counts = Counter()
        for i in idxs:
            counts.update(rows[i]["keys_present"])
        folds.append(
            {
                "fold": fold,
                "test_indices": idxs,
                "test_ids": [ids[i] for i in idxs],
                "num_examples": len(idxs),
                "item_document_counts": dict(sorted(counts.items())),
            }
        )

    out = {
        "summary": {
            "split_mode": "iterative_multilabel_stratified",
            "stratifier": "iterstrat.ml_stratifiers.MultilabelStratifiedKFold",
            "num_examples": len(pairs),
            "num_folds": int(args.num_folds),
            "seed": int(args.seed),
            "num_items": len(keys),
            "note": "Folds are stratified by document-level presence of each evaluation item.",
        },
        "inputs": {
            "progress_jsonl": args.progress_jsonl,
            "annotations_jsonl": args.annotations_jsonl,
            "id_key": args.id_key,
            "progress_key": args.progress_key,
            "annotation_key": args.annotation_key,
            "max_samples": int(args.max_samples),
        },
        "item_keys": keys,
        "item_document_counts_total": dict(
            sorted((key, int(y[:, j].sum())) for j, key in enumerate(keys))
        ),
        "ids_sorted": ids,
        "rows": rows,
        "folds": folds,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved folds: {out_path}")


if __name__ == "__main__":
    main()
