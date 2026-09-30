#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np

from trainpipe.common.jsonl_io import save_json, save_jsonl
from trainpipe.sft.data import (
    build_pairs_by_id,
    cv_train_val_test_indices,
    load_annotation_dict,
    load_progress_dict,
)


def _rows_from_pairs(
    pairs: list[tuple[str, str, str]],
    idxs: Iterable[int],
    id_prefix: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for i in idxs:
        _id, note, ann = pairs[int(i)]
        rid = f"{id_prefix}{_id}" if id_prefix else _id
        rows.append(
            {
                "id": rid,
                "progress_note": note,
                "annotation": ann,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare one CV fold datasets_raw for mixed training (synthetic + simulation)."
    )
    p.add_argument("--sim_progress_jsonl", type=str, required=True)
    p.add_argument("--sim_annotations_jsonl", type=str, required=True)
    p.add_argument("--synth_progress_jsonl", type=str, required=True)
    p.add_argument("--synth_annotations_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True, help="Fold output dir (datasets_raw is created under this path)")

    p.add_argument("--id_key", type=str, default="id")
    p.add_argument("--progress_key", type=str, default="progress_note")
    p.add_argument("--annotation_key", type=str, default="annotation")

    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--cv_fold", type=int, required=True)
    p.add_argument("--cv_split_seed", type=int, default=42)
    p.add_argument("--cv_val_size", type=int, default=14)
    p.add_argument("--cv_val_seed_base", type=int, default=42)

    p.add_argument("--mix_synth_ratio", type=int, default=50)
    p.add_argument("--mix_sim_ratio", type=int, default=50)
    p.add_argument("--mix_seed_base", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mix_synth_ratio < 0 or args.mix_sim_ratio <= 0:
        raise ValueError("mix_synth_ratio must be >= 0 and mix_sim_ratio must be > 0")

    sim_progress = load_progress_dict(args.sim_progress_jsonl, args.id_key, args.progress_key)
    sim_ann = load_annotation_dict(args.sim_annotations_jsonl, args.id_key, args.annotation_key)
    synth_progress = load_progress_dict(args.synth_progress_jsonl, args.id_key, args.progress_key)
    synth_ann = load_annotation_dict(args.synth_annotations_jsonl, args.id_key, args.annotation_key)

    sim_pairs = build_pairs_by_id(sim_progress, sim_ann, max_samples=0)
    synth_pairs = build_pairs_by_id(synth_progress, synth_ann, max_samples=0)

    sim_n = len(sim_pairs)
    train_idx, val_idx, test_idx = cv_train_val_test_indices(
        n=sim_n,
        num_folds=int(args.num_folds),
        test_fold=int(args.cv_fold),
        split_seed=int(args.cv_split_seed),
        val_size=int(args.cv_val_size),
        val_seed_base=int(args.cv_val_seed_base),
    )

    sim_train_rows = _rows_from_pairs(sim_pairs, train_idx)
    val_rows = _rows_from_pairs(sim_pairs, val_idx)
    test_rows = _rows_from_pairs(sim_pairs, test_idx)
    synth_pool_n = len(synth_pairs)
    synth_pool_rows = _rows_from_pairs(synth_pairs, range(synth_pool_n), id_prefix="synth::")

    synth_target = int(round(len(sim_train_rows) * float(args.mix_synth_ratio) / float(args.mix_sim_ratio)))
    synth_target = max(0, synth_target)
    synth_take = min(synth_target, synth_pool_n)

    sample_seed = int(args.mix_seed_base) + int(args.cv_fold)
    rng = np.random.default_rng(sample_seed)
    synth_choice = rng.choice(synth_pool_n, size=synth_take, replace=False) if synth_take > 0 else []
    synth_rows = [synth_pool_rows[int(i)] for i in synth_choice] if synth_take > 0 else []

    train_rows = sim_train_rows + synth_rows
    if train_rows:
        perm = rng.permutation(len(train_rows))
        train_rows = [train_rows[int(i)] for i in perm]

    ds_dir = os.path.join(args.output_dir, "datasets_raw")
    save_jsonl(sim_train_rows, os.path.join(ds_dir, "train_sim.jsonl"))
    save_jsonl(synth_pool_rows, os.path.join(ds_dir, "synth_pool.jsonl"))
    save_jsonl(train_rows, os.path.join(ds_dir, "train.jsonl"))
    save_jsonl(val_rows, os.path.join(ds_dir, "validation.jsonl"))
    save_jsonl(test_rows, os.path.join(ds_dir, "test.jsonl"))

    meta = {
        "split_mode": "mix_cv_prepared",
        "cv": {
            "num_folds": int(args.num_folds),
            "cv_fold(test_fold)": int(args.cv_fold),
            "cv_split_seed": int(args.cv_split_seed),
            "cv_val_size": int(args.cv_val_size),
            "cv_val_seed_base": int(args.cv_val_seed_base),
        },
        "mix": {
            "mix_synth_ratio": int(args.mix_synth_ratio),
            "mix_sim_ratio": int(args.mix_sim_ratio),
            "mix_seed_base": int(args.mix_seed_base),
            "sample_seed": int(sample_seed),
        },
        "counts": {
            "sim_total": sim_n,
            "synth_total": synth_pool_n,
            "sim_train": len(sim_train_rows),
            "sim_validation": len(val_rows),
            "sim_test": len(test_rows),
            "synth_pool": len(synth_pool_rows),
            "synth_target_from_ratio": int(synth_target),
            "synth_sampled": len(synth_rows),
            "mixed_train_total": len(train_rows),
        },
        "inputs": {
            "sim_progress_jsonl": args.sim_progress_jsonl,
            "sim_annotations_jsonl": args.sim_annotations_jsonl,
            "synth_progress_jsonl": args.synth_progress_jsonl,
            "synth_annotations_jsonl": args.synth_annotations_jsonl,
        },
    }
    save_json(meta, os.path.join(args.output_dir, "mix_meta.json"))

    print(f"[OK] prepared fold={args.cv_fold} -> {args.output_dir}")
    print(
        "[COUNT] "
        f"sim_train={len(sim_train_rows)} sim_val={len(val_rows)} sim_test={len(test_rows)} "
        f"synth_sampled={len(synth_rows)} mixed_train={len(train_rows)}"
    )


if __name__ == "__main__":
    main()
