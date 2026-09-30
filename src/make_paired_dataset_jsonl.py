#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from trainpipe.common.jsonl_io import save_jsonl
from trainpipe.sft.data import build_pairs_by_id, load_annotation_dict, load_progress_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join progress and annotation JSONLs by id.")
    p.add_argument("--progress_jsonl", required=True)
    p.add_argument("--annotations_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
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
    rows = [
        {
            args.id_key: _id,
            args.progress_key: note,
            args.annotation_key: annotation,
        }
        for _id, note, annotation in pairs
    ]
    save_jsonl(rows, args.output_jsonl)
    print(f"[OK] saved paired JSONL: {args.output_jsonl} rows={len(rows)}")


if __name__ == "__main__":
    main()
