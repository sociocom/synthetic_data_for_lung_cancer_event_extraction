#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create ICL candidate files by sampling from simu_lung_cancer.

- Input: data/raw/simu_lung_cancer/*.txt (progress notes) and *.json (annotations)
- Output:
  - data/raw/contexts/simu_progress_notes/01_*.txt .. 05_*.txt
  - data/raw/contexts/simu_annotation/01_*.json .. 05_*.json

Optionally also export the remaining (non-ICL) pairs to:
  - data/processed/simulation/simu_progress_notes.jsonl
  - data/processed/simulation/simu_annotation.jsonl

This is intended to carve out a small ICL pool from an evaluation dataset.

Default behavior applies no-AE transform to annotations:
- drop keys: Reaction, ToxicityGrade
- remove events that become empty or Timestamp-only
- write transformed annotations to copied context .json and remaining JSONL
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


DEFAULT_DROP_KEYS = {"Reaction", "ToxicityGrade"}
DEFAULT_TIMESTAMP_KEY = "Timestamp"


def _pair_stems(input_dir: Path) -> list[str]:
    """Return stems that have both .txt and .json."""
    txts = {p.stem for p in input_dir.glob("*.txt")}
    jsons = {p.stem for p in input_dir.glob("*.json")}
    stems = sorted(txts & jsons)
    return stems


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _parse_annotation_events(obj: object) -> list[dict]:
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _transform_noae(
    events: list[dict],
    *,
    drop_keys: set[str],
    timestamp_key: str,
) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        cleaned = {k: v for k, v in ev.items() if k not in drop_keys}
        keys = set(cleaned.keys())
        if not keys or keys == {timestamp_key}:
            continue
        out.append(cleaned)
    return out


def _load_annotation_text(
    src_json: Path,
    *,
    apply_noae: bool,
    drop_keys: set[str],
    timestamp_key: str,
    pretty: bool,
) -> str:
    if not apply_noae:
        return src_json.read_text(encoding="utf-8")

    raw_obj = json.loads(src_json.read_text(encoding="utf-8"))
    events = _parse_annotation_events(raw_obj)
    transformed = _transform_noae(events, drop_keys=drop_keys, timestamp_key=timestamp_key)
    if pretty:
        return json.dumps(transformed, ensure_ascii=False, indent=4) + "\n"
    return json.dumps(transformed, ensure_ascii=False)


def _copy_pair(
    *,
    input_dir: Path,
    stem: str,
    out_progress_dir: Path,
    out_annotation_dir: Path,
    prefix: str = "",
    overwrite: bool = False,
    apply_noae: bool = True,
    drop_keys: set[str] | None = None,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> None:
    src_txt = input_dir / f"{stem}.txt"
    src_json = input_dir / f"{stem}.json"
    if not src_txt.exists() or not src_json.exists():
        raise FileNotFoundError(f"Missing pair for stem={stem}: {src_txt} / {src_json}")

    dst_txt = out_progress_dir / f"{prefix}{stem}.txt"
    dst_json = out_annotation_dir / f"{prefix}{stem}.json"

    if not overwrite:
        if dst_txt.exists() or dst_json.exists():
            raise FileExistsError(
                f"Destination already exists (use --overwrite): {dst_txt} or {dst_json}"
            )

    shutil.copyfile(src_txt, dst_txt)

    used_drop_keys = DEFAULT_DROP_KEYS if drop_keys is None else drop_keys
    ann_text = _load_annotation_text(
        src_json,
        apply_noae=apply_noae,
        drop_keys=used_drop_keys,
        timestamp_key=timestamp_key,
        pretty=True,
    )
    dst_json.write_text(ann_text, encoding="utf-8")


def _write_remaining_jsonl(
    *,
    input_dir: Path,
    stems: list[str],
    out_dir: Path,
    overwrite: bool,
    apply_noae: bool,
    drop_keys: set[str],
    timestamp_key: str,
) -> tuple[Path, Path, int]:
    """Write remaining pairs as two JSONL files.

    - simu_progress_notes.jsonl: {"id": <stem>, "progress_note": <txt content>}
    - simu_annotation.jsonl:     {"id": <stem>, "annotation": <json content (raw text)>}

    Returns: (progress_jsonl_path, annotation_jsonl_path, count)
    """

    _ensure_dir(out_dir)

    progress_path = out_dir / "simu_progress_notes.jsonl"
    annotation_path = out_dir / "simu_annotation.jsonl"

    if not overwrite:
        if progress_path.exists() or annotation_path.exists():
            raise FileExistsError(
                "Destination already exists (use --overwrite): "
                f"{progress_path} or {annotation_path}"
            )

    count = 0
    with progress_path.open("w", encoding="utf-8") as f_prog, annotation_path.open(
        "w", encoding="utf-8"
    ) as f_ann:
        for stem in stems:
            src_txt = input_dir / f"{stem}.txt"
            src_json = input_dir / f"{stem}.json"
            if not src_txt.exists() or not src_json.exists():
                raise FileNotFoundError(
                    f"Missing pair for stem={stem}: {src_txt} / {src_json}"
                )

            progress_note = src_txt.read_text(encoding="utf-8")
            annotation = _load_annotation_text(
                src_json,
                apply_noae=apply_noae,
                drop_keys=drop_keys,
                timestamp_key=timestamp_key,
                pretty=False,
            )

            f_prog.write(
                json.dumps(
                    {"id": stem, "progress_note": progress_note},
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_ann.write(
                json.dumps(
                    {"id": stem, "annotation": annotation},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return progress_path, annotation_path, count


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sample N paired (.txt, .json) examples from simu_lung_cancer and copy to contexts dirs with 01_ prefix. "
            "Optionally export remaining pairs to data/processed/simulation as JSONL."
        )
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("../data/raw/simu_lung_cancer"),
        help="Directory containing paired .txt/.json files (default: ../data/raw/simu_lung_cancer)",
    )
    ap.add_argument(
        "--out_progress_dir",
        type=Path,
        default=Path("../data/raw/contexts/simu_progress_notes"),
        help="Output dir for sampled progress note .txt (default: ../data/raw/contexts/simu_progress_notes)",
    )
    ap.add_argument(
        "--out_annotation_dir",
        type=Path,
        default=Path("../data/raw/contexts/simu_annotation"),
        help="Output dir for sampled annotation .json (default: ../data/raw/contexts/simu_annotation)",
    )

    ap.add_argument(
        "--export_remaining",
        action="store_true",
        help="Also export non-sampled (remaining) pairs to --remaining_out_dir as JSONL",
    )
    ap.add_argument(
        "--remaining_out_dir",
        type=Path,
        default=Path("../data/processed/simulation"),
        help="Output dir for remaining JSONL files (default: ../data/processed/simulation)",
    )

    ap.add_argument("--n", type=int, default=5, help="Number of examples to sample (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files with same names in output dirs",
    )
    ap.add_argument(
        "--disable_noae",
        action="store_true",
        help="Disable default no-AE transform and keep original annotation JSON as-is.",
    )
    ap.add_argument(
        "--drop_keys",
        type=str,
        default="Reaction,ToxicityGrade",
        help="Comma-separated keys to drop in no-AE mode (default: Reaction,ToxicityGrade)",
    )
    ap.add_argument(
        "--timestamp_key",
        type=str,
        default=DEFAULT_TIMESTAMP_KEY,
        help="Timestamp key used when removing Timestamp-only events (default: Timestamp)",
    )

    args = ap.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    stems = _pair_stems(input_dir)
    if not stems:
        raise FileNotFoundError(f"No paired .txt/.json found in: {input_dir}")
    if len(stems) < args.n:
        raise ValueError(f"Need at least n={args.n} pairs, got {len(stems)}")

    apply_noae = not bool(args.disable_noae)
    drop_keys = {k.strip() for k in str(args.drop_keys).split(",") if k.strip()}

    rng = random.Random(args.seed)
    chosen = rng.sample(stems, k=args.n)
    chosen_set = set(chosen)

    out_p = args.out_progress_dir
    out_a = args.out_annotation_dir
    _ensure_dir(out_p)
    _ensure_dir(out_a)

    for i, stem in enumerate(chosen, start=1):
        prefix = f"{i:02d}_"
        _copy_pair(
            input_dir=input_dir,
            stem=stem,
            out_progress_dir=out_p,
            out_annotation_dir=out_a,
            prefix=prefix,
            overwrite=bool(args.overwrite),
            apply_noae=apply_noae,
            drop_keys=drop_keys,
            timestamp_key=str(args.timestamp_key),
        )

    exported_remaining = 0
    remaining_progress_path: Path | None = None
    remaining_annotation_path: Path | None = None

    if args.export_remaining:
        remaining_stems = [s for s in stems if s not in chosen_set]
        remaining_progress_path, remaining_annotation_path, exported_remaining = _write_remaining_jsonl(
            input_dir=input_dir,
            stems=remaining_stems,
            out_dir=args.remaining_out_dir,
            overwrite=bool(args.overwrite),
            apply_noae=apply_noae,
            drop_keys=drop_keys,
            timestamp_key=str(args.timestamp_key),
        )

    print("[OK] sampled stems (ICL pool):")
    for i, stem in enumerate(chosen, start=1):
        print(f"  {i:02d}: {stem}")
    print(f"[OK] wrote ICL progress_notes: {out_p}")
    print(f"[OK] wrote ICL annotations:    {out_a}")
    print(f"[OK] no-AE transform enabled:  {apply_noae}")
    if apply_noae:
        print(f"[OK] drop_keys:               {sorted(drop_keys)}")
        print(f"[OK] timestamp_key:           {args.timestamp_key}")

    if args.export_remaining:
        print(f"[OK] exported remaining pairs: {exported_remaining}")
        if remaining_progress_path is not None and remaining_annotation_path is not None:
            print(f"[OK] wrote remaining progress_notes jsonl: {remaining_progress_path}")
            print(f"[OK] wrote remaining annotations jsonl:    {remaining_annotation_path}")


if __name__ == "__main__":
    main()
