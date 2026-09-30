# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

from trainpipe.common.jsonl_io import iter_jsonl, load_text_file
from trainpipe.sft.prompting import STRUCTURED_TEMPLATE


@dataclass(frozen=True)
class TokenStatsArgs:
    model_id: str
    progress_jsonl: str
    annotations_jsonl: str
    guideline_path: str

    progress_key: str = "progress_note"
    annotation_key: str = "annotation"
    id_key: str = "id"

    max_length: int = 4096
    do_truncate: bool = False
    print_topk: int = 20
    max_samples: int = 0


def _load_dict(path: str, id_key: str, value_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for obj in iter_jsonl(path):
        _id = obj.get(id_key)
        val = obj.get(value_key)
        if _id is None or val is None:
            continue
        _id = str(_id)
        val = str(val)
        if val.strip():
            out[_id] = val
    if not out:
        raise RuntimeError(f"No valid values loaded from {path} using keys id={id_key} value={value_key}")
    return out


def _build_pairs(progress: Dict[str, str], ann: Dict[str, str], max_samples: int) -> List[Tuple[str, str, str]]:
    ids = sorted(set(progress.keys()) & set(ann.keys()))
    if not ids:
        raise RuntimeError("No matched ids between progress_jsonl and annotations_jsonl")
    if max_samples and max_samples > 0:
        ids = ids[:max_samples]
    return [(_id, progress[_id], ann[_id]) for _id in ids]


def compute_token_stats(
    progress_jsonl: str,
    annotations_jsonl: str,
    progress_key: str,
    annotation_key: str,
    id_key: str,
    guideline_path: str,
    model_id: Optional[str] = None,
    max_length: int = 4096,
    do_truncate: bool = False,
    print_topk: int = 20,
    max_samples: int = 0,
) -> None:
    if model_id is None:
        raise ValueError("model_id is required")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("tokenizer.eos_token is None. Please check the tokenizer.")

    guideline = load_text_file(guideline_path)
    template = STRUCTURED_TEMPLATE.replace("<guideline>", guideline)

    progress = _load_dict(progress_jsonl, id_key=id_key, value_key=progress_key)
    ann = _load_dict(annotations_jsonl, id_key=id_key, value_key=annotation_key)
    pairs = _build_pairs(progress, ann, max_samples=max_samples)

    results: List[Tuple[str, int]] = []

    for _id, note, a in pairs:
        prompt = template.replace("<text>", str(note))
        full_text = prompt + str(a) + tokenizer.eos_token

        enc = tokenizer(
            full_text,
            truncation=do_truncate,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        results.append((_id, len(enc["input_ids"])))

    lengths = np.array([x[1] for x in results], dtype=np.int64)

    def pct(p: float) -> int:
        return int(np.percentile(lengths, p))

    min_len = int(lengths.min())
    max_len_v = int(lengths.max())

    min_ids = [rid for rid, tl in results if tl == min_len]
    max_ids = [rid for rid, tl in results if tl == max_len_v]

    print(f"num_samples,{len(results)}")
    print(f"min,{min_len}")
    print(f"min_ids,{';'.join(sorted(min_ids))}")
    print(f"max,{max_len_v}")
    print(f"max_ids,{';'.join(sorted(max_ids))}")
    print(f"mean,{float(lengths.mean()):.2f}")
    print(f"p50,{pct(50)}")
    print(f"p90,{pct(90)}")
    print(f"p95,{pct(95)}")
    print(f"p99,{pct(99)}")
    print(f"do_truncate,{do_truncate}")
    print(f"max_length,{max_length}")

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    k = max(0, min(print_topk, len(results_sorted)))
    if k > 0:
        print("top_longest,id,token_len")
        for rid, tl in results_sorted[:k]:
            print(f"top_longest,{rid},{tl}")


def _parse_args() -> TokenStatsArgs:
    p = argparse.ArgumentParser(description="Count tokens for trainpipe JSONL pairs (prompt+answer+eos).")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--progress_jsonl", type=str, required=True)
    p.add_argument("--annotations_jsonl", type=str, required=True)
    p.add_argument("--guideline_path", type=str, required=True)

    p.add_argument("--progress_key", type=str, default="progress_note")
    p.add_argument("--annotation_key", type=str, default="annotation")
    p.add_argument("--id_key", type=str, default="id")

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--do_truncate", action="store_true")
    p.add_argument("--print_topk", type=int, default=20)
    p.add_argument("--max_samples", type=int, default=0)
    a = p.parse_args()

    return TokenStatsArgs(
        model_id=a.model_id,
        progress_jsonl=a.progress_jsonl,
        annotations_jsonl=a.annotations_jsonl,
        guideline_path=a.guideline_path,
        progress_key=a.progress_key,
        annotation_key=a.annotation_key,
        id_key=a.id_key,
        max_length=a.max_length,
        do_truncate=a.do_truncate,
        print_topk=a.print_topk,
        max_samples=a.max_samples,
    )


def main() -> None:
    args = _parse_args()
    compute_token_stats(
        model_id=args.model_id,
        progress_jsonl=args.progress_jsonl,
        annotations_jsonl=args.annotations_jsonl,
        progress_key=args.progress_key,
        annotation_key=args.annotation_key,
        id_key=args.id_key,
        guideline_path=args.guideline_path,
        max_length=args.max_length,
        do_truncate=args.do_truncate,
        print_topk=args.print_topk,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
