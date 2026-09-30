# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from inferpipe.io import collect_rows, slice_range
from inferpipe.modeling import load_tokenizer
from inferpipe.prompting import (
    build_chat_text,
    build_structured_prompt,
    load_prompts_toml,
    read_text_file,
    render_user_prompt,
)


@dataclass(frozen=True)
class TokenStatsArgs:
    model_id: str
    input_jsonl: str
    id_key: str = "id"
    input_key: str = "progress_note"
    prompt_file: str = ""
    guideline_path: str = ""
    max_input_length: int = 4096
    do_truncate: bool = False
    print_topk: int = 20
    max_samples: int = 0
    start: int = 0
    end: int = 0


def compute_token_stats(args: TokenStatsArgs) -> None:
    tokenizer = load_tokenizer(args.model_id)
    rows, missing = collect_rows(
        input_jsonl=args.input_jsonl,
        id_key=args.id_key,
        input_key=args.input_key,
        resume=False,
        done_ids=set(),
    )

    s, e = slice_range(len(rows), args.start, args.end)
    rows = rows[s:e]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise RuntimeError("No valid rows found. Check input_jsonl/id_key/input_key.")

    prompts: List[Tuple[str, str]] = []
    if args.prompt_file:
        system_prompt, user_template = load_prompts_toml(args.prompt_file)
        for r in rows:
            user_prompt = render_user_prompt(user_template, r["text"])
            prompt_text = build_chat_text(tokenizer, system_prompt, user_prompt)
            prompts.append((r["id"], prompt_text))
        mode = f"toml:{args.prompt_file}"
    else:
        guideline = read_text_file(args.guideline_path)
        for r in rows:
            prompt_text = build_structured_prompt(guideline, r["text"])
            prompts.append((r["id"], prompt_text))
        mode = f"structured:{args.guideline_path or '(empty guideline)'}"

    results: List[Tuple[str, int, int]] = []
    for rid, prompt_text in prompts:
        full_ids = tokenizer(prompt_text, padding=False, truncation=False)["input_ids"]
        full_len = len(full_ids)

        if args.do_truncate:
            tr_ids = tokenizer(
                prompt_text,
                padding=False,
                truncation=True,
                max_length=args.max_input_length,
            )["input_ids"]
            tok_len = len(tr_ids)
        else:
            tok_len = full_len

        over = max(0, full_len - args.max_input_length)
        results.append((rid, tok_len, over))

    lengths = np.array([x[1] for x in results], dtype=np.int64)
    overflows = [x[2] for x in results]
    over_count = sum(1 for v in overflows if v > 0)

    def pct(p: float) -> int:
        return int(np.percentile(lengths, p))

    min_len = int(lengths.min())
    max_len = int(lengths.max())
    min_ids = [rid for rid, tl, _ in results if tl == min_len]
    max_ids = [rid for rid, tl, _ in results if tl == max_len]

    print(f"mode,{mode}")
    print(f"num_samples,{len(results)}")
    print(f"missing_fields,{missing}")
    print(f"slice,{s}:{e}")
    print(f"do_truncate,{args.do_truncate}")
    print(f"max_input_length,{args.max_input_length}")
    print(f"num_over_max_input_length,{over_count}")
    print(f"over_max_ratio,{(over_count / len(results)):.6f}")
    print(f"min,{min_len}")
    print(f"min_ids,{';'.join(sorted(min_ids))}")
    print(f"max,{max_len}")
    print(f"max_ids,{';'.join(sorted(max_ids))}")
    print(f"mean,{float(lengths.mean()):.2f}")
    print(f"p50,{pct(50)}")
    print(f"p90,{pct(90)}")
    print(f"p95,{pct(95)}")
    print(f"p99,{pct(99)}")

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    k = max(0, min(args.print_topk, len(results_sorted)))
    if k > 0:
        print("top_longest,id,token_len,overflow_tokens")
        for rid, tl, ov in results_sorted[:k]:
            print(f"top_longest,{rid},{tl},{ov}")


def _parse_args() -> TokenStatsArgs:
    p = argparse.ArgumentParser(description="Count prompt tokens for inferpipe inference inputs.")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--id_key", type=str, default="id")
    p.add_argument("--input_key", type=str, default="progress_note")
    p.add_argument("--prompt_file", type=str, default="")
    p.add_argument("--guideline_path", type=str, default="")
    p.add_argument("--max_input_length", type=int, default=4096)
    p.add_argument("--do_truncate", action="store_true")
    p.add_argument("--print_topk", type=int, default=20)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    a = p.parse_args()
    return TokenStatsArgs(
        model_id=a.model_id,
        input_jsonl=a.input_jsonl,
        id_key=a.id_key,
        input_key=a.input_key,
        prompt_file=a.prompt_file,
        guideline_path=a.guideline_path,
        max_input_length=a.max_input_length,
        do_truncate=a.do_truncate,
        print_topk=a.print_topk,
        max_samples=a.max_samples,
        start=a.start,
        end=a.end,
    )


def main() -> None:
    args = _parse_args()
    compute_token_stats(args)


if __name__ == "__main__":
    main()
