#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time

import torch

from genpipe.common.icl import PairedICL
from inferpipe.args import parse_args
from inferpipe.generation import generate_batch
from inferpipe.io import collect_rows, ensure_output_parent, load_done_ids, slice_range
from inferpipe.modeling import load_model_and_tokenizer, load_tokenizer
from inferpipe.postprocess import canonicalize_prediction_json, harmony_prediction_text
from inferpipe.prompting import (
    build_chat_text,
    build_structured_prompt,
    load_prompts_toml,
    read_text_file,
    render_user_prompt,
)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")

    use_adapter = bool(args.use_adapter)
    if args.no_adapter:
        use_adapter = False
    if use_adapter and not args.adapter_path:
        raise ValueError("--use_adapter requires --adapter_path (or ADAPTER_PATH env)")

    done_ids = load_done_ids(args.output_jsonl, id_key=args.id_key) if args.resume else set()

    rows, missing = collect_rows(
        input_jsonl=args.input_jsonl,
        id_key=args.id_key,
        input_key=args.input_key,
        resume=args.resume,
        done_ids=done_ids,
    )

    s, e = slice_range(len(rows), args.start, args.end)
    rows = rows[s:e]

    use_context = bool(args.icl_input_dir and args.icl_output_dir and int(args.n_context) > 0)
    if use_context and not args.prompt_file:
        raise ValueError("ICL requires --prompt_file so {{CONTEXT}} can be rendered in user template.")

    icl = None
    if use_context:
        icl = PairedICL(
            icl_input_dir=args.icl_input_dir,
            icl_output_dir=args.icl_output_dir,
            n=int(args.n_context),
            seed=int(args.icl_seed),
            input_label=args.icl_input_label,
            output_label=args.icl_output_label,
        )

    # Dry-run: print rendered prompts only (no model generation).
    if args.dry_run:
        n = max(1, int(args.dry_run_n))
        system_prompt = ""
        user_template = ""
        if args.prompt_file:
            system_prompt, user_template = load_prompts_toml(args.prompt_file)
            tokenizer = load_tokenizer(args.model_id)
            mode = f"toml ({args.prompt_file})"
        else:
            guideline = read_text_file(args.guideline_path)
            mode = "structured_template (guideline_path)"

        print(f"[DRY_RUN] mode={mode}, total_slice={len(rows)}, print_n={n}, missing={missing}")
        for i, r in enumerate(rows[:n], start=1):
            if args.prompt_file:
                ctx = icl.context_for(r["id"]) if icl else None
                prompt_text = build_chat_text(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    user_prompt=render_user_prompt(user_template, r["text"], context_text=ctx),
                    reasoning_effort=args.reasoning_effort,
                )
            else:
                prompt_text = build_structured_prompt(guideline, r["text"])

            print("=" * 100)
            print(f"[DRY_RUN] {i}/{min(n, len(rows))} id={r['id']}")
            print(prompt_text)
        return

    ensure_output_parent(args.output_jsonl)

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        use_adapter=use_adapter,
        adapter_path=args.adapter_path,
        device=args.device,
        quant=args.quant,
        gpu_headroom_gb=args.gpu_headroom_gb,
        attn_implementation=args.attn_implementation,
    )
    if args.final_channel:
        print(f"[INFO] final_channel=True reasoning_effort={args.reasoning_effort or '(template default)'}")

    system_prompt = ""
    user_template = ""
    if args.prompt_file:
        system_prompt, user_template = load_prompts_toml(args.prompt_file)
        print(f"[INFO] Prompt mode: toml ({args.prompt_file})")
    else:
        guideline = read_text_file(args.guideline_path)
        print("[INFO] Prompt mode: structured_template (guideline_path)")

    total = len(rows)
    bs = max(1, int(args.batch_size))

    with open(args.output_jsonl, "a", encoding="utf-8") as wf:
        for i in range(0, total, bs):
            chunk = rows[i : i + bs]
            if args.prompt_file:
                prompts = [
                    build_chat_text(
                        tokenizer=tokenizer,
                        system_prompt=system_prompt,
                        user_prompt=render_user_prompt(
                            user_template,
                            r["text"],
                            context_text=(icl.context_for(r["id"]) if icl else None),
                        ),
                        reasoning_effort=args.reasoning_effort,
                    )
                    for r in chunk
                ]
            else:
                prompts = [build_structured_prompt(guideline, r["text"]) for r in chunk]

            t0 = time.time()
            preds, n_gen = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                keep_special_tokens=args.final_channel,
            )
            sec_per_row = (time.time() - t0) / max(1, len(chunk))

            for r, pred, ntok in zip(chunk, preds, n_gen):
                raw_text = pred
                if args.final_channel:
                    pred, found = harmony_prediction_text(raw_text)
                else:
                    found = None
                out_obj = {
                    args.id_key: r["id"],
                    args.input_key: r["text"],
                    args.pred_key: canonicalize_prediction_json(pred, max_events=args.max_events),
                }
                if args.raw_pred_key:
                    out_obj[args.raw_pred_key] = raw_text
                out_obj["n_gen_tokens"] = int(ntok)
                out_obj["gen_seconds"] = round(sec_per_row, 2)
                if found is not None:
                    out_obj["final_channel_found"] = bool(found)
                    out_obj["hit_max_new_tokens"] = bool(ntok >= args.max_new_tokens)
                wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                wf.flush()

            if (i // bs + 1) % 10 == 0:
                print(f"Progress: {min(i+bs, total)}/{total} (slice {s}:{e}), missing={missing}")

    print(f"Done. wrote={total} to {args.output_jsonl} (slice {s}:{e}), missing={missing}")


if __name__ == "__main__":
    main()
