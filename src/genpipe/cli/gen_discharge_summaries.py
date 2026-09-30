#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

from genpipe.common.runner import run_job, JobConfig
from genpipe.common.llm import LLMConfig
from genpipe.common.icl import SingleTextICL


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate discharge_summary from case_report_ja (JSONL) with ICL (single examples)."
    )
    p.add_argument("--input_jsonl", required=True, type=str)
    p.add_argument("--output_jsonl", required=True, type=str)
    p.add_argument("--prompt_file", required=True, type=str)  # TOML only

    p.add_argument("--context_dir", type=str, default="../data/raw/contexts/discharge_summary_20")
    p.add_argument("--n_context", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    p.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--model", type=str, default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_retry", type=int, default=int(os.getenv("MAX_RETRY", "2")))
    p.add_argument("--max_workers", type=int, default=int(os.getenv("MAX_WORKERS", "100")))

    # プロンプトがどうなっているかの確認ができる。nは何件分表示するかを示す。
    p.add_argument("--dry_run", action="store_true",)
    p.add_argument("--dry_run_n", type=int, default=5,)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="Skip ids already in output (default).")
    g.add_argument("--no_resume", action="store_true")
    p.set_defaults(resume=True)
    return p.parse_args()


def main():
    args = parse_args()

    job = JobConfig(
        input_id_key="id",
        input_text_key="case_report_ja",
        output_text_key="discharge_summary",
        use_context=True,
    )

    icl = SingleTextICL(context_dir=args.context_dir, n=args.n_context, seed=args.seed)

    llm_cfg = LLMConfig(
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retry=args.max_retry,
    )

    resume = False if args.no_resume else args.resume

    if args.dry_run:
        # Print prompts without calling the LLM.
        from genpipe.common.jsonl_io import iter_jsonl, load_done_ids
        from genpipe.common.prompts import load_prompts_toml, render_user_prompt

        prompts = load_prompts_toml(args.prompt_file)
        done_ids = load_done_ids(args.output_jsonl, id_key=job.input_id_key) if resume else set()

        n_printed = 0
        total = 0
        skipped = 0
        missing = 0

        for obj in iter_jsonl(args.input_jsonl):
            total += 1
            if job.input_id_key not in obj or job.input_text_key not in obj:
                missing += 1
                continue

            rid = str(obj[job.input_id_key])
            text = obj[job.input_text_key]
            if text is None:
                missing += 1
                continue
            text = str(text)

            if resume and rid in done_ids:
                skipped += 1
                continue

            ctx = icl.context_for(rid)
            user_prompt = render_user_prompt(prompts.user, input_text=text, context_text=ctx)

            print("=" * 100)
            print(f"[DRY_RUN] id={rid}")
            print("\n--- system ---\n")
            print(prompts.system)
            print("\n--- user (rendered) ---\n")
            print(user_prompt)

            n_printed += 1
            if n_printed >= max(1, int(args.dry_run_n)):
                break

        print(
            f"\n[DRY_RUN] printed={n_printed}, total_scanned={total}, skipped_existing={skipped}, missing_fields={missing}"
        )
        return

    run_job(
        job=job,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        prompt_file=args.prompt_file,
        llm_cfg=llm_cfg,
        max_workers=args.max_workers,
        resume=resume,
        context_provider=icl,
    )


if __name__ == "__main__":
    main()
