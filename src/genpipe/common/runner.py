# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .jsonl_io import iter_jsonl, append_jsonl, load_done_ids, sort_jsonl_inplace_by_numeric_id
from .prompts import load_prompts_toml, render_user_prompt
from .llm import LLMConfig, make_client, call_once


class ContextProvider(Protocol):
    def context_for(self, record_id: str) -> str: ...


@dataclass(frozen=True)
class JobConfig:
    # fixed input/output keys
    input_id_key: str = "id"
    input_text_key: str = "text"
    output_text_key: str = "output"

    # ICL
    use_context: bool = False


def run_job(
    job: JobConfig,
    input_jsonl: str,
    output_jsonl: str,
    prompt_file: str,
    llm_cfg: LLMConfig,
    max_workers: int,
    resume: bool,
    context_provider: Optional[ContextProvider] = None,
    progress_every: int = 50,
) -> None:
    api_key = os.getenv(llm_cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key env var is not set: {llm_cfg.api_key_env}")

    prompts = load_prompts_toml(prompt_file)
    client: OpenAI = make_client(api_key=api_key, base_url=llm_cfg.base_url)

    done_ids = load_done_ids(output_jsonl, id_key=job.input_id_key) if resume else set()

    tasks: List[Tuple[str, str]] = []
    total = 0
    skipped = 0
    missing = 0

    for obj in iter_jsonl(input_jsonl):
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

        tasks.append((rid, text))

    if not tasks:
        print(f"[INFO] No tasks to run. total={total}, skipped_existing={skipped}, missing_fields={missing}")
        return

    print(
        f"[INFO] total={total}, to_run={len(tasks)}, skipped_existing={skipped}, missing_fields={missing}, "
        f"workers={max_workers}, model={llm_cfg.model}"
    )

    def _one(record_id: str, input_text: str) -> Tuple[str, str]:
        ctx = None
        if job.use_context:
            if context_provider is None:
                raise RuntimeError("job.use_context=True but context_provider is None")
            ctx = context_provider.context_for(record_id)

        user_prompt = render_user_prompt(prompts.user, input_text=input_text, context_text=ctx)
        content = call_once(
            client=client,
            cfg=llm_cfg,
            system_prompt=prompts.system,
            user_prompt=user_prompt,
            cid_for_log=record_id,
        )
        return record_id, content

    ok = 0
    ng = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, rid, text) for rid, text in tasks]

        for fut in as_completed(futures):
            try:
                rid, content = fut.result()
                append_jsonl(output_jsonl, {job.input_id_key: rid, job.output_text_key: content})
                ok += 1
                if progress_every > 0 and ok % progress_every == 0:
                    print(f"[INFO] progress ok={ok}, ng={ng}")
            except Exception as e:
                ng += 1
                print(f"[ERR] {e}")

    # Make final output deterministic: sort by numeric id ascending.
    # This keeps max throughput during generation (write as completed),
    # then performs an external sort and atomically replaces the file.
    try:
        sort_jsonl_inplace_by_numeric_id(output_jsonl, id_key=job.input_id_key)
    except Exception as e:
        print(f"[WARN] Failed to sort output JSONL: {e}")

    print(f"[DONE] ok={ok}, ng={ng}, output={output_jsonl}")
