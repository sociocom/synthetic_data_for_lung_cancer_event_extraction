# -*- coding: utf-8 -*-
"""Measure inference latency and peak GPU memory of the deployed SLM (JMIR revision, Z5/Y3).

Uses the exact inference stack of exp08 (inferpipe: 4-bit NF4 + LoRA adapter, structured guideline
prompt, greedy decoding, batch size 1) on the 96 evaluation records. No training is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch

from inferpipe.generation import generate_batch
from inferpipe.io import collect_rows
from inferpipe.modeling import load_model_and_tokenizer
from inferpipe.prompting import build_structured_prompt, read_text_file
from paper_revision.runs import EXP08_RUNS, REPO


def nvsmi_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", os.environ.get("NVSMI_INDEX", "0")]
    )
    return int(out.decode().strip().splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--condition", default="ft_synthetic", choices=["ft_synthetic", "ft_all_fold0"])
    ap.add_argument("--out_dir", default=str(REPO / "data/paper/revision/calc"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run = EXP08_RUNS[args.seed]
    model_id = "/path/to/models/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5/"
    if args.condition == "ft_synthetic":
        adapter = run / f"seed={args.seed}/ft_synthetic/best"
    else:
        adapter = run / f"seed={args.seed}/ft_all/folds/fold=0/best"
    eval_jsonl = run / "mock_eval_paired.jsonl"
    guideline = read_text_file(str(REPO / "src/prompts/guideline_lung_noae_v2.txt"))

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, use_adapter=True, adapter_path=str(adapter), device="auto")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after_load = {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "nvsmi_used_mib": nvsmi_used_mib(),
    }
    n_trainable = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
    n_total = sum(p.numel() for p in model.parameters())

    rows, _ = collect_rows(input_jsonl=str(eval_jsonl), id_key="id", input_key="progress_note", resume=False, done_ids=set())
    if args.limit:
        rows = rows[: args.limit]

    per_doc = []
    # warm-up (CUDA kernels / bnb) on the first doc, not recorded
    generate_batch(model, tokenizer, [build_structured_prompt(guideline, rows[0]["text"])], 12000, 8, 0.0, 1.0, 1.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated() / 2**20
    nv_peak = nvsmi_used_mib()
    t_all = time.perf_counter()
    for r in rows:
        prompt = build_structured_prompt(guideline, r["text"])
        n_in = len(tokenizer(prompt, truncation=True, max_length=12000)["input_ids"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        pred = generate_batch(model, tokenizer, [prompt], 12000, 2048, 0.0, 1.0, 1.0)[0]
        torch.cuda.synchronize()
        el = time.perf_counter() - t
        n_out = len(tokenizer(pred, add_special_tokens=False)["input_ids"])
        nv = nvsmi_used_mib()
        nv_peak = max(nv_peak, nv)
        per_doc.append(
            {
                "id": r["id"],
                "input_tokens": n_in,
                "output_tokens": n_out,
                "seconds": round(el, 3),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "nvsmi_used_mib": nv,
            }
        )
        print(f"{r['id']}\tin={n_in}\tout={n_out}\t{el:.1f}s\tpeak_alloc={per_doc[-1]['peak_allocated_mib']:.0f}MiB\tnvsmi={nv}MiB", flush=True)
    total_s = time.perf_counter() - t_all

    secs = [d["seconds"] for d in per_doc]
    summary = {
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", os.environ.get("NVSMI_INDEX", "0")]).decode().strip(),
        "model_id": model_id,
        "adapter": str(adapter),
        "quantization": "bitsandbytes 4-bit NF4, double quant, bf16 compute",
        "n_total_params_reported_by_torch": n_total,
        "n_lora_params": n_trainable,
        "model_load_seconds": round(load_s, 1),
        "memory_after_load_mib": after_load,
        "n_docs": len(per_doc),
        "total_seconds_sequential": round(total_s, 1),
        "seconds_per_doc_mean": round(sum(secs) / len(secs), 2),
        "seconds_per_doc_median": round(sorted(secs)[len(secs) // 2], 2),
        "seconds_per_doc_min": round(min(secs), 2),
        "seconds_per_doc_max": round(max(secs), 2),
        "input_tokens_mean": round(sum(d["input_tokens"] for d in per_doc) / len(per_doc), 1),
        "input_tokens_max": max(d["input_tokens"] for d in per_doc),
        "output_tokens_mean": round(sum(d["output_tokens"] for d in per_doc) / len(per_doc), 1),
        "output_tokens_max": max(d["output_tokens"] for d in per_doc),
        "peak_allocated_mib_max": max(d["peak_allocated_mib"] for d in per_doc),
        "peak_reserved_mib_max": max(d["peak_reserved_mib"] for d in per_doc),
        "nvsmi_used_mib_peak": nv_peak,
        "baseline_allocated_mib_after_warmup": round(base_alloc, 1),
        "settings": {"batch_size": 1, "max_new_tokens": 2048, "max_input_length": 12000, "temperature": 0.0, "prompt": "structured_template guideline_lung_noae_v2"},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.condition}_seed{args.seed}"
    with open(os.path.join(args.out_dir, f"inference_latency_per_doc_{tag}.tsv"), "w") as f:
        keys = list(per_doc[0].keys())
        f.write("\t".join(keys) + "\n")
        for d in per_doc:
            f.write("\t".join(str(d[k]) for k in keys) + "\n")
    with open(os.path.join(args.out_dir, f"inference_measurement_{tag}.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
