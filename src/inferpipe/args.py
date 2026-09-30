# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL inference for trainpipe SFT (optional LoRA)")

    p.add_argument(
        "--model_id",
        type=str,
        default=os.getenv("MODEL_ID", "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"),
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        default=os.getenv("ADAPTER_PATH", ""),
        help="Path to LoRA adapter dir (e.g., results/sft/<date>/best).",
    )
    p.add_argument(
        "--use_adapter",
        action="store_true",
        help="Attach LoRA adapters from --adapter_path.",
    )
    p.add_argument(
        "--no_adapter",
        action="store_true",
        help="Force disable adapters even if --adapter_path is set.",
    )

    p.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL. Expected keys: id, progress_note (by default).",
    )
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--id_key", type=str, default="id")
    p.add_argument("--input_key", type=str, default="progress_note")
    p.add_argument("--pred_key", type=str, default="pred_annotation")
    p.add_argument(
        "--raw_pred_key",
        type=str,
        default="",
        help="Optional key name to additionally store raw (pre-normalization) model output text.",
    )

    p.add_argument(
        "--guideline_path",
        type=str,
        default="",
        help="Optional guideline text file. Default is empty.",
    )
    p.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Optional TOML prompt file with keys: system, user. If set, STRUCTURED_TEMPLATE is not used.",
    )
    p.add_argument(
        "--icl_input_dir",
        type=str,
        default="",
        help="Optional ICL input directory (paired with --icl_output_dir).",
    )
    p.add_argument(
        "--icl_output_dir",
        type=str,
        default="",
        help="Optional ICL output directory (paired with --icl_input_dir).",
    )
    p.add_argument(
        "--n_context",
        type=int,
        default=0,
        help="Number of ICL examples to sample per input. Enable when > 0 and paired dirs are set.",
    )
    p.add_argument("--icl_seed", type=int, default=42, help="Seed for deterministic per-record ICL sampling.")
    p.add_argument("--icl_input_label", type=str, default="診療録", help="Label used in paired ICL prompt context.")
    p.add_argument("--icl_output_label", type=str, default="アノテーション", help="Label used in paired ICL prompt context.")

    p.add_argument("--max_input_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument(
        "--max_events",
        type=int,
        default=64,
        help="Max number of output events kept after JSON normalization.",
    )

    p.add_argument(
        "--quant",
        type=str,
        choices=["nf4", "none"],
        default=os.getenv("QUANT", "nf4"),
        help="nf4: bitsandbytes 4-bit (default, used for the SLM runs). "
        "none: load in bf16 (MXFP4 checkpoints such as gpt-oss are dequantized to bf16).",
    )
    p.add_argument(
        "--reasoning_effort",
        type=str,
        default=os.getenv("REASONING_EFFORT", ""),
        help="Passed to chat templates that support it (gpt-oss harmony: low/medium/high). Empty = template default.",
    )
    p.add_argument(
        "--final_channel",
        action="store_true",
        default=os.getenv("FINAL_CHANNEL", "false").lower() == "true",
        help="Harmony-format outputs (gpt-oss): keep special tokens when decoding and use only the "
        "<|channel|>final<|message|> ... segment as the prediction. Full text is kept in --raw_pred_key.",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default=os.getenv("ATTN_IMPL", ""),
        help="transformers attn_implementation (e.g. sdpa, eager, flex_attention). Empty = library default. "
        "gpt-oss has no sdpa kernel; eager materializes the full score matrix, so use flex_attention.",
    )
    p.add_argument(
        "--gpu_headroom_gb",
        type=float,
        default=float(os.getenv("GPU_HEADROOM_GB", "10")),
        help="With --quant none and device auto, per-GPU memory left free for activations / KV cache.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' or 'cuda' or 'cpu'. With quantized loading use 'auto' typically.",
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based, inclusive) in the filtered input list.",
    )
    p.add_argument(
        "--end",
        type=int,
        default=0,
        help="End index (0-based, exclusive). 0 means until the end.",
    )

    p.add_argument("--resume", action="store_true", help="Skip IDs already present in output_jsonl.")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print rendered prompts without loading model/generating outputs.",
    )
    p.add_argument(
        "--dry_run_n",
        type=int,
        default=5,
        help="Number of prompts to print in dry-run mode.",
    )

    return p.parse_args()
