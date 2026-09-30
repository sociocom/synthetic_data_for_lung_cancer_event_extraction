#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from datetime import datetime

from trainpipe import TrainSFTArgs, TrainSFTPreparedArgs, run_sft, run_sft_prepared
from trainpipe.token_stats import compute_token_stats


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir(base_dir: str, exp_name: str) -> str:
    exp_name = exp_name.strip().replace(" ", "_")
    if not exp_name:
        raise ValueError("exp_name must be non-empty")
    return os.path.join(base_dir, f"{exp_name}_{_timestamp()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL-based holdout/CV Structured SFT trainer")

    subparsers = p.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run SFT training (holdout or one-fold CV)")
    train_parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model id or local path")
    train_parser.add_argument(
        "--init_adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter directory to continue training from (two-stage setup).",
    )
    train_parser.add_argument("--progress_jsonl", type=str, required=True, help="progress_notes.jsonl path")
    train_parser.add_argument("--annotations_jsonl", type=str, required=True, help="annotations.jsonl path")

    train_parser.add_argument("--progress_key", type=str, default="progress_note")
    train_parser.add_argument("--annotation_key", type=str, default="annotation")
    train_parser.add_argument("--id_key", type=str, default="id")

    train_parser.add_argument("--guideline_path", type=str, required=True, help="guideline text file path (required)")

    # Output control
    train_parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base directory for outputs. If set, output_dir is auto-generated as <base>/<exp_name>_<timestamp>/...",
    )
    train_parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Human-readable experiment name (used with --output_base_dir).",
    )

    # Backward compatible explicit output_dir
    train_parser.add_argument("--output_dir", type=str, default=None)

    train_parser.add_argument("--seed", type=int, default=42)

    # CV controls (fold-based execution)
    train_parser.add_argument("--cv_num_folds", type=int, default=0, help="Enable K-fold CV if >=2")
    train_parser.add_argument("--cv_fold", type=int, default=-1, help="Test fold index in [0, cv_num_folds)")
    train_parser.add_argument("--cv_split_seed", type=int, default=42, help="Seed used to generate K-fold split")
    train_parser.add_argument(
        "--cv_folds_json",
        type=str,
        default="",
        help="Optional JSON file with precomputed CV test folds. If set, overrides random K-fold splitting.",
    )
    train_parser.add_argument(
        "--cv_val_size",
        type=int,
        default=0,
        help="(CV) Validation set size sampled from train-pool (all folds except test fold).",
    )
    train_parser.add_argument(
        "--cv_val_seed_base",
        type=int,
        default=42,
        help="(CV) Base seed; actual val sampling seed defaults to (cv_val_seed_base + cv_fold).",
    )

    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--test_ratio", type=float, default=0.0)

    train_parser.add_argument("--save_total_limit", type=int, default=2)
    train_parser.add_argument("--loss_csv", type=str, default="loss_history.csv")
    train_parser.add_argument("--eval_steps", type=int, default=50)
    train_parser.add_argument("--num_train_epochs", type=float, default=5)
    train_parser.add_argument("--max_length", type=int, default=2048)

    train_parser.add_argument("--early_stopping_patience", type=int, default=3)
    train_parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    train_parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    train_parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.05)

    train_parser.add_argument("--max_samples", type=int, default=0)

    train_prepared_parser = subparsers.add_parser(
        "train-prepared",
        help="Run SFT training from prepared train/validation(/test) JSONLs.",
    )
    train_prepared_parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model id or local path")
    train_prepared_parser.add_argument(
        "--init_adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter directory to continue training from.",
    )
    train_prepared_parser.add_argument("--train_jsonl", type=str, required=True, help="Prepared train JSONL path")
    train_prepared_parser.add_argument("--validation_jsonl", type=str, required=True, help="Prepared validation JSONL path")
    train_prepared_parser.add_argument(
        "--synth_pool_jsonl",
        type=str,
        default="",
        help="Optional synthetic pool JSONL path. If set, synthetic rows are re-sampled every epoch.",
    )
    train_prepared_parser.add_argument(
        "--test_jsonl",
        type=str,
        default="",
        help="Optional prepared test JSONL path (saved for provenance/evaluation).",
    )

    train_prepared_parser.add_argument("--progress_key", type=str, default="progress_note")
    train_prepared_parser.add_argument("--annotation_key", type=str, default="annotation")
    train_prepared_parser.add_argument("--id_key", type=str, default="id")

    train_prepared_parser.add_argument("--guideline_path", type=str, required=True, help="guideline text file path (required)")

    train_prepared_parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base directory for outputs. If set, output_dir is auto-generated as <base>/<exp_name>_<timestamp>/...",
    )
    train_prepared_parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Human-readable experiment name (used with --output_base_dir).",
    )
    train_prepared_parser.add_argument("--output_dir", type=str, default=None)

    train_prepared_parser.add_argument("--seed", type=int, default=42)
    train_prepared_parser.add_argument("--save_total_limit", type=int, default=2)
    train_prepared_parser.add_argument("--loss_csv", type=str, default="loss_history.csv")
    train_prepared_parser.add_argument("--eval_steps", type=int, default=50)
    train_prepared_parser.add_argument("--num_train_epochs", type=float, default=5)
    train_prepared_parser.add_argument("--max_length", type=int, default=2048)
    train_prepared_parser.add_argument("--early_stopping_patience", type=int, default=3)
    train_prepared_parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    train_prepared_parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    train_prepared_parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    train_prepared_parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_prepared_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_prepared_parser.add_argument("--warmup_ratio", type=float, default=0.05)
    train_prepared_parser.add_argument("--mix_synth_ratio", type=int, default=50)
    train_prepared_parser.add_argument("--mix_sim_ratio", type=int, default=50)
    train_prepared_parser.add_argument("--mix_seed_base", type=int, default=42)
    train_prepared_parser.add_argument(
        "--fail_on_non_finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail immediately if non-finite values are detected during epoch-resampled training.",
    )
    train_prepared_parser.add_argument(
        "--keep_epoch_artifacts",
        action="store_true",
        help="Keep per-epoch trainer artifacts under output_dir/epoch_runs for debugging.",
    )

    token_stats_parser = subparsers.add_parser("token-stats", help="Compute token length statistics")
    token_stats_parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model id or local path")
    token_stats_parser.add_argument("--progress_jsonl", type=str, required=True, help="progress_notes.jsonl path")
    token_stats_parser.add_argument("--annotations_jsonl", type=str, required=True, help="annotations.jsonl path")
    token_stats_parser.add_argument("--progress_key", type=str, default="progress_note")
    token_stats_parser.add_argument("--annotation_key", type=str, default="annotation")
    token_stats_parser.add_argument("--id_key", type=str, default="id")
    token_stats_parser.add_argument("--guideline_path", type=str, required=True, help="guideline text file path (required)")
    token_stats_parser.add_argument("--max_length", type=int, default=4096)
    token_stats_parser.add_argument("--do_truncate", action="store_true")
    token_stats_parser.add_argument("--print_topk", type=int, default=20)
    token_stats_parser.add_argument("--max_samples", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        # Resolve output_dir
        if args.output_dir is None:
            if args.output_base_dir is None or args.exp_name is None:
                raise ValueError("Provide either --output_dir OR (--output_base_dir and --exp_name)")
            run_dir = _make_run_dir(args.output_base_dir, args.exp_name)
        else:
            run_dir = args.output_dir

        # For CV, place fold outputs under run_dir/folds/fold=<k>/
        is_cv = args.cv_num_folds is not None and args.cv_num_folds >= 2
        if is_cv:
            if args.cv_fold is None or args.cv_fold < 0:
                raise ValueError("With --cv_num_folds >= 2, --cv_fold must be set")
            fold_dir = os.path.join(run_dir, "folds", f"fold={args.cv_fold}")
            output_dir = fold_dir
            # Training randomness is fixed across folds (requested): use base seed as-is.
            train_seed = int(args.seed)
        else:
            output_dir = run_dir
            train_seed = int(args.seed)

        run_sft(TrainSFTArgs(
            model_id=args.model_id,
            init_adapter_path=args.init_adapter_path,
            progress_jsonl=args.progress_jsonl,
            annotations_jsonl=args.annotations_jsonl,
            output_dir=output_dir,
            progress_key=args.progress_key,
            annotation_key=args.annotation_key,
            id_key=args.id_key,
            guideline_path=args.guideline_path,
            seed=train_seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            save_total_limit=args.save_total_limit,
            loss_csv=args.loss_csv,
            eval_steps=args.eval_steps,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            max_samples=args.max_samples,
            cv_num_folds=int(args.cv_num_folds) if args.cv_num_folds is not None else 0,
            cv_fold=int(args.cv_fold) if args.cv_fold is not None else -1,
            cv_split_seed=int(args.cv_split_seed) if args.cv_split_seed is not None else 42,
            cv_folds_json=str(args.cv_folds_json) if getattr(args, "cv_folds_json", None) else "",
            cv_val_size=int(args.cv_val_size) if getattr(args, "cv_val_size", None) is not None else 0,
            cv_val_seed_base=int(args.cv_val_seed_base) if getattr(args, "cv_val_seed_base", None) is not None else 42,
        ))

    elif args.command == "train-prepared":
        if args.output_dir is None:
            if args.output_base_dir is None or args.exp_name is None:
                raise ValueError("Provide either --output_dir OR (--output_base_dir and --exp_name)")
            output_dir = _make_run_dir(args.output_base_dir, args.exp_name)
        else:
            output_dir = args.output_dir

        run_sft_prepared(TrainSFTPreparedArgs(
            model_id=args.model_id,
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            synth_pool_jsonl=args.synth_pool_jsonl,
            test_jsonl=args.test_jsonl,
            output_dir=output_dir,
            init_adapter_path=args.init_adapter_path,
            progress_key=args.progress_key,
            annotation_key=args.annotation_key,
            id_key=args.id_key,
            guideline_path=args.guideline_path,
            seed=int(args.seed),
            save_total_limit=int(args.save_total_limit),
            loss_csv=args.loss_csv,
            eval_steps=int(args.eval_steps),
            num_train_epochs=float(args.num_train_epochs),
            max_length=int(args.max_length),
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_threshold=float(args.early_stopping_threshold),
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            learning_rate=float(args.learning_rate),
            warmup_ratio=float(args.warmup_ratio),
            mix_synth_ratio=int(args.mix_synth_ratio),
            mix_sim_ratio=int(args.mix_sim_ratio),
            mix_seed_base=int(args.mix_seed_base),
            fail_on_non_finite=bool(args.fail_on_non_finite),
            keep_epoch_artifacts=bool(args.keep_epoch_artifacts),
        ))

    elif args.command == "token-stats":
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
