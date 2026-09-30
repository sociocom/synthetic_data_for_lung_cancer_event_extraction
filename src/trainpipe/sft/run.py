# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import json
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import AutoTokenizer, TrainerCallback, set_seed

from trainpipe.common.jsonl_io import iter_jsonl, load_text_file, save_json, save_jsonl
from trainpipe.sft.args import TrainSFTArgs, TrainSFTPreparedArgs
from trainpipe.sft.data import (
    build_pairs_by_id,
    build_raw_splits_from_pairs,
    cv_train_val_test_indices,
    kfold_split_all,
    load_annotation_dict,
    load_progress_dict,
    raw_to_hf_dataset,
    split_indices,
)
from trainpipe.sft.prompting import make_build_text_fn, make_tok_fn
from trainpipe.sft.trainer import build_trainer, save_loss_history


def _load_raw_rows_from_jsonl(path: str, id_key: str, progress_key: str, annotation_key: str) -> list[dict]:
    rows: list[dict] = []
    for obj in iter_jsonl(path):
        _id = obj.get(id_key)
        note = obj.get(progress_key)
        ann = obj.get(annotation_key)
        if _id is None or note is None or ann is None:
            continue
        note_s = str(note).strip()
        ann_s = str(ann).strip()
        if not note_s or not ann_s:
            continue
        rows.append(
            {
                "id": str(_id),
                "progress_note": note_s,
                "annotation": ann_s,
            }
        )
    return rows


def _prepare_datasets(
    tokenizer: AutoTokenizer,
    guideline: str,
    train_raw: list[dict],
    val_raw: list[dict],
    max_length: int,
) -> tuple[Dataset, Dataset | None]:
    raw_train = raw_to_hf_dataset(train_raw)
    raw_val = raw_to_hf_dataset(val_raw) if val_raw else None

    build_text_fn = make_build_text_fn(guideline_text=guideline, tokenizer=tokenizer)
    tok_fn = make_tok_fn(tokenizer=tokenizer, max_length=max_length)

    train_with_text = raw_train.map(build_text_fn, batched=True)
    val_with_text = raw_val.map(build_text_fn, batched=True) if raw_val is not None else None

    train_dataset: Dataset = train_with_text.map(tok_fn, batched=True, remove_columns=["id", "progress_note", "annotation", "text"])
    val_dataset: Dataset | None = (
        val_with_text.map(tok_fn, batched=True, remove_columns=["id", "progress_note", "annotation", "text"])
        if val_with_text is not None
        else None
    )
    return train_dataset, val_dataset


def _find_non_finite_logs(log_history: list[dict]) -> list[dict]:
    bad: list[dict] = []
    for e in log_history:
        if not isinstance(e, dict):
            continue
        for k in ("loss", "eval_loss", "grad_norm", "entropy", "eval_entropy"):
            v = e.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isfinite(fv):
                bad.append({"step": e.get("step"), "key": k, "value": v})
    return bad


def _has_non_finite_trainable_params(model: torch.nn.Module) -> bool:
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if not p.is_floating_point():
                continue
            if not torch.isfinite(p).all():
                return True
    return False


def _is_finite_metric(v: object) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _load_precomputed_test_fold_indices(
    path: str,
    *,
    test_fold: int,
    pairs: list[tuple[str, str, str]],
) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    folds = obj.get("folds")
    if not isinstance(folds, list):
        raise ValueError(f"cv_folds_json must contain a list field 'folds': {path}")

    id_to_idx = {_id: i for i, (_id, _note, _ann) in enumerate(pairs)}
    selected = None
    for rec in folds:
        if not isinstance(rec, dict):
            continue
        if int(rec.get("fold", -1)) == int(test_fold):
            selected = rec
            break
    if selected is None:
        raise ValueError(f"Fold {test_fold} not found in cv_folds_json: {path}")

    if "test_ids" in selected:
        ids = selected["test_ids"]
        if not isinstance(ids, list):
            raise ValueError(f"fold={test_fold} test_ids must be a list")
        missing = [str(x) for x in ids if str(x) not in id_to_idx]
        if missing:
            raise ValueError(f"fold={test_fold} has ids not present in data: {missing[:5]}")
        return np.array([id_to_idx[str(x)] for x in ids], dtype=int)

    if "test_indices" in selected:
        idxs = np.array([int(x) for x in selected["test_indices"]], dtype=int)
        if len(idxs) == 0:
            raise ValueError(f"fold={test_fold} has empty test_indices")
        if int(idxs.min()) < 0 or int(idxs.max()) >= len(pairs):
            raise ValueError(f"fold={test_fold} test_indices out of range")
        return idxs

    raise ValueError(f"fold={test_fold} must contain test_ids or test_indices")


class _EpochResampledMixSampler(Sampler[int]):
    def __init__(
        self,
        sim_count: int,
        synth_pool_count: int,
        synth_target: int,
        seed_base: int,
    ) -> None:
        self._sim_count = int(sim_count)
        self._synth_pool_count = int(synth_pool_count)
        self._synth_take = min(max(int(synth_target), 0), self._synth_pool_count)
        self._seed_base = int(seed_base)
        self._next_epoch_zero_based = 0
        self.epoch_sampling_log: list[dict] = []

    def set_epoch(self, epoch_zero_based: int) -> None:
        self._next_epoch_zero_based = max(int(epoch_zero_based), 0)

    def __len__(self) -> int:
        return self._sim_count + self._synth_take

    def __iter__(self):
        epoch_zero_based = int(self._next_epoch_zero_based)
        sample_seed = self._seed_base + epoch_zero_based + 1
        rng = np.random.default_rng(sample_seed)

        sim_indices = np.arange(self._sim_count, dtype=np.int64)
        if self._synth_take > 0:
            synth_local = rng.choice(self._synth_pool_count, size=self._synth_take, replace=False)
            synth_indices = synth_local + self._sim_count
            mixed_indices = np.concatenate((sim_indices, synth_indices.astype(np.int64, copy=False)))
        else:
            mixed_indices = sim_indices

        if len(mixed_indices) > 1:
            rng.shuffle(mixed_indices)

        self.epoch_sampling_log.append(
            {
                "epoch": epoch_zero_based + 1,
                "sample_seed": int(sample_seed),
                "sim_train_count": int(self._sim_count),
                "synth_target": int(self._synth_take),
                "synth_sampled": int(self._synth_take),
                "mixed_train_count": int(len(mixed_indices)),
            }
        )
        self._next_epoch_zero_based = epoch_zero_based + 1
        return iter(int(i) for i in mixed_indices.tolist())


class _SamplerEpochCallback(TrainerCallback):
    def __init__(self, sampler: _EpochResampledMixSampler) -> None:
        self._sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = 0 if state.epoch is None else int(math.floor(float(state.epoch)))
        self._sampler.set_epoch(epoch)
        return control


class _NonFiniteGuardCallback(TrainerCallback):
    def __init__(self) -> None:
        self.bad_logs: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not isinstance(logs, dict):
            return control
        found = False
        for k in ("loss", "eval_loss", "grad_norm", "entropy", "eval_entropy"):
            v = logs.get(k)
            if v is None:
                continue
            try:
                finite = math.isfinite(float(v))
            except Exception:
                finite = True
            if not finite:
                found = True
                self.bad_logs.append({"step": int(state.global_step), "key": k, "value": v})
        if found:
            control.should_training_stop = True
        return control


def run_sft(args: TrainSFTArgs) -> None:
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    guideline = load_text_file(args.guideline_path)

    progress = load_progress_dict(args.progress_jsonl, args.id_key, args.progress_key)
    ann = load_annotation_dict(args.annotations_jsonl, args.id_key, args.annotation_key)

    pairs = build_pairs_by_id(progress, ann, max_samples=args.max_samples)
    n = len(pairs)

    os.makedirs(args.output_dir, exist_ok=True)

    # Decide split mode
    if args.cv_num_folds and args.cv_num_folds >= 2:
        if args.cv_fold < 0:
            raise ValueError("cv_fold must be set when cv_num_folds >= 2")

        # Protocol:
        #   test = fold=cv_fold
        #   val  = sample cv_val_size from train-pool using seed (cv_val_seed_base + test_fold)
        #   if cv_val_size <= 0: no validation split (test fold + train pool only)
        num_folds = int(args.cv_num_folds)
        test_fold = int(args.cv_fold)
        if test_fold >= num_folds:
            raise ValueError("cv_fold must be in [0, cv_num_folds)")
        cv_val_size = int(args.cv_val_size) if args.cv_val_size is not None else 0

        if args.cv_folds_json:
            test_idx = _load_precomputed_test_fold_indices(
                args.cv_folds_json,
                test_fold=test_fold,
                pairs=pairs,
            )
            test_set = set(int(i) for i in test_idx.tolist())
            train_pool = np.array([i for i in range(n) if i not in test_set], dtype=int)
            if len(test_idx) < 1:
                raise ValueError("Test fold is empty in precomputed CV split")
            if len(train_pool) < 1:
                raise ValueError("Train split is empty in precomputed CV split")
            if cv_val_size > 0:
                desired = min(int(cv_val_size), len(train_pool) - 1)
                desired = max(1, desired)
                rng = np.random.default_rng(int(args.cv_val_seed_base) + int(test_fold))
                perm = rng.permutation(len(train_pool))
                val_idx = train_pool[perm[:desired]]
                train_idx = train_pool[perm[desired:]]
                split_mode = "cv_precomputed_train_val_test"
            else:
                train_idx = train_pool
                val_idx = np.array([], dtype=int)
                split_mode = "cv_precomputed_train_test_no_val"
        elif cv_val_size > 0:
            train_idx, val_idx, test_idx = cv_train_val_test_indices(
                n=n,
                num_folds=num_folds,
                test_fold=test_fold,
                split_seed=int(args.cv_split_seed),
                val_size=cv_val_size,
                val_seed_base=int(args.cv_val_seed_base),
            )
            split_mode = "cv_train_val_test"
        else:
            folds = kfold_split_all(
                n=n,
                num_folds=num_folds,
                seed=int(args.cv_split_seed),
            )
            test_idx = folds[test_fold]
            train_idx = np.concatenate([f for i, f in enumerate(folds) if i != test_fold])
            if len(train_idx) < 1:
                raise ValueError("Train split is empty in CV no-validation mode")
            val_idx = np.array([], dtype=int)
            split_mode = "cv_train_test_no_val"
    else:
        train_idx, val_idx, test_idx = split_indices(n=n, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
        split_mode = "holdout"

    # Save provenance
    meta: Dict[str, object] = {
        "split_mode": split_mode,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "cv_num_folds": args.cv_num_folds,
        "cv_fold(test_fold)": args.cv_fold,
        "cv_split_seed": args.cv_split_seed,
        "cv_folds_json": args.cv_folds_json,
        "cv_val_size": args.cv_val_size,
        "cv_val_seed_base": args.cv_val_seed_base,
        "max_samples": args.max_samples,
        "n_pairs": n,
    }
    save_json(meta, os.path.join(args.output_dir, "run_meta.json"))

    train_raw, val_raw, test_raw = build_raw_splits_from_pairs(pairs, train_idx, val_idx, test_idx)

    ds_dir = os.path.join(args.output_dir, "datasets_raw")
    save_jsonl(train_raw, os.path.join(ds_dir, "train.jsonl"))
    save_jsonl(val_raw, os.path.join(ds_dir, "validation.jsonl"))
    if test_raw:
        save_jsonl(test_raw, os.path.join(ds_dir, "test.jsonl"))

    raw_train = raw_to_hf_dataset(train_raw)
    raw_val = raw_to_hf_dataset(val_raw) if val_raw else None

    build_text_fn = make_build_text_fn(guideline_text=guideline, tokenizer=tokenizer)
    tok_fn = make_tok_fn(tokenizer=tokenizer, max_length=args.max_length)

    train_with_text = raw_train.map(build_text_fn, batched=True)
    val_with_text = raw_val.map(build_text_fn, batched=True) if raw_val is not None else None

    train_dataset: Dataset = train_with_text.map(tok_fn, batched=True, remove_columns=["id", "progress_note", "annotation", "text"])
    val_dataset: Dataset | None = (
        val_with_text.map(tok_fn, batched=True, remove_columns=["id", "progress_note", "annotation", "text"])
        if val_with_text is not None
        else None
    )

    trainer = build_trainer(
        model_id=args.model_id,
        init_adapter_path=args.init_adapter_path,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)

    loss_csv_path = os.path.join(args.output_dir, args.loss_csv)
    save_loss_history(trainer, loss_csv_path)


def run_sft_prepared(args: TrainSFTPreparedArgs) -> None:
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    guideline = load_text_file(args.guideline_path)

    train_raw = _load_raw_rows_from_jsonl(
        path=args.train_jsonl,
        id_key=args.id_key,
        progress_key=args.progress_key,
        annotation_key=args.annotation_key,
    )
    val_raw = _load_raw_rows_from_jsonl(
        path=args.validation_jsonl,
        id_key=args.id_key,
        progress_key=args.progress_key,
        annotation_key=args.annotation_key,
    )
    synth_pool_raw: list[dict] = []
    if args.synth_pool_jsonl:
        synth_pool_raw = _load_raw_rows_from_jsonl(
            path=args.synth_pool_jsonl,
            id_key=args.id_key,
            progress_key=args.progress_key,
            annotation_key=args.annotation_key,
        )

    test_raw = []
    if args.test_jsonl:
        test_raw = _load_raw_rows_from_jsonl(
            path=args.test_jsonl,
            id_key=args.id_key,
            progress_key=args.progress_key,
            annotation_key=args.annotation_key,
        )

    if not train_raw:
        raise RuntimeError(f"No valid train rows loaded from: {args.train_jsonl}")
    if not val_raw:
        raise RuntimeError(f"No valid validation rows loaded from: {args.validation_jsonl}")

    os.makedirs(args.output_dir, exist_ok=True)
    ds_dir = os.path.join(args.output_dir, "datasets_raw")
    save_jsonl(val_raw, os.path.join(ds_dir, "validation.jsonl"))
    if test_raw:
        save_jsonl(test_raw, os.path.join(ds_dir, "test.jsonl"))

    # Mix training path (A-method):
    # keep sim-train fixed and re-sample synth rows at every epoch via Sampler.
    if args.synth_pool_jsonl:
        if args.mix_synth_ratio < 0 or args.mix_sim_ratio <= 0:
            raise ValueError("mix_synth_ratio must be >=0 and mix_sim_ratio must be >0")
        if not synth_pool_raw:
            raise RuntimeError(f"No valid synthetic pool rows loaded from: {args.synth_pool_jsonl}")
        if args.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0")
        synth_target = int(round(len(train_raw) * float(args.mix_synth_ratio) / float(args.mix_sim_ratio)))
        synth_target = max(0, synth_target)
        synth_take = min(synth_target, len(synth_pool_raw))

        save_jsonl(train_raw, os.path.join(ds_dir, "train_sim.jsonl"))
        save_jsonl(synth_pool_raw, os.path.join(ds_dir, "synth_pool.jsonl"))
        train_pool_raw = train_raw + synth_pool_raw
        save_jsonl(train_pool_raw, os.path.join(ds_dir, "train_pool.jsonl"))
        save_jsonl(train_raw, os.path.join(ds_dir, "train.jsonl"))

        train_dataset, val_dataset = _prepare_datasets(
            tokenizer=tokenizer,
            guideline=guideline,
            train_raw=train_pool_raw,
            val_raw=val_raw,
            max_length=args.max_length,
        )

        mix_sampler = _EpochResampledMixSampler(
            sim_count=len(train_raw),
            synth_pool_count=len(synth_pool_raw),
            synth_target=synth_target,
            seed_base=int(args.mix_seed_base),
        )
        extra_callbacks: list[TrainerCallback] = [_SamplerEpochCallback(mix_sampler)]
        non_finite_guard_cb: _NonFiniteGuardCallback | None = None
        if args.fail_on_non_finite:
            non_finite_guard_cb = _NonFiniteGuardCallback()
            extra_callbacks.append(non_finite_guard_cb)

        trainer = build_trainer(
            model_id=args.model_id,
            init_adapter_path=args.init_adapter_path,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            eval_steps=args.eval_steps,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            train_sampler=mix_sampler,
            extra_callbacks=extra_callbacks,
        )

        trainer.train()

        best_dir = os.path.join(args.output_dir, "best")
        trainer.save_model(best_dir)
        loss_csv_path = os.path.join(args.output_dir, args.loss_csv)
        save_loss_history(trainer, loss_csv_path)

        bad_logs = _find_non_finite_logs(trainer.state.log_history)
        if non_finite_guard_cb is not None and non_finite_guard_cb.bad_logs:
            bad_logs.extend(non_finite_guard_cb.bad_logs)
        bad_params = _has_non_finite_trainable_params(trainer.model)
        best_metric = trainer.state.best_metric
        metric_finite = _is_finite_metric(best_metric)

        status = "completed"
        fail_reason = ""
        if bad_logs:
            status = "failed"
            fail_reason = f"non-finite logs detected (count={len(bad_logs)})"
        elif bad_params:
            status = "failed"
            fail_reason = "non-finite trainable parameter detected"
        elif not metric_finite:
            status = "failed"
            fail_reason = f"best metric is not finite: {best_metric}"

        if args.keep_epoch_artifacts:
            epoch_runs_dir = os.path.join(args.output_dir, "epoch_runs")
            os.makedirs(epoch_runs_dir, exist_ok=True)
            save_json(mix_sampler.epoch_sampling_log, os.path.join(epoch_runs_dir, "epoch_sampling.json"))
            if bad_logs:
                save_json(bad_logs, os.path.join(epoch_runs_dir, "bad_logs.json"))

        meta: Dict[str, object] = {
            "split_mode": "prepared_with_epoch_resample",
            "resample_method": "sampler",
            "status": status,
            "fail_reason": fail_reason,
            "seed": args.seed,
            "train_jsonl(sim_train)": args.train_jsonl,
            "validation_jsonl": args.validation_jsonl,
            "synth_pool_jsonl": args.synth_pool_jsonl,
            "test_jsonl": args.test_jsonl,
            "mix": {
                "mix_synth_ratio": int(args.mix_synth_ratio),
                "mix_sim_ratio": int(args.mix_sim_ratio),
                "mix_seed_base": int(args.mix_seed_base),
                "num_train_epochs_requested": float(args.num_train_epochs),
                "synth_target_per_epoch": int(synth_target),
                "synth_take_per_epoch": int(synth_take),
            },
            "counts": {
                "sim_train": len(train_raw),
                "validation": len(val_raw),
                "test": len(test_raw),
                "synth_pool": len(synth_pool_raw),
                "train_pool": len(train_pool_raw),
            },
            "selection": {
                "metric": "eval_loss",
                "best_metric": float(best_metric) if metric_finite else None,
            },
            "epoch_sampling": mix_sampler.epoch_sampling_log,
            "non_finite": {
                "bad_log_count": len(bad_logs),
                "bad_param_non_finite": bool(bad_params),
            },
        }
        save_json(meta, os.path.join(args.output_dir, "run_meta.json"))
        if status != "completed" and args.fail_on_non_finite:
            raise RuntimeError(f"epoch-resampled training failed: {fail_reason}")
        return

    meta: Dict[str, object] = {
        "split_mode": "prepared",
        "seed": args.seed,
        "train_jsonl": args.train_jsonl,
        "validation_jsonl": args.validation_jsonl,
        "test_jsonl": args.test_jsonl,
        "n_train": len(train_raw),
        "n_validation": len(val_raw),
        "n_test": len(test_raw),
    }
    save_json(meta, os.path.join(args.output_dir, "run_meta.json"))

    save_jsonl(train_raw, os.path.join(ds_dir, "train.jsonl"))

    train_dataset, val_dataset = _prepare_datasets(
        tokenizer=tokenizer,
        guideline=guideline,
        train_raw=train_raw,
        val_raw=val_raw,
        max_length=args.max_length,
    )

    trainer = build_trainer(
        model_id=args.model_id,
        init_adapter_path=args.init_adapter_path,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)

    loss_csv_path = os.path.join(args.output_dir, args.loss_csv)
    save_loss_history(trainer, loss_csv_path)
