# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainSFTArgs:
    model_id: str
    progress_jsonl: str
    annotations_jsonl: str
    output_dir: str
    init_adapter_path: str = ""

    progress_key: str = "progress_note"
    annotation_key: str = "annotation"
    id_key: str = "id"

    guideline_path: str = ""
    seed: int = 42

    val_ratio: float = 0.2
    test_ratio: float = 0.0

    save_total_limit: int = 2
    loss_csv: str = "loss_history.csv"
    eval_steps: int = 50
    num_train_epochs: float = 5
    max_length: int = 2048

    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05

    max_samples: int = 0

    # Optional CV (if num_folds >= 2)
    cv_num_folds: int = 0
    cv_fold: int = -1
    cv_split_seed: int = 42
    cv_folds_json: str = ""

    # CV protocol options
    # - cv_fold is treated as the TEST fold index.
    # - cv_val_size samples are drawn from the TRAIN-POOL (all folds except test fold).
    # - sampling seed = (cv_val_seed_base + cv_fold).
    cv_val_size: int = 0
    cv_val_seed_base: int = 42


@dataclass(frozen=True)
class TrainSFTPreparedArgs:
    model_id: str
    train_jsonl: str
    validation_jsonl: str
    output_dir: str
    synth_pool_jsonl: str = ""
    test_jsonl: str = ""
    init_adapter_path: str = ""

    progress_key: str = "progress_note"
    annotation_key: str = "annotation"
    id_key: str = "id"

    guideline_path: str = ""
    seed: int = 42

    save_total_limit: int = 2
    loss_csv: str = "loss_history.csv"
    eval_steps: int = 50
    num_train_epochs: float = 5
    max_length: int = 2048

    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05

    # If synth_pool_jsonl is set, synthetic rows are re-sampled every epoch.
    mix_synth_ratio: int = 50
    mix_sim_ratio: int = 50
    mix_seed_base: int = 42

    # Safety/maintenance knobs for epoch-resampled training.
    fail_on_non_finite: bool = True
    keep_epoch_artifacts: bool = False
