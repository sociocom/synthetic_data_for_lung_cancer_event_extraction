# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from trainpipe.sft.prompting import make_data_collator


def find_all_linear_names(model) -> List[str]:
    cls = bnb.nn.Linear4bit
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            if not name:
                continue
            cand = name.split(".")[-1]
            if cand:
                names.add(cand)

    if not names:
        names.update(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    return sorted(names)


class _SFTTrainerWithSampler(SFTTrainer):
    def __init__(self, *args, train_sampler: Optional[Sampler] = None, **kwargs):
        self._train_sampler_override = train_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self._train_sampler_override is not None:
            return self._train_sampler_override
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()


def build_trainer(
    model_id: str,
    init_adapter_path: str,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    output_dir: str,
    save_total_limit: int,
    eval_steps: int,
    num_train_epochs: float,
    max_length: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    train_sampler: Optional[Sampler] = None,
    extra_callbacks: Optional[List[TrainerCallback]] = None,
) -> SFTTrainer:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        use_cache=False,
    )

    if getattr(model.config, "rope_scaling", None) is not None:
        model.config.rope_scaling = None
    use_init_adapter = bool(init_adapter_path)
    if use_init_adapter:
        if not os.path.isdir(init_adapter_path):
            raise FileNotFoundError(f"init_adapter_path not found: {init_adapter_path}")
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
        lora_cfg = None
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    has_validation = val_dataset is not None and len(val_dataset) > 0

    sft_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        ddp_find_unused_parameters=False,
        report_to=[],
        save_total_limit=save_total_limit,
        save_strategy="steps",
        save_steps=eval_steps,
        max_length=max_length,
        packing=False,
    )
    if has_validation:
        sft_args_kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        sft_args_kwargs.update(
            eval_strategy="no",
            load_best_model_at_end=False,
        )
    sft_args = SFTConfig(**sft_args_kwargs)

    collator = make_data_collator(tokenizer)

    callbacks: List[TrainerCallback] = []
    if has_validation and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = _SFTTrainerWithSampler(
        model=model,
        peft_config=lora_cfg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=sft_args,
        callbacks=callbacks,
        train_sampler=train_sampler,
    )

    return trainer


def collect_loss_history_rows(trainer: SFTTrainer) -> List[Dict[str, Optional[float]]]:
    rows_by_step: Dict[int, Dict[str, Optional[float]]] = {}

    for entry in trainer.state.log_history:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step", None)
        epoch = entry.get("epoch", None)
        if step is None:
            continue
        step = int(step)

        rec = rows_by_step.setdefault(
            step,
            {
                "epoch": epoch,
                "train_loss": None,
                "val_loss": None,
                "learning_rate": None,
                "grad_norm": None,
            },
        )
        if epoch is not None:
            rec["epoch"] = epoch
        if "loss" in entry:
            rec["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            rec["val_loss"] = entry["eval_loss"]
        if "learning_rate" in entry:
            rec["learning_rate"] = entry["learning_rate"]
        if "grad_norm" in entry:
            rec["grad_norm"] = entry["grad_norm"]

    rows: List[Dict[str, Optional[float]]] = []
    for step in sorted(rows_by_step.keys()):
        rec = rows_by_step[step]
        rows.append(
            {
                "step": step,
                "epoch": rec["epoch"],
                "train_loss": rec["train_loss"],
                "val_loss": rec["val_loss"],
                "learning_rate": rec["learning_rate"],
                "grad_norm": rec["grad_norm"],
            }
        )
    return rows


def save_loss_history(trainer: SFTTrainer, out_csv_path: str) -> None:
    rows = collect_loss_history_rows(trainer)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "val_loss", "learning_rate", "grad_norm"])
        for r in rows:
            writer.writerow(
                [
                    r["step"],
                    r["epoch"],
                    r["train_loss"],
                    r["val_loss"],
                    r["learning_rate"],
                    r["grad_norm"],
                ]
            )
