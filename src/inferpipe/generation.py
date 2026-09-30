# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import torch


def _eos_token_ids(model, tokenizer):
    """Prefer the checkpoint's generation_config (may list several ids, e.g. gpt-oss
    <|return|>/<|endoftext|>/<|call|>), fall back to the tokenizer's single eos."""
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is None:
        return tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    eos = list(eos)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos:
        eos.append(tokenizer.eos_token_id)
    return eos


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    keep_special_tokens: bool = False,
) -> Tuple[List[str], List[int]]:
    """Returns (decoded texts, number of generated tokens per prompt)."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    # Some tokenizers (e.g. llm-jp / SIP-jmed) emit token_type_ids, which generate() rejects.
    enc = {k: v.to(model.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=_eos_token_ids(model, tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )

    out = model.generate(**enc, **gen_kwargs)

    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    preds: List[str] = []
    n_tokens: List[int] = []
    pad_id = tokenizer.pad_token_id
    for i in range(out.shape[0]):
        gen_ids = out[i, int(input_lens[i]) :]
        if pad_id is not None:
            n_tokens.append(int((gen_ids != pad_id).sum().item()))
        else:
            n_tokens.append(int(gen_ids.numel()))
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=not keep_special_tokens).strip())
    return preds, n_tokens
