# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoTokenizer


STRUCTURED_TEMPLATE = """\
<guideline>

### Input:
<text>

### Output:
"""


def make_build_text_fn(guideline_text: str, tokenizer: AutoTokenizer):
    template = STRUCTURED_TEMPLATE.replace("<guideline>", guideline_text)

    def build_text(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        texts: List[str] = []
        for note, ann in zip(batch["progress_note"], batch["annotation"]):
            prompt = template.replace("<text>", str(note))
            text = prompt + str(ann) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    return build_text


def _find_sublist(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last = len(haystack) - len(needle)
    for i in range(last + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def make_tok_fn(tokenizer: AutoTokenizer, max_length: int):
    answer_tag_with_nl = "### Output:\n"
    answer_tag = "### Output:"
    answer_ids_with_nl = tokenizer(answer_tag_with_nl, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_tag, add_special_tokens=False)["input_ids"]

    def tok_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

        labels = []
        for ids in enc["input_ids"]:
            pos = _find_sublist(ids, answer_ids_with_nl)
            if pos != -1:
                cut = pos + len(answer_ids_with_nl)
            else:
                pos2 = _find_sublist(ids, answer_ids)
                cut = (pos2 + len(answer_ids)) if pos2 != -1 else 0

            lab = ids.copy()
            for j in range(min(cut, len(lab))):
                lab[j] = -100
            labels.append(lab)

        enc["labels"] = labels
        return enc

    return tok_fn


def make_data_collator(tokenizer: AutoTokenizer):
    def data_collator(features: List[Dict]):
        labels = [f["labels"] for f in features]
        feats_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = tokenizer.pad(
            feats_wo_labels,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return data_collator
