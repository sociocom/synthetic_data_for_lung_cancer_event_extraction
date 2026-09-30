# -*- coding: utf-8 -*-
"""Run registry for the JMIR revision recalculation.

Only the runs listed here are used. They were verified on 2026-09-24 to reproduce
the manuscript's Table 2 numbers (see data/paper/revision/01_recalculation_plan.md).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLD_JSONL = REPO / "data/processed/simulation/simu_annotation_noae.jsonl"

SFT_BASE = Path("/path/to/results/sft")
EXP08_RUNS = {
    42: SFT_BASE / "exp08_multiseed_robustness_20260427_014246",
    43: SFT_BASE / "exp08_multiseed_robustness_20260427_014843",
    44: SFT_BASE / "exp08_multiseed_robustness_20260427_014903",
    45: SFT_BASE / "exp08_multiseed_robustness_20260427_111859",
    46: SFT_BASE / "exp08_multiseed_robustness_20260427_112347",
}
ZERO_SHOT_PRED = REPO / "results/zero_shot/zero_shot_20260306_134052/inference/pred_raw.jsonl"
FEW_SHOT_PRED = REPO / "results/few_shot/few_shot_llama_20260306_031618/inference/pred_raw.jsonl"
# Verified 2026-09-24: re-running 05_few_shot with the current config (Llama) reproduced all 96
# predictions byte-for-byte -> results/few_shot/few_shot_llama_verify_20260924_200910

# exp09 prompted-LLM baselines (2-shot, same prompt/ICL/seed as few_shot; bf16; greedy; reasoning low).
# Fill in after each run finishes and its score/ dir exists. Missing entries are skipped downstream.
LLM_BASELINE_PREDS: dict[str, Path | None] = {
    "llm_gptoss": REPO / "results/llm_baselines/gptoss120b_fewshot_20260928_090512/inference/pred_raw.jsonl",  # 2026-09-28, 96 docs, 96/96 final channel
    "llm_weblab": REPO / "results/llm_baselines/weblab_medllm120b_fewshot_20260928_112416/inference/pred_reparsed.jsonl",  # 2026-09-28, 96 docs; final channel missing in some docs -> lenient fallback (pred_strict.jsonl = strict variant)
}


def available_llm_conditions() -> list[str]:
    return [c for c, p in LLM_BASELINE_PREDS.items() if p is not None and Path(p).exists()]


# Manuscript naming: FT-Mock (internal) == FT-Manual (paper)
CONDITION_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "ft_synthetic": "FT-Synthetic",
    "ft_mock": "FT-Manual",
    "ft_all": "FT-ALL",
    "llm_gptoss": "Few-shot (gpt-oss-120b)",
    "llm_weblab": "Few-shot (Weblab-MedLLM)",
}

# 37 evaluation items used in the manuscript (Table 1):
# EVAL_KEY_TYPES (40) minus p63, Reaction, ToxicityGrade. TBB/TBLB are merged as TBB_TBLB.
EXCLUDED_KEYS = {"p63", "Reaction", "ToxicityGrade"}


def pred_files(condition: str, seed: int | None = None) -> list[Path]:
    if condition == "zero_shot":
        return [ZERO_SHOT_PRED]
    if condition == "few_shot":
        return [FEW_SHOT_PRED]
    if condition in LLM_BASELINE_PREDS:
        p = LLM_BASELINE_PREDS[condition]
        assert p is not None, f"{condition} not registered yet"
        return [Path(p)]
    assert seed is not None
    run = EXP08_RUNS[seed]
    seed_dir = run / f"seed={seed}"
    if condition == "ft_synthetic":
        return [seed_dir / "ft_synthetic/inference/pred_test_test.jsonl"]
    if condition in ("ft_mock", "ft_all"):
        return [seed_dir / condition / f"folds/fold={k}/inference/pred_test_test.jsonl" for k in range(5)]
    raise ValueError(condition)


def fold_of_doc(seed: int, condition: str) -> dict[str, int]:
    """Map doc id -> fold index for CV conditions (from the fold prediction files)."""
    import json
    out: dict[str, int] = {}
    for k, f in enumerate(pred_files(condition, seed)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out[str(json.loads(line)["id"])] = k
    return out
