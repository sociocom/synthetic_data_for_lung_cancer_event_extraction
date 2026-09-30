# -*- coding: utf-8 -*-
"""Document-level paired bootstrap CIs and paired permutation tests.

Input : data/paper/revision/calc/doc_key_counts.tsv  (from extract_doc_key_counts.py)
Output: data/paper/revision/calc/
  overall_ci.tsv        per condition: seed-averaged micro/macro F1 with 95% CI
  per_seed_ci.tsv       per condition x seed: micro/macro F1 with 95% CI
  diff_ci.tsv           paired differences with 95% CI and bootstrap p
  permutation.tsv       paired permutation p-values (doc-level label swap)
  per_item_ci.tsv       per item x condition: F1 with 95% CI (+ FT-ALL - FT-Manual diff CI)

Protocol
- 37 evaluation items (EVAL_KEY_TYPES minus p63/Reaction/ToxicityGrade), TBB/TBLB merged, strict matching.
- Resampling unit = document (n=96). The same resample indices are applied to every
  condition and seed, so differences are paired.
- Multi-seed conditions: the statistic is the mean over the 5 seeds of the per-seed
  metric (this is exactly how Table 2 reports the point estimate). CIs are percentile.
- Macro-F1 in a resample averages over the items that have at least one gold or predicted
  value in that resample (as in src.score_lung, absent items are not averaged in). Items with
  1-3 reference instances can be absent from a resample.
- B = 10,000 bootstrap resamples, 10,000 permutations, rng seed 20260924.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paper_revision.runs import EXCLUDED_KEYS, CONDITION_LABELS, REPO, available_llm_conditions  # noqa: E402

CALC = REPO / "data/paper/revision/calc"
B = 10_000
N_PERM = 10_000
RNG_SEED = 20260924
CONDS = ["zero_shot", "few_shot", "ft_synthetic", "ft_mock", "ft_all"]
DIFFS = [("ft_all", "ft_mock"), ("ft_synthetic", "zero_shot"), ("ft_all", "ft_synthetic"),
         ("ft_mock", "ft_synthetic"), ("ft_synthetic", "few_shot"), ("few_shot", "zero_shot")]
# exp09 prompted-LLM baselines (added when their predictions are registered in runs.py)
LLM_CONDS = available_llm_conditions()
CONDS = CONDS + LLM_CONDS
for _c in LLM_CONDS:
    DIFFS += [("ft_all", _c), ("ft_synthetic", _c), (_c, "few_shot")]
if "llm_gptoss" in LLM_CONDS and "llm_weblab" in LLM_CONDS:
    DIFFS.append(("llm_weblab", "llm_gptoss"))


def load():
    df = pd.read_csv(CALC / "doc_key_counts.tsv", sep="\t", dtype={"seed": str, "fold": str}, keep_default_na=False)
    df = df[~df.key.isin(EXCLUDED_KEYS)].copy()
    docs = sorted(df.doc_id.unique())
    keys = sorted(df.key.unique())
    assert len(docs) == 96 and len(keys) == 37, (len(docs), len(keys))
    di = {d: i for i, d in enumerate(docs)}
    ki = {k: i for i, k in enumerate(keys)}
    arrays = {}  # cond -> dict(tp,fp,fn) each (S, D, K)
    for cond in CONDS:
        sub = df[df.condition == cond]
        seeds = sorted(sub.seed.unique())
        S = len(seeds)
        tp = np.zeros((S, 96, 37)); fp = np.zeros((S, 96, 37)); fn = np.zeros((S, 96, 37))
        for s_i, s in enumerate(seeds):
            ss = sub[sub.seed == s]
            r = ss.doc_id.map(di).to_numpy(); c = ss.key.map(ki).to_numpy()
            tp[s_i, r, c] = ss.tp.to_numpy(); fp[s_i, r, c] = ss.fp.to_numpy(); fn[s_i, r, c] = ss.fn.to_numpy()
        arrays[cond] = {"tp": tp, "fp": fp, "fn": fn, "seeds": seeds}
    return arrays, docs, keys


def f1_from(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def macro_from(per_key_f1, support):
    """Mean F1 over keys that have any gold or predicted item in the (re)sample.
    Mirrors src.score_lung, where a key absent from both gold and pred is not part of per_key."""
    present = support > 0
    return (per_key_f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def metrics_weighted(arr, W):
    """W: (B, D) document weights. Returns micro (S,B), macro (S,B), per-key F1 (S,B,K)."""
    tp = np.einsum("bd,sdk->sbk", W, arr["tp"])
    fp = np.einsum("bd,sdk->sbk", W, arr["fp"])
    fn = np.einsum("bd,sdk->sbk", W, arr["fn"])
    micro = f1_from(tp.sum(-1), fp.sum(-1), fn.sum(-1))
    per_key = f1_from(tp, fp, fn)
    macro = macro_from(per_key, tp + fp + fn)
    return micro, macro, per_key


def ci(x, axis=0):
    lo, hi = np.percentile(x, [2.5, 97.5], axis=axis)
    return lo, hi


def main():
    arrays, docs, keys = load()
    rng = np.random.default_rng(RNG_SEED)
    D = 96
    idx = rng.integers(0, D, size=(B, D))
    W = np.zeros((B, D))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=D)
    point = {}
    boot = {}
    for cond in CONDS:
        arr = arrays[cond]
        pm, pM, pk = metrics_weighted(arr, np.ones((1, D)))  # point estimate (S,1,...)
        point[cond] = {"micro": pm[:, 0], "macro": pM[:, 0], "per_key": pk[:, 0, :]}
        bm, bM, bk = metrics_weighted(arr, W)
        boot[cond] = {"micro": bm, "macro": bM, "per_key": bk}
        print(f"bootstrapped {cond}", file=sys.stderr)

    # ---- overall (seed-averaged) ----
    rows = []
    for cond in CONDS:
        for met in ("micro", "macro"):
            pt = point[cond][met].mean()
            dist = boot[cond][met].mean(0)  # (B,)
            lo, hi = ci(dist)
            rows.append({"condition": CONDITION_LABELS[cond], "metric": f"{met}_f1", "n_seeds": len(arrays[cond]["seeds"]),
                         "point": pt, "ci_low": lo, "ci_high": hi,
                         "seed_sd": point[cond][met].std(ddof=1) if len(point[cond][met]) > 1 else float("nan"),
                         "seed_min": point[cond][met].min(), "seed_max": point[cond][met].max()})
    pd.DataFrame(rows).to_csv(CALC / "overall_ci.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- per seed ----
    rows = []
    for cond in CONDS:
        for s_i, s in enumerate(arrays[cond]["seeds"]):
            for met in ("micro", "macro"):
                lo, hi = ci(boot[cond][met][s_i])
                rows.append({"condition": CONDITION_LABELS[cond], "seed": s, "metric": f"{met}_f1",
                             "point": point[cond][met][s_i], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(CALC / "per_seed_ci.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- paired differences ----
    rows = []
    for a, b_ in DIFFS:
        for met in ("micro", "macro"):
            pt = point[a][met].mean() - point[b_][met].mean()
            dist = boot[a][met].mean(0) - boot[b_][met].mean(0)
            lo, hi = ci(dist)
            p_boot = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
            rows.append({"comparison": f"{CONDITION_LABELS[a]} - {CONDITION_LABELS[b_]}", "metric": f"{met}_f1",
                         "point": pt, "ci_low": lo, "ci_high": hi, "p_bootstrap_two_sided": min(1.0, p_boot),
                         "ci_excludes_zero": (lo > 0) or (hi < 0)})
    pd.DataFrame(rows).to_csv(CALC / "diff_ci.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- paired permutation test (doc-level label swap) ----
    rows = []
    for a, b_ in DIFFS:
        A, Bc = arrays[a], arrays[b_]
        S = max(A["tp"].shape[0], Bc["tp"].shape[0])
        def bcast(x, S=S):
            return np.repeat(x, S, axis=0) if x.shape[0] == 1 else x
        tpA, fpA, fnA = (bcast(A[k]) for k in ("tp", "fp", "fn"))
        tpB, fpB, fnB = (bcast(Bc[k]) for k in ("tp", "fp", "fn"))
        obs_m = point[a]["micro"].mean() - point[b_]["micro"].mean()
        obs_M = point[a]["macro"].mean() - point[b_]["macro"].mean()
        cnt_m = cnt_M = 0
        flips = rng.integers(0, 2, size=(N_PERM, D)).astype(bool)
        for i in range(N_PERM):
            f = flips[i][None, :, None]
            tpa = np.where(f, tpB, tpA); fpa = np.where(f, fpB, fpA); fna = np.where(f, fnB, fnA)
            tpb = np.where(f, tpA, tpB); fpb = np.where(f, fpA, fpB); fnb = np.where(f, fnA, fnB)
            dm = (f1_from(tpa.sum((1, 2)), fpa.sum((1, 2)), fna.sum((1, 2))) - f1_from(tpb.sum((1, 2)), fpb.sum((1, 2)), fnb.sum((1, 2)))).mean()
            dM = (macro_from(f1_from(tpa.sum(1), fpa.sum(1), fna.sum(1)), (tpa + fpa + fna).sum(1))
                  - macro_from(f1_from(tpb.sum(1), fpb.sum(1), fnb.sum(1)), (tpb + fpb + fnb).sum(1))).mean()
            cnt_m += abs(dm) >= abs(obs_m) - 1e-12
            cnt_M += abs(dM) >= abs(obs_M) - 1e-12
        rows.append({"comparison": f"{CONDITION_LABELS[a]} - {CONDITION_LABELS[b_]}", "metric": "micro_f1", "observed": obs_m, "p_permutation": (cnt_m + 1) / (N_PERM + 1)})
        rows.append({"comparison": f"{CONDITION_LABELS[a]} - {CONDITION_LABELS[b_]}", "metric": "macro_f1", "observed": obs_M, "p_permutation": (cnt_M + 1) / (N_PERM + 1)})
        print(f"permutation {a}-{b_} done", file=sys.stderr)
    pd.DataFrame(rows).to_csv(CALC / "permutation.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- per item ----
    gold_n = (arrays["ft_all"]["tp"][0] + arrays["ft_all"]["fn"][0]).sum(0)  # (K,) reference instances
    rows = []
    for k_i, key in enumerate(keys):
        row = {"item": key, "reference_instances": int(gold_n[k_i])}
        for cond in CONDS:
            pt = point[cond]["per_key"][:, k_i].mean()
            dist = boot[cond]["per_key"][:, :, k_i].mean(0)
            lo, hi = ci(dist)
            lab = CONDITION_LABELS[cond]
            row[f"{lab}_f1"] = pt; row[f"{lab}_ci_low"] = lo; row[f"{lab}_ci_high"] = hi
        d = boot["ft_all"]["per_key"][:, :, k_i].mean(0) - boot["ft_mock"]["per_key"][:, :, k_i].mean(0)
        lo, hi = ci(d)
        row["FT-ALL_minus_FT-Manual"] = point["ft_all"]["per_key"][:, k_i].mean() - point["ft_mock"]["per_key"][:, k_i].mean()
        row["diff_ci_low"] = lo; row["diff_ci_high"] = hi; row["diff_ci_excludes_zero"] = (lo > 0) or (hi < 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(CALC / "per_item_ci.tsv", sep="\t", index=False, float_format="%.6f")
    print("done")


if __name__ == "__main__":
    main()
