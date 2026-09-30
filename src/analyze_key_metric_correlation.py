#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
DOC_PRESENCE_PATH = ROOT / "docs/analysis/synthetic_38keys_doc_presence_merged_tbb_tblb.tsv"
LIST_BIAS_PATH = ROOT / "docs/analysis/synthetic_38keys_list_value_bias_20260325.tsv"
METRICS_PATH = ROOT / "docs/analysis/metrics_exp01_exp02_exp03_exp04_38keys_strict_merged_tbb_tblb.tsv"

MERGED_OUT_PATH = ROOT / "docs/analysis/key_correlation_input_20260408.tsv"
CORR_OUT_PATH = ROOT / "docs/analysis/key_correlation_results_20260408.tsv"
QUADRANT_OUT_PATH = ROOT / "docs/analysis/key_quadrant_analysis_20260409.tsv"
REPORT_OUT_PATH = ROOT / "docs/reports/report_20260408_key_metric_correlation.md"


BINARY_MARKER_KEYS = {
    "EGFR",
    "ALK",
    "ROS1",
    "RET",
    "BRAF",
    "KRAS",
    "HER2",
    "NTRK",
    "TTF1",
    "p40",
    "CK5_6",
    "NapsinA",
    "Oncomine",
    "NGS",
}

MULTI_VALUED_COMPLEX_KEYS = {
    "ProcedureDiagnosis",
    "Site",
    "TNM_T",
    "TNM_N",
    "TNM_M",
    "Stage",
    "SitesOfMetastasis",
    "SiteOfRecurrence",
    "HistologicalGrading",
    "Cytology",
    "TBB_TBLB",
    "EBUS_TBNA",
    "PathologicalExamination",
    "AssessedAnatomicSite",
    "ROSE",
    "Treatment",
}


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_float(value: str) -> float | None:
    if value in {"", "NA", "NaN", "nan", None}:
        return None
    return float(value)


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearsonr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denom_x = math.sqrt(sum(v * v for v in dx))
    denom_y = math.sqrt(sum(v * v for v in dy))
    if denom_x == 0 or denom_y == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def spearmanr(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearsonr(rankdata(x), rankdata(y))


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_group(key: str) -> str:
    if key in BINARY_MARKER_KEYS:
        return "binary_marker"
    if key in MULTI_VALUED_COMPLEX_KEYS:
        return "multi_valued_complex"
    return "other"


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quadrant_label(exp03_f1: float, exp02_minus_exp03: float, exp03_median: float) -> Tuple[str, str]:
    high_exp03 = exp03_f1 >= exp03_median
    synthetic_keeps_up = exp02_minus_exp03 >= 0.0
    if high_exp03 and synthetic_keeps_up:
        return ("A", "easy_and_synthetic_compatible")
    if high_exp03 and not synthetic_keeps_up:
        return ("B", "easy_but_synthetic_mismatched")
    if (not high_exp03) and synthetic_keeps_up:
        return ("D", "difficult_but_helped_by_synthetic")
    return ("C", "intrinsically_difficult")


def main() -> None:
    doc_presence = {row["key"]: row for row in read_tsv(DOC_PRESENCE_PATH)}
    list_bias = {row["key"]: row for row in read_tsv(LIST_BIAS_PATH)}
    metrics = {row["key"]: row for row in read_tsv(METRICS_PATH)}

    merged_rows: List[Dict[str, object]] = []
    for key, metric_row in metrics.items():
        if key == "p63":
            continue
        doc_row = doc_presence.get(key)
        bias_row = list_bias.get(key)
        row: Dict[str, object] = {
            "key": key,
            "group": classify_group(key),
            "has_list_bias": int(bias_row is not None),
            "allowed_count": bias_row["allowed_count"] if bias_row else "",
            "docs_with_key": doc_row["docs_with_key"] if doc_row else "",
            "doc_coverage": doc_row["doc_coverage"] if doc_row else "",
            "token_total": bias_row["token_total"] if bias_row else "",
            "allowed_coverage_ratio": bias_row["allowed_coverage_ratio"] if bias_row else "",
            "normalized_entropy": bias_row["normalized_entropy"] if bias_row else "",
            "top1_share": bias_row["top1_share"] if bias_row else "",
            "exp01_f1": metric_row["exp01_f1"],
            "exp02_f1": metric_row["exp02_f1"],
            "exp03_f1": metric_row["exp03_f1"],
            "exp04_f1": metric_row["exp04_f1"],
            "exp02_minus_exp01_f1": "",
            "exp04_minus_exp03_f1": "",
            "gold_count": metric_row["exp02_gold"],
        }

        exp01_f1 = to_float(metric_row["exp01_f1"])
        exp02_f1 = to_float(metric_row["exp02_f1"])
        exp03_f1 = to_float(metric_row["exp03_f1"])
        exp04_f1 = to_float(metric_row["exp04_f1"])
        if exp01_f1 is not None and exp02_f1 is not None:
            row["exp02_minus_exp01_f1"] = f"{exp02_f1 - exp01_f1:.6f}"
        if exp02_f1 is not None and exp03_f1 is not None:
            row["exp02_minus_exp03_f1"] = f"{exp02_f1 - exp03_f1:.6f}"
        else:
            row["exp02_minus_exp03_f1"] = ""
        if exp03_f1 is not None and exp04_f1 is not None:
            row["exp04_minus_exp03_f1"] = f"{exp04_f1 - exp03_f1:.6f}"

        docs_with_key = to_float(str(row["docs_with_key"])) if row["docs_with_key"] != "" else None
        token_total = to_float(str(row["token_total"])) if row["token_total"] != "" else None
        row["log10_docs_with_key"] = f"{math.log10(docs_with_key):.6f}" if docs_with_key and docs_with_key > 0 else ""
        row["log10_token_total"] = f"{math.log10(token_total):.6f}" if token_total and token_total > 0 else ""
        merged_rows.append(row)

    merged_fieldnames = [
        "key",
        "group",
        "has_list_bias",
        "allowed_count",
        "docs_with_key",
        "doc_coverage",
        "token_total",
        "log10_docs_with_key",
        "log10_token_total",
        "allowed_coverage_ratio",
        "normalized_entropy",
        "top1_share",
        "exp01_f1",
        "exp02_f1",
        "exp03_f1",
        "exp04_f1",
        "exp02_minus_exp01_f1",
        "exp02_minus_exp03_f1",
        "exp04_minus_exp03_f1",
        "gold_count",
    ]
    write_tsv(MERGED_OUT_PATH, merged_rows, merged_fieldnames)

    all_rows = merged_rows
    non_timestamp_rows = [row for row in merged_rows if row["key"] != "Timestamp"]
    gold10_rows = [row for row in non_timestamp_rows if to_float(str(row["gold_count"])) is not None and to_float(str(row["gold_count"])) >= 10]
    gold20_rows = [row for row in non_timestamp_rows if to_float(str(row["gold_count"])) is not None and to_float(str(row["gold_count"])) >= 20]

    quadrant_source_rows = [
        row for row in non_timestamp_rows
        if to_float(str(row["exp02_f1"])) is not None and to_float(str(row["exp03_f1"])) is not None
    ]
    exp03_median = median([
        to_float(str(row["exp03_f1"]))
        for row in quadrant_source_rows
        if to_float(str(row["exp03_f1"])) is not None
    ])
    quadrant_rows: List[Dict[str, object]] = []
    for row in quadrant_source_rows:
        exp02 = to_float(str(row["exp02_f1"]))
        exp03 = to_float(str(row["exp03_f1"]))
        delta_02_03 = exp02 - exp03
        q_short, q_long = quadrant_label(exp03, delta_02_03, exp03_median)
        row["quadrant"] = q_short
        row["quadrant_label"] = q_long
        quadrant_rows.append({
            "key": row["key"],
            "group": row["group"],
            "quadrant": q_short,
            "quadrant_label": q_long,
            "exp02_f1": f"{exp02:.6f}",
            "exp03_f1": f"{exp03:.6f}",
            "exp04_f1": str(row["exp04_f1"]),
            "exp02_minus_exp03_f1": f"{delta_02_03:.6f}",
            "exp04_minus_exp03_f1": str(row["exp04_minus_exp03_f1"]),
            "allowed_count": str(row["allowed_count"]),
            "docs_with_key": str(row["docs_with_key"]),
            "token_total": str(row["token_total"]),
            "gold_count": str(row["gold_count"]),
        })
    quadrant_fieldnames = [
        "key",
        "group",
        "quadrant",
        "quadrant_label",
        "exp02_f1",
        "exp03_f1",
        "exp04_f1",
        "exp02_minus_exp03_f1",
        "exp04_minus_exp03_f1",
        "allowed_count",
        "docs_with_key",
        "token_total",
        "gold_count",
    ]
    write_tsv(QUADRANT_OUT_PATH, quadrant_rows, quadrant_fieldnames)

    datasets = [
        ("all_keys", all_rows),
        ("non_timestamp", non_timestamp_rows),
        ("non_timestamp_gold_ge_10", gold10_rows),
        ("non_timestamp_gold_ge_20", gold20_rows),
        ("binary_marker", [row for row in merged_rows if row["group"] == "binary_marker"]),
        ("multi_valued_complex", [row for row in merged_rows if row["group"] == "multi_valued_complex"]),
        (
            "binary_marker_gold_ge_10",
            [row for row in merged_rows if row["group"] == "binary_marker" and to_float(str(row["gold_count"])) is not None and to_float(str(row["gold_count"])) >= 10],
        ),
        (
            "multi_valued_complex_gold_ge_10",
            [row for row in merged_rows if row["group"] == "multi_valued_complex" and to_float(str(row["gold_count"])) is not None and to_float(str(row["gold_count"])) >= 10],
        ),
    ]
    predictors = [
        "allowed_count",
        "log10_docs_with_key",
        "log10_token_total",
        "allowed_coverage_ratio",
        "normalized_entropy",
        "top1_share",
    ]
    outcomes = [
        "exp02_f1",
        "exp02_minus_exp01_f1",
        "exp04_minus_exp03_f1",
        "exp03_f1",
    ]

    corr_rows: List[Dict[str, object]] = []
    for dataset_name, rows in datasets:
        for predictor in predictors:
            for outcome in outcomes:
                xy: List[Tuple[float, float]] = []
                for row in rows:
                    x = to_float(str(row[predictor])) if row[predictor] != "" else None
                    y = to_float(str(row[outcome])) if row[outcome] != "" else None
                    if x is None or y is None:
                        continue
                    xy.append((x, y))
                xs = [x for x, _ in xy]
                ys = [y for _, y in xy]
                spearman = spearmanr(xs, ys) if xy else None
                pearson = pearsonr(xs, ys) if xy else None
                corr_rows.append({
                    "dataset": dataset_name,
                    "n": len(xy),
                    "predictor": predictor,
                    "outcome": outcome,
                    "spearman_r": "" if spearman is None else f"{spearman:.6f}",
                    "pearson_r": "" if pearson is None else f"{pearson:.6f}",
                })

    corr_fieldnames = ["dataset", "n", "predictor", "outcome", "spearman_r", "pearson_r"]
    write_tsv(CORR_OUT_PATH, corr_rows, corr_fieldnames)

    corr_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in corr_rows:
        corr_lookup[(str(row["dataset"]), str(row["predictor"]), str(row["outcome"]))] = row

    def get_corr(dataset: str, predictor: str, outcome: str) -> str:
        row = corr_lookup[(dataset, predictor, outcome)]
        return format_float(to_float(str(row["spearman_r"])))

    non_timestamp_sorted = sorted(
        non_timestamp_rows,
        key=lambda r: to_float(str(r["exp02_f1"])) if to_float(str(r["exp02_f1"])) is not None else -1.0,
    )
    low_exp02 = non_timestamp_sorted[:5]
    high_exp02 = list(reversed(non_timestamp_sorted[-5:]))

    residual_rows = []
    docs_x = []
    f1_y = []
    valid_rows = []
    for row in non_timestamp_rows:
        x = to_float(str(row["log10_docs_with_key"]))
        y = to_float(str(row["exp02_f1"]))
        if x is None or y is None:
            continue
        docs_x.append(x)
        f1_y.append(y)
        valid_rows.append(row)
    mean_x = sum(docs_x) / len(docs_x)
    mean_y = sum(f1_y) / len(f1_y)
    sxx = sum((x - mean_x) ** 2 for x in docs_x)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(docs_x, f1_y)) / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    for row in valid_rows:
        x = to_float(str(row["log10_docs_with_key"]))
        y = to_float(str(row["exp02_f1"]))
        pred = intercept + slope * x
        residual_rows.append({
            "key": row["key"],
            "exp02_f1": y,
            "predicted_exp02_f1_from_doc_count": pred,
            "residual": y - pred,
            "docs_with_key": int(float(str(row["docs_with_key"]))),
            "normalized_entropy": to_float(str(row["normalized_entropy"])) if row["normalized_entropy"] != "" else None,
            "top1_share": to_float(str(row["top1_share"])) if row["top1_share"] != "" else None,
        })
    most_under = sorted(residual_rows, key=lambda r: r["residual"])[:5]
    most_over = sorted(residual_rows, key=lambda r: r["residual"], reverse=True)[:5]
    binary_rows = [row for row in merged_rows if row["group"] == "binary_marker"]
    complex_rows = [row for row in merged_rows if row["group"] == "multi_valued_complex"]
    binary_sorted = sorted(binary_rows, key=lambda r: to_float(str(r["exp02_f1"])) if to_float(str(r["exp02_f1"])) is not None else -1.0)
    complex_sorted = sorted(complex_rows, key=lambda r: to_float(str(r["exp02_f1"])) if to_float(str(r["exp02_f1"])) is not None else -1.0)
    quadrant_groups = {
        "A": [row for row in quadrant_source_rows if row.get("quadrant") == "A"],
        "B": [row for row in quadrant_source_rows if row.get("quadrant") == "B"],
        "C": [row for row in quadrant_source_rows if row.get("quadrant") == "C"],
        "D": [row for row in quadrant_source_rows if row.get("quadrant") == "D"],
    }

    report_lines: List[str] = []
    report_lines.append("# Report 2026-04-08: 合成データ項目量・分布指標と項目別F1の相関")
    report_lines.append("")
    report_lines.append("## 1. 目的")
    report_lines.append("- 付録表の「全項目が100文書以上に出現した」という記述を補い、synthetic data の各項目の学習機会と FT-Synthetic の項目別性能の関係を探索的に確認する。")
    report_lines.append("- 特に、文書出現数だけでなく、出現回数、候補値 coverage、値分布の多様性・偏り、候補空間の広さが F1 とどう関係するかを調べる。")
    report_lines.append("")
    report_lines.append("## 2. 入力")
    report_lines.append(f"- `docs_with_key`, `doc_coverage`: `{DOC_PRESENCE_PATH.relative_to(ROOT)}`")
    report_lines.append(f"- `allowed_count`, `token_total`, `allowed_coverage_ratio`, `normalized_entropy`, `top1_share`: `{LIST_BIAS_PATH.relative_to(ROOT)}`")
    report_lines.append(f"- per-key F1: `{METRICS_PATH.relative_to(ROOT)}`")
    report_lines.append("- 対象は `p63` を除く 38 項目。`TBB`/`TBLB` は既存集計に合わせて `TBB_TBLB` として扱った。")
    report_lines.append("- 層別解析では、`binary_marker` を遺伝子・免疫染色・分子マーカー系の二値項目群、`multi_valued_complex` を候補値数が多く意味的にも複雑な list 項目群として手作業で定義した。")
    report_lines.append("- 主目的変数は `exp02_f1`（FT-Synthetic の項目別 F1）。補助的に `exp02_minus_exp01_f1`, `exp04_minus_exp03_f1`, `exp03_f1` も見た。")
    report_lines.append("- 相関は Spearman を主、Pearson を参考として算出した。")
    report_lines.append("")
    report_lines.append("## 3. 主結果")
    report_lines.append("")
    report_lines.append("### 3.1 FT-Synthetic の項目別F1との相関")
    report_lines.append("")
    report_lines.append("| dataset | predictor | outcome | Spearman r |")
    report_lines.append("|---|---|---:|---:|")
    for dataset in ["all_keys", "non_timestamp", "non_timestamp_gold_ge_10", "non_timestamp_gold_ge_20"]:
        for predictor in predictors:
            report_lines.append(
                f"| {dataset} | {predictor} | exp02_f1 | {get_corr(dataset, predictor, 'exp02_f1')} |"
            )
    report_lines.append("")
    report_lines.append("要点:")
    report_lines.append(
        f"- `exp02_f1` と `log10_docs_with_key` の Spearman 相関は、`Timestamp` を除くと {get_corr('non_timestamp', 'log10_docs_with_key', 'exp02_f1')}、`gold_count >= 10` に限定すると {get_corr('non_timestamp_gold_ge_10', 'log10_docs_with_key', 'exp02_f1')} で、少なくとも今回の per-key 集計では明確な正相関は確認できなかった。"
    )
    report_lines.append(
        f"- `exp02_f1` と `log10_token_total` の相関も {get_corr('non_timestamp', 'log10_token_total', 'exp02_f1')} と弱く、単純な出現量だけでは項目別 F1 は説明しにくかった。"
    )
    report_lines.append(
        f"- `allowed_count` は `exp02_f1` と {get_corr('non_timestamp', 'allowed_count', 'exp02_f1')} の負相関で、候補空間が広い項目ほど難しい傾向がみられた。"
    )
    report_lines.append(
        f"- `allowed_coverage_ratio` は {get_corr('non_timestamp', 'allowed_coverage_ratio', 'exp02_f1')}、`normalized_entropy` は {get_corr('non_timestamp', 'normalized_entropy', 'exp02_f1')}、`top1_share` は {get_corr('non_timestamp', 'top1_share', 'exp02_f1')} で、直感とは逆向きの相関が出た。これは低エントロピー・高 top1_share の二値マーカー項目が高F1を示し、分布指標が“学習しやすさ”というより“項目難易度”を強く反映しているためと考えられる。"
    )
    report_lines.append("")
    report_lines.append("### 3.2 二値マーカー群と多値・複雑項目群に分けた相関")
    report_lines.append("")
    report_lines.append("| dataset | predictor | exp02_f1 Spearman r |")
    report_lines.append("|---|---|---:|")
    for dataset in [
        "binary_marker",
        "binary_marker_gold_ge_10",
        "multi_valued_complex",
        "multi_valued_complex_gold_ge_10",
    ]:
        for predictor in predictors:
            report_lines.append(f"| {dataset} | {predictor} | {get_corr(dataset, predictor, 'exp02_f1')} |")
    report_lines.append("")
    report_lines.append("要点:")
    report_lines.append(
        f"- `binary_marker` 群では `normalized_entropy` が {get_corr('binary_marker', 'normalized_entropy', 'exp02_f1')}、`top1_share` が {get_corr('binary_marker', 'top1_share', 'exp02_f1')} で、全項目一括と同様に“偏っているほど高F1”に見える。これは二値項目では偏りそのものが学習容易性と矛盾しないためである。"
    )
    report_lines.append(
        f"- `multi_valued_complex` 群では `log10_docs_with_key` が {get_corr('multi_valued_complex', 'log10_docs_with_key', 'exp02_f1')}、`allowed_count` が {get_corr('multi_valued_complex', 'allowed_count', 'exp02_f1')}、`allowed_coverage_ratio` が {get_corr('multi_valued_complex', 'allowed_coverage_ratio', 'exp02_f1')} で、単純な量だけでなく候補空間の広さや複雑さの影響が残った。"
    )
    report_lines.append(
        f"- `multi_valued_complex_gold_ge_10` でも `log10_docs_with_key` は {get_corr('multi_valued_complex_gold_ge_10', 'log10_docs_with_key', 'exp02_f1')} と弱く、複雑項目群に限っても「たくさん出たから高F1」とは言えなかった。"
    )
    report_lines.append("")
    report_lines.append("### 3.3 改善量との相関")
    report_lines.append("")
    report_lines.append("| predictor | exp02_minus_exp01_f1 | exp04_minus_exp03_f1 | exp03_f1 |")
    report_lines.append("|---|---:|---:|---:|")
    for predictor in predictors:
        report_lines.append(
            f"| {predictor} | {get_corr('non_timestamp', predictor, 'exp02_minus_exp01_f1')} | {get_corr('non_timestamp', predictor, 'exp04_minus_exp03_f1')} | {get_corr('non_timestamp', predictor, 'exp03_f1')} |"
        )
    report_lines.append("")
    report_lines.append("要点:")
    report_lines.append(
        f"- synthetic 指標と `exp02_minus_exp01_f1` の相関も `exp02_f1` と同様に単純ではなく、Zero-shot からの改善量も量だけでは整理できなかった。"
    )
    report_lines.append(
        f"- 一方で `exp04_minus_exp03_f1` との相関は全体に弱く、two-stage の追加利得は単純な出現量だけでは説明しにくかった。"
    )
    report_lines.append("")
    report_lines.append("### 3.4 `exp03_f1` と `exp02-exp03` による4象限整理")
    report_lines.append("")
    report_lines.append(f"- 横軸: `exp03_f1`（mock-only の項目別 F1, 項目固有難易度の proxy）")
    report_lines.append(f"- 縦軸: `exp02_minus_exp03_f1`（synthetic-only が mock-only に対してどれだけ良いか悪いか）")
    report_lines.append(f"- `exp03_f1` の high/low は中央値 `{exp03_median:.3f}`、`exp02_minus_exp03_f1` の high/low は `0` を閾値にした。")
    report_lines.append("")
    report_lines.append("| quadrant | label | interpretation | n_keys |")
    report_lines.append("|---|---|---|---:|")
    report_lines.append(f"| A | easy_and_synthetic_compatible | 項目自体も比較的易しく、synthetic でも十分学べる | {len(quadrant_groups['A'])} |")
    report_lines.append(f"| B | easy_but_synthetic_mismatched | 項目自体は取れるが、synthetic supervision が相対的に合っていない | {len(quadrant_groups['B'])} |")
    report_lines.append(f"| C | intrinsically_difficult | mock-only でも synthetic-only でも難しい | {len(quadrant_groups['C'])} |")
    report_lines.append(f"| D | difficult_but_helped_by_synthetic | 項目自体は難しいが、synthetic が相対的に助けている | {len(quadrant_groups['D'])} |")
    report_lines.append("")
    report_lines.append("要点:")
    report_lines.append("- この整理は、合成データの量や分布だけではなく、項目固有の難しさと synthetic supervision の適合度を分けてみるためのものである。")
    report_lines.append("- 特に B 群は「項目自体は不可能ではないのに synthetic では崩れる」項目で、synthetic note の表現や自動 annotation のずれを疑う根拠になる。")
    report_lines.append("- D 群は、two-stage 学習の意義を説明しやすい候補群である。")
    report_lines.append("")
    report_lines.append("## 4. 項目別の見え方")
    report_lines.append("")
    report_lines.append("### 4.1 FT-Synthetic で低かった項目（non-Timestamp, exp02_f1 下位5件）")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | docs_with_key | token_total | allowed_coverage_ratio | normalized_entropy | top1_share |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in low_exp02:
        report_lines.append(
            f"| {row['key']} | {format_float(to_float(str(row['exp02_f1'])))} | {row['docs_with_key']} | {row['token_total'] or 'NA'} | {format_float(to_float(str(row['allowed_coverage_ratio'])))} | {format_float(to_float(str(row['normalized_entropy'])))} | {format_float(to_float(str(row['top1_share'])))} |"
        )
    report_lines.append("")
    report_lines.append("### 4.2 FT-Synthetic で高かった項目（non-Timestamp, exp02_f1 上位5件）")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | docs_with_key | token_total | allowed_coverage_ratio | normalized_entropy | top1_share |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in high_exp02:
        report_lines.append(
            f"| {row['key']} | {format_float(to_float(str(row['exp02_f1'])))} | {row['docs_with_key']} | {row['token_total'] or 'NA'} | {format_float(to_float(str(row['allowed_coverage_ratio'])))} | {format_float(to_float(str(row['normalized_entropy'])))} | {format_float(to_float(str(row['top1_share'])))} |"
        )
    report_lines.append("")
    report_lines.append("### 4.3 binary_marker 群と multi_valued_complex 群の内訳")
    report_lines.append("")
    report_lines.append(f"- `binary_marker` ({len(binary_rows)} keys): {', '.join(row['key'] for row in binary_rows)}")
    report_lines.append(f"- `multi_valued_complex` ({len(complex_rows)} keys): {', '.join(row['key'] for row in complex_rows)}")
    report_lines.append("")
    report_lines.append("### 4.4 binary_marker 群で FT-Synthetic が低かった項目")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | docs_with_key | token_total | normalized_entropy | top1_share |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for row in binary_sorted[:5]:
        report_lines.append(
            f"| {row['key']} | {format_float(to_float(str(row['exp02_f1'])))} | {row['docs_with_key']} | {row['token_total'] or 'NA'} | {format_float(to_float(str(row['normalized_entropy'])))} | {format_float(to_float(str(row['top1_share'])))} |"
        )
    report_lines.append("")
    report_lines.append("### 4.5 multi_valued_complex 群で FT-Synthetic が低かった項目")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | docs_with_key | token_total | allowed_count | allowed_coverage_ratio | normalized_entropy |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in complex_sorted[:5]:
        report_lines.append(
            f"| {row['key']} | {format_float(to_float(str(row['exp02_f1'])))} | {row['docs_with_key']} | {row['token_total'] or 'NA'} | {row['allowed_count'] or 'NA'} | {format_float(to_float(str(row['allowed_coverage_ratio'])))} | {format_float(to_float(str(row['normalized_entropy'])))} |"
        )
    report_lines.append("")
    report_lines.append("### 4.6 文書出現数からの単純予測より悪かった項目（exp02_f1 residual 下位5件）")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | predicted_from_doc_count | residual | docs_with_key | normalized_entropy | top1_share |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in most_under:
        report_lines.append(
            f"| {row['key']} | {format_float(row['exp02_f1'])} | {format_float(row['predicted_exp02_f1_from_doc_count'])} | {format_float(row['residual'])} | {row['docs_with_key']} | {format_float(row['normalized_entropy'])} | {format_float(row['top1_share'])} |"
        )
    report_lines.append("")
    report_lines.append("### 4.7 文書出現数からの単純予測より良かった項目（exp02_f1 residual 上位5件）")
    report_lines.append("")
    report_lines.append("| key | exp02_f1 | predicted_from_doc_count | residual | docs_with_key | normalized_entropy | top1_share |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in most_over:
        report_lines.append(
            f"| {row['key']} | {format_float(row['exp02_f1'])} | {format_float(row['predicted_exp02_f1_from_doc_count'])} | {format_float(row['residual'])} | {row['docs_with_key']} | {format_float(row['normalized_entropy'])} | {format_float(row['top1_share'])} |"
        )
    report_lines.append("")
    report_lines.append("### 4.8 4象限ごとの代表項目")
    report_lines.append("")
    report_lines.append("| quadrant | representative keys |")
    report_lines.append("|---|---|")
    for quadrant in ["A", "B", "C", "D"]:
        reps = sorted(
            quadrant_groups[quadrant],
            key=lambda r: (
                -(to_float(str(r["exp03_f1"])) if to_float(str(r["exp03_f1"])) is not None else -999.0),
                -(to_float(str(r["exp02_minus_exp03_f1"])) if to_float(str(r["exp02_minus_exp03_f1"])) is not None else -999.0),
            ),
        )
        if quadrant in {"B", "C"}:
            reps = sorted(
                quadrant_groups[quadrant],
                key=lambda r: to_float(str(r["exp02_minus_exp03_f1"])) if to_float(str(r["exp02_minus_exp03_f1"])) is not None else 999.0,
            )
        rep_keys = ", ".join(row["key"] for row in reps[:5])
        report_lines.append(f"| {quadrant} | {rep_keys} |")
    report_lines.append("")
    report_lines.append("### 4.9 4象限分類表")
    report_lines.append("")
    report_lines.append("| key | quadrant | group | exp02_f1 | exp03_f1 | exp02-exp03 | exp04-exp03 |")
    report_lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in sorted(quadrant_source_rows, key=lambda r: (r["quadrant"], r["key"])):
        report_lines.append(
            f"| {row['key']} | {row['quadrant']} | {row['group']} | {format_float(to_float(str(row['exp02_f1'])))} | {format_float(to_float(str(row['exp03_f1'])))} | {format_float(to_float(str(row['exp02_minus_exp03_f1'])))} | {format_float(to_float(str(row['exp04_minus_exp03_f1'])))} |"
        )
    report_lines.append("")
    report_lines.append("解釈メモ:")
    report_lines.append("- `ProcedureDiagnosis` や `Treatment` のように出現量は大きいのに F1 が伸びない項目は、量不足よりも annotation の整合性や候補空間の複雑さが支配的である可能性が高い。")
    report_lines.append("- `NGS` は合成側では `normalized_entropy` が高く二値分布も偏り切っていない一方、`exp02_f1` は極端に低く、synthetic note と mock 評価の表現差を疑う項目として整合的である。")
    report_lines.append("- `RadiationDose`, `RadiationFractions`, `Stage`, `TNM_T/N/M` は比較的高い F1 を示したが、同じ高出現でも `Treatment` や `ProcedureDiagnosis` は低いため、出現量だけでなく定義の曖昧さや評価整合性が重要である。")
    report_lines.append("- 二値マーカー項目では、低エントロピー・高 top1_share でも高F1になりうる。そのため entropy や top1_share は、全項目一括では“データの良さ”ではなく“項目の単純さ”の proxy としても振る舞う。")
    report_lines.append("- したがって Discussion では、`binary_marker` 群と `multi_valued_complex` 群を分けて記述し、後者で外れ値となる `ProcedureDiagnosis`, `Treatment`, `NGS`, `SiteOfRecurrence` を重点的に解釈するのが自然である。")
    report_lines.append("- 4象限でみると、B 群は synthetic supervision の不整合、C 群は項目固有難易度、D 群は synthetic pretraining の潜在的な利得を示す候補として読める。")
    report_lines.append("")
    report_lines.append("## 5. まとめ")
    report_lines.append("- 「全項目が100文書以上に出現した」という事実だけでは不十分で、FT-Synthetic の項目別F1は、出現文書数・出現回数とも強くは結び付かなかった。")
    report_lines.append("- 候補値 coverage・entropy・top1_share も全項目一括では単純解釈できず、これらは合成データの質だけでなく、項目自体の複雑さや二値/多値の違いも反映していた。")
    report_lines.append("- それでも `ProcedureDiagnosis`, `Treatment`, `NGS` のような外れ値は残り、これらは生成された note の情報の現れ方や自動 annotation と評価基準の不整合を示唆する。")
    report_lines.append("")
    report_lines.append("## 6. 出力ファイル")
    report_lines.append(f"- `{MERGED_OUT_PATH.relative_to(ROOT)}`")
    report_lines.append(f"- `{CORR_OUT_PATH.relative_to(ROOT)}`")
    report_lines.append(f"- `{QUADRANT_OUT_PATH.relative_to(ROOT)}`")
    report_lines.append(f"- `{REPORT_OUT_PATH.relative_to(ROOT)}`")

    REPORT_OUT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
