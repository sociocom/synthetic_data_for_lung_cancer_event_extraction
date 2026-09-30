#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "docs/analysis/metrics_exp01_exp02_exp03_exp04_38keys_strict_merged_tbb_tblb.tsv"
QUADRANT_PATH = ROOT / "docs/analysis/key_quadrant_analysis_20260409.tsv"

DELTA_OUT_PATH = ROOT / "docs/analysis/exp03_exp04_per_key_delta_20260409.tsv"
REPORT_OUT_PATH = ROOT / "docs/reports/report_20260409_exp03_exp04_delta_analysis.md"


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: str) -> float | None:
    if value in {"", "NA", "NaN", "nan", None}:
        return None
    return float(value)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    metrics_rows = read_tsv(METRICS_PATH)
    quadrant_rows = {row["key"]: row for row in read_tsv(QUADRANT_PATH)}

    delta_rows: List[Dict[str, object]] = []
    for row in metrics_rows:
        key = row["key"]
        if key == "p63":
            continue

        exp03_p = to_float(row["exp03_p"])
        exp03_r = to_float(row["exp03_r"])
        exp03_f1 = to_float(row["exp03_f1"])
        exp03_tp = to_float(row["exp03_tp"])
        exp03_fp = to_float(row["exp03_fp"])
        exp03_fn = to_float(row["exp03_fn"])

        exp04_p = to_float(row["exp04_p"])
        exp04_r = to_float(row["exp04_r"])
        exp04_f1 = to_float(row["exp04_f1"])
        exp04_tp = to_float(row["exp04_tp"])
        exp04_fp = to_float(row["exp04_fp"])
        exp04_fn = to_float(row["exp04_fn"])

        delta_p = exp04_p - exp03_p
        delta_r = exp04_r - exp03_r
        delta_f1 = exp04_f1 - exp03_f1
        delta_tp = exp04_tp - exp03_tp
        delta_fp = exp04_fp - exp03_fp
        delta_fn = exp04_fn - exp03_fn

        improvement_type = "mixed"
        if delta_tp > 0 and delta_fn < 0 and delta_fp <= 0:
            improvement_type = "recall_driven_cleaner"
        elif delta_tp > 0 and delta_fn < 0:
            improvement_type = "recall_driven"
        elif delta_fp < 0 and delta_tp >= 0:
            improvement_type = "precision_driven"
        elif delta_fp > 0 and delta_fn < 0:
            improvement_type = "precision_recall_tradeoff"
        elif delta_f1 <= 0:
            improvement_type = "no_gain_or_worse"

        q = quadrant_rows.get(key, {})
        delta_rows.append({
            "key": key,
            "group": q.get("group", ""),
            "quadrant": q.get("quadrant", ""),
            "exp03_p": f"{exp03_p:.6f}",
            "exp03_r": f"{exp03_r:.6f}",
            "exp03_f1": f"{exp03_f1:.6f}",
            "exp03_tp": int(exp03_tp),
            "exp03_fp": int(exp03_fp),
            "exp03_fn": int(exp03_fn),
            "exp04_p": f"{exp04_p:.6f}",
            "exp04_r": f"{exp04_r:.6f}",
            "exp04_f1": f"{exp04_f1:.6f}",
            "exp04_tp": int(exp04_tp),
            "exp04_fp": int(exp04_fp),
            "exp04_fn": int(exp04_fn),
            "delta_p": f"{delta_p:.6f}",
            "delta_r": f"{delta_r:.6f}",
            "delta_f1": f"{delta_f1:.6f}",
            "delta_tp": int(delta_tp),
            "delta_fp": int(delta_fp),
            "delta_fn": int(delta_fn),
            "improvement_type": improvement_type,
            "gold_count": row["exp03_gold"],
        })

    fieldnames = [
        "key", "group", "quadrant",
        "exp03_p", "exp03_r", "exp03_f1", "exp03_tp", "exp03_fp", "exp03_fn",
        "exp04_p", "exp04_r", "exp04_f1", "exp04_tp", "exp04_fp", "exp04_fn",
        "delta_p", "delta_r", "delta_f1", "delta_tp", "delta_fp", "delta_fn",
        "improvement_type", "gold_count",
    ]
    write_tsv(DELTA_OUT_PATH, delta_rows, fieldnames)

    sorted_by_delta = sorted(delta_rows, key=lambda r: float(r["delta_f1"]))
    top_improve = list(reversed(sorted_by_delta[-10:]))
    top_worsen = sorted_by_delta[:10]

    recall_driven = [r for r in delta_rows if float(r["delta_r"]) > 0 and float(r["delta_p"]) <= 0]
    precision_driven = [r for r in delta_rows if float(r["delta_p"]) > 0 and float(r["delta_r"]) <= 0]
    both_up = [r for r in delta_rows if float(r["delta_p"]) > 0 and float(r["delta_r"]) > 0]
    neither_up = [r for r in delta_rows if float(r["delta_p"]) <= 0 and float(r["delta_r"]) <= 0]

    quadrant_summary: Dict[str, List[Dict[str, object]]] = {}
    for r in delta_rows:
        quadrant_summary.setdefault(r["quadrant"], []).append(r)

    report: List[str] = []
    report.append("# Report 2026-04-09: exp03 vs exp04 差分分析")
    report.append("")
    report.append("## 1. 目的")
    report.append("- `FT-Mock (exp03)` から `FT-ALL (exp04)` への改善が、項目別にどのような形で生じたかを確認する。")
    report.append("- 特に、改善が precision 主導か recall 主導か、また `tp/fp/fn` のどこで起きているかを整理する。")
    report.append("")
    report.append("## 2. 入力")
    report.append(f"- `{METRICS_PATH.relative_to(ROOT)}`")
    report.append(f"- `{QUADRANT_PATH.relative_to(ROOT)}`")
    report.append("")
    report.append("## 3. 方法")
    report.append("- 各項目について `exp04 - exp03` の `precision`, `recall`, `F1`, `tp`, `fp`, `fn` を計算した。")
    report.append("- 補助的に、既存の4象限分類 (`A/B/C/D`) と突き合わせた。")
    report.append("")
    report.append("## 4. F1 改善上位項目")
    report.append("")
    report.append("| key | quadrant | delta_f1 | delta_p | delta_r | delta_tp | delta_fp | delta_fn |")
    report.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in top_improve:
        report.append(
            f"| {r['key']} | {r['quadrant']} | {fmt(float(r['delta_f1']))} | {fmt(float(r['delta_p']))} | {fmt(float(r['delta_r']))} | {r['delta_tp']} | {r['delta_fp']} | {r['delta_fn']} |"
        )
    report.append("")
    report.append("## 5. F1 悪化項目")
    report.append("")
    report.append("| key | quadrant | delta_f1 | delta_p | delta_r | delta_tp | delta_fp | delta_fn |")
    report.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in top_worsen:
        report.append(
            f"| {r['key']} | {r['quadrant']} | {fmt(float(r['delta_f1']))} | {fmt(float(r['delta_p']))} | {fmt(float(r['delta_r']))} | {r['delta_tp']} | {r['delta_fp']} | {r['delta_fn']} |"
        )
    report.append("")
    report.append("## 6. パターン別集計")
    report.append("")
    report.append(f"- `precision` のみ上昇: {len(precision_driven)} 項目")
    report.append(f"- `recall` のみ上昇: {len(recall_driven)} 項目")
    report.append(f"- `precision` と `recall` の両方上昇: {len(both_up)} 項目")
    report.append(f"- 両方とも非上昇: {len(neither_up)} 項目")
    report.append("")
    report.append("## 7. 主要所見")
    report.append("- 改善上位には `RadiationFractions`, `SiteOfRecurrence`, `RadiationDose`, `BRAF`, `Cytology` などが含まれ、特に `RadiationDose` / `RadiationFractions` では `tp` 増加と `fn` 減少が大きく、recall 主導の改善がみられた。")
    report.append("- `ProcedureDiagnosis`, `AssessedAnatomicSite`, `SitesOfMetastasis`, `Cytology`, `Site` などでは `tp` 増加と `fn` 減少が見られ、exp04 が見落としを減らした可能性がある。")
    report.append("- `EBUS_TBNA`, `Treatment`, `TBB_TBLB` などでは F1 が悪化または非改善であり、synthetic pretraining が必ずしも全項目に一様に効くわけではなかった。")
    report.append("- 4象限との対応では、D 群 (`difficult_but_helped_by_synthetic`) に `RadiationDose`, `RadiationFractions`, `BRAF`, `SiteOfRecurrence` が入り、synthetic が実際に補助した候補項目として整合的だった。")
    report.append("- 一方で B 群 (`easy_but_synthetic_mismatched`) の `Stage`, `TNM_T/N/M`, `Oncomine` は exp04 で一定の改善を示したものの、exp03 を完全には上回らず、synthetic supervision の不整合が残っている可能性がある。")
    report.append("")
    report.append("## 8. Discussion に使える要約")
    report.append("`FT-Mock` から `FT-ALL` への改善を項目別に分解すると、改善は一様ではなく、主に一部項目での `tp` 増加と `fn` 減少、すなわち recall 改善として現れていた。とくに `RadiationDose` や `RadiationFractions` では改善が大きく、synthetic pretraining が模擬診療録だけでは不足しがちな学習信号を補った可能性が示唆された。一方で `Treatment` や `EBUS_TBNA` のように改善が乏しい、あるいは悪化する項目も存在し、two-stage 学習の利得は項目依存的であった。これらの結果は、synthetic data の効果が単なる頻度の増加ではなく、項目ごとの表現分布や annotation 整合性との適合度に依存していることを示唆する。")
    report.append("")
    report.append("## 9. 出力ファイル")
    report.append(f"- `{DELTA_OUT_PATH.relative_to(ROOT)}`")
    report.append(f"- `{REPORT_OUT_PATH.relative_to(ROOT)}`")

    REPORT_OUT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
