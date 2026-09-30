# TERMSに示された単語を、JSONL形式の症例報告データのタイトルとアブストラクトから検索し、
# 各単語についてマッチしたかどうかを0/1で示すCSVファイルを生成する
# uv run python search_flag.py

import argparse
import csv
import json
import re
from pathlib import Path


# Terms to search for (edit this list as needed)
# TERMS: list[str] = [
#     "cancer",
#     "ADR",
#     "ADE",
#     "adverse drug reaction",
#     "adverse drug effect",
#     "adverse drug event",
#     "peripheral neuropathy",
# ]

TERMS: list[str] = [
    "lung cancer",
    "year",
    "years",
    "old",
]


def compile_term_patterns(terms: list[str]) -> dict[str, re.Pattern]:
    """
    compile rule:
      - Use \\b word-boundary matching
      - case-insensitive for ALL terms (including ADR/ADE)
      - regex pattern is safely escaped (so the term is treated as a literal string)
    """
    patterns: dict[str, re.Pattern] = {}
    for t in terms:
        t_stripped = t.strip()
        if not t_stripped:
            continue

        escaped = re.escape(t_stripped)
        pat_body = rf"\b{escaped}\b"
        patterns[t_stripped] = re.compile(pat_body, flags=re.IGNORECASE)

    return patterns


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def build_flags(
    input_jsonl: str,
    terms: list[str],
    output_path: str,
) -> dict:
    patterns = compile_term_patterns(terms)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", *terms])

        for obj in iter_jsonl(input_jsonl):
            _id = str(obj.get("id", "")).strip()
            title = obj.get("title", "") or ""
            abstract = obj.get("abstract", "") or ""

            row = [_id]
            for t in terms:
                key = t.strip()
                pat = patterns.get(key)

                hit = 1 if (pat is not None and (pat.search(title) or pat.search(abstract))) else 0
                row.append(hit)

            writer.writerow(row)
            n += 1

    return {"rows": n, "output": str(out_path)}


def _parse_terms_arg(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "From a JSONL (id/title/abstract), create a CSV of id + 0/1 flags for terms.\n"
            "Matching rule: \\b boundary + case-insensitive for all terms."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default="../data/raw/case_reports_all.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="../data/processed/case_reports_all_flags.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--terms",
        default=None,
        help="Optional comma-separated term list to override TERMS in code.",
    )
    args = parser.parse_args()

    terms = _parse_terms_arg(args.terms) if args.terms else TERMS
    if not terms:
        raise ValueError("No terms provided. Set TERMS in code or pass --terms.")

    report = build_flags(args.input_jsonl, terms, args.output)

    print(f"terms\t{len(terms)}")
    print(f"rows\t{report['rows']}")
    print(f"output\t{report['output']}")


if __name__ == "__main__":
    main()
