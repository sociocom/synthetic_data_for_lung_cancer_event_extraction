#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from src.analyze_annotation_value_distribution import parse_annotation_field, split_tokens
    from src.list_value_schema import load_allowed_values
except ModuleNotFoundError:
    from analyze_annotation_value_distribution import parse_annotation_field, split_tokens
    from list_value_schema import load_allowed_values


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Row must be JSON object at {path}:{line_no}")
            yield obj


def analyze(
    annotation_jsonl: Path,
    allowed_values: dict[str, list[str]],
    *,
    sample_limit: int,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    token_counter: dict[str, Counter[str]] = {k: Counter() for k in allowed_values}
    raw_counter: dict[str, Counter[str]] = {k: Counter() for k in allowed_values}
    sample_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        k: defaultdict(list) for k in allowed_values
    }

    rows = 0
    events = 0
    parse_errors = 0

    for obj in iter_jsonl(annotation_jsonl):
        rows += 1
        row_id = str(obj.get("id", ""))
        try:
            evs = parse_annotation_field(obj.get("annotation"))
        except Exception:
            parse_errors += 1
            continue

        events += len(evs)
        for ev_idx, ev in enumerate(evs):
            if not isinstance(ev, dict):
                continue
            for key, allowed in allowed_values.items():
                if key not in ev:
                    continue
                raw_value = ev[key]
                raw_text = str(raw_value)
                raw_counter[key][raw_text] += 1

                for token in split_tokens(raw_value):
                    token_counter[key][token] += 1
                    if token in allowed:
                        continue
                    bucket = sample_rows[key][token]
                    if len(bucket) >= sample_limit:
                        continue
                    bucket.append(
                        {
                            "id": row_id,
                            "event_index": ev_idx,
                            "raw_value": raw_text,
                            "timestamp": str(ev.get("Timestamp", "")),
                        }
                    )

    key_reports: dict[str, Any] = {}
    for key, allowed in allowed_values.items():
        allowed_set = set(allowed)
        tok_c = token_counter[key]
        total = sum(tok_c.values())
        known = sum(v for t, v in tok_c.items() if t in allowed_set)
        unknown = total - known
        unknown_counter = Counter({t: v for t, v in tok_c.items() if t not in allowed_set})
        observed_known = [v for v in allowed if v in tok_c]
        missing_allowed = [v for v in allowed if v not in tok_c]

        unknown_values: list[dict[str, Any]] = []
        for token, count in unknown_counter.most_common():
            unknown_values.append(
                {
                    "token": token,
                    "count": count,
                }
            )

        key_reports[key] = {
            "allowed_count": len(allowed),
            "allowed_values": allowed,
            "token_total_count": total,
            "token_unique_count": len(tok_c),
            "known_token_count": known,
            "unknown_token_count": unknown,
            "unknown_ratio": (unknown / total) if total else 0.0,
            "observed_known_value_count": len(observed_known),
            "observed_known_values": observed_known,
            "missing_allowed_value_count": len(missing_allowed),
            "missing_allowed_values": missing_allowed,
            "unknown_value_count": len(unknown_values),
            "unknown_values": unknown_values,
            "top_unknown_tokens": unknown_counter.most_common(20),
            "top_raw_values": raw_counter[key].most_common(20),
        }

    report = {
        "input_annotation_jsonl": str(annotation_jsonl),
        "rows": rows,
        "events": events,
        "parse_errors": parse_errors,
        "keys": key_reports,
    }
    return report, sample_rows


def write_tsv(
    path: Path,
    report: dict[str, Any],
    sample_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    "key",
                    "allowed_count",
                    "token_total_count",
                    "known_token_count",
                    "unknown_token_count",
                    "unknown_ratio",
                    "observed_known_value_count",
                    "missing_allowed_value_count",
                    "unknown_token",
                    "unknown_count",
                    "sample_ids",
                    "sample_raw_values",
                ]
            )
            + "\n"
        )

        for key, detail in sorted(report["keys"].items()):
            base = [
                key,
                str(detail["allowed_count"]),
                str(detail["token_total_count"]),
                str(detail["known_token_count"]),
                str(detail["unknown_token_count"]),
                f'{detail["unknown_ratio"]:.6f}',
                str(detail["observed_known_value_count"]),
                str(detail["missing_allowed_value_count"]),
            ]
            unknown_values = detail.get("unknown_values", [])
            if not unknown_values:
                f.write("\t".join(base + ["", "", "", ""]) + "\n")
                continue

            for row in unknown_values:
                token = str(row.get("token", ""))
                samples = sample_rows.get(key, {}).get(token, [])
                sample_ids = ",".join(str(s.get("id", "")) for s in samples)
                sample_raw_values = " | ".join(str(s.get("raw_value", "")) for s in samples)
                f.write(
                    "\t".join(
                        base
                        + [
                            token,
                            str(row.get("count", 0)),
                            sample_ids,
                            sample_raw_values,
                        ]
                    )
                    + "\n"
                )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze out-of-candidate values for all prompt keys with allowed-value lists."
    )
    p.add_argument("--annotation_jsonl", type=Path, required=True, help="Annotation JSONL path")
    p.add_argument(
        "--schema_file",
        type=Path,
        default=None,
        help="List-value schema JSON path. Preferred over --prompt_file.",
    )
    p.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="Legacy prompt TOML path. Used only when --schema_file is omitted.",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path("results/analysis/out_of_candidate_values.json"),
        help="Output JSON path",
    )
    p.add_argument(
        "--out_tsv",
        type=Path,
        default=Path("results/analysis/out_of_candidate_values.tsv"),
        help="Output TSV path",
    )
    p.add_argument(
        "--sample_limit",
        type=int,
        default=5,
        help="Max sample rows to keep per unknown token",
    )
    args = p.parse_args()

    if not args.annotation_jsonl.exists():
        raise FileNotFoundError(f"annotation_jsonl not found: {args.annotation_jsonl}")
    if args.sample_limit <= 0:
        raise ValueError("--sample_limit must be positive")

    allowed_map, schema_source = load_allowed_values(schema_file=args.schema_file, prompt_file=args.prompt_file)
    report, sample_rows = analyze(args.annotation_jsonl, allowed_map, sample_limit=args.sample_limit)
    report["allowed_values_source"] = schema_source
    report["sample_limit"] = args.sample_limit

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.out_tsv, report, sample_rows)

    key_summaries = []
    for key, detail in report["keys"].items():
        if detail["token_total_count"] == 0:
            continue
        key_summaries.append(
            (
                detail["unknown_ratio"],
                detail["unknown_token_count"],
                detail["token_total_count"],
                key,
            )
        )
    key_summaries.sort(reverse=True)

    print(f"allowed_values_source={schema_source}")
    print(f"annotation_jsonl={args.annotation_jsonl}")
    print(f"rows={report['rows']}, events={report['events']}, parse_errors={report['parse_errors']}")
    print(f"saved_json={args.out_json}")
    print(f"saved_tsv={args.out_tsv}")
    print("")
    print("[top_unknown_ratio]")
    for ratio, unknown, total, key in key_summaries[:10]:
        print(f"{key}: unknown={unknown}/{total} unknown_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
