#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.list_value_schema import load_allowed_values
except ModuleNotFoundError:
    from list_value_schema import load_allowed_values


LIST_SPLIT_RE = re.compile(r"[,、，]+")


def _is_json_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def _to_pretty_json_with_compact_scalar_lists(obj: Any, *, level: int = 0, indent: int = 2) -> str:
    sp = " " * level
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines: list[str] = []
        items = list(obj.items())
        for idx, (k, v) in enumerate(items):
            key = json.dumps(str(k), ensure_ascii=False)
            rendered = _to_pretty_json_with_compact_scalar_lists(v, level=level + indent, indent=indent)
            comma = "," if idx < len(items) - 1 else ""
            lines.append(f'{" " * (level + indent)}{key}: {rendered}{comma}')
        return "{\n" + "\n".join(lines) + f"\n{sp}" + "}"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(_is_json_scalar(x) for x in obj):
            return json.dumps(obj, ensure_ascii=False)
        lines: list[str] = []
        for idx, item in enumerate(obj):
            rendered = _to_pretty_json_with_compact_scalar_lists(item, level=level + indent, indent=indent)
            comma = "," if idx < len(obj) - 1 else ""
            lines.append(f'{" " * (level + indent)}{rendered}{comma}')
        return "[\n" + "\n".join(lines) + f"\n{sp}" + "]"

    return json.dumps(obj, ensure_ascii=False)


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


def parse_annotation_field(value: Any) -> list[dict[str, Any]]:
    parsed: Any = value
    if isinstance(value, str):
        parsed = json.loads(value)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict)]


def split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(split_tokens(v))
        return out
    s = str(value).strip()
    if not s:
        return []
    return [tok.strip() for tok in LIST_SPLIT_RE.split(s) if tok.strip()]


def analyze(
    annotation_jsonl: Path,
    target_keys: list[str],
    allowed_values: dict[str, list[str]],
) -> dict[str, Any]:
    raw_counter: dict[str, Counter[str]] = {k: Counter() for k in target_keys}
    token_counter: dict[str, Counter[str]] = {k: Counter() for k in target_keys}

    rows = 0
    events = 0
    parse_errors = 0

    for obj in iter_jsonl(annotation_jsonl):
        rows += 1
        ann = obj.get("annotation")
        try:
            evs = parse_annotation_field(ann)
        except Exception:
            parse_errors += 1
            continue
        events += len(evs)
        for ev in evs:
            for k in target_keys:
                if k not in ev:
                    continue
                raw_value = ev[k]
                raw_counter[k][str(raw_value)] += 1
                for tok in split_tokens(raw_value):
                    token_counter[k][tok] += 1

    report: dict[str, Any] = {
        "input": str(annotation_jsonl),
        "rows": rows,
        "events": events,
        "parse_errors": parse_errors,
        "keys": {},
    }

    for k in target_keys:
        allowed = allowed_values.get(k, [])
        allowed_set = set(allowed)
        tok_c = token_counter[k]
        total = sum(tok_c.values())
        known = sum(v for t, v in tok_c.items() if t in allowed_set)
        unknown = total - known
        unknown_ratio = (unknown / total) if total else 0.0
        unknown_counter = Counter({t: v for t, v in tok_c.items() if t not in allowed_set})
        known_counter = Counter({t: v for t, v in tok_c.items() if t in allowed_set})
        observed_known_values = sorted(known_counter.keys())
        missing_allowed_values = [v for v in allowed if v not in known_counter]
        coverage_ratio = (len(observed_known_values) / len(allowed)) if allowed else 0.0

        report["keys"][k] = {
            "allowed_count": len(allowed),
            "allowed_values": allowed,
            "token_total_count": total,
            "token_unique_count": len(tok_c),
            "known_token_count": known,
            "unknown_token_count": unknown,
            "unknown_ratio": unknown_ratio,
            "observed_known_value_count": len(observed_known_values),
            "observed_known_values": observed_known_values,
            "missing_allowed_value_count": len(missing_allowed_values),
            "missing_allowed_values": missing_allowed_values,
            "coverage_ratio": coverage_ratio,
            "top_tokens": tok_c.most_common(30),
            "top_known_tokens": known_counter.most_common(30),
            "top_unknown_tokens": unknown_counter.most_common(50),
            "top_raw_values": raw_counter[k].most_common(30),
        }

    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze annotation value distributions and schema-list compliance for selected keys."
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
        "--keys",
        type=str,
        default="ProcedureDiagnosis,Treatment,Site",
        help="Comma-separated target keys",
    )
    p.add_argument("--out_json", type=Path, default=Path("results/analysis/annotation_value_distribution.json"))
    args = p.parse_args()

    keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    if not keys:
        raise ValueError("No valid keys were provided.")
    if not args.annotation_jsonl.exists():
        raise FileNotFoundError(f"annotation_jsonl not found: {args.annotation_jsonl}")

    allowed_map, schema_source = load_allowed_values(schema_file=args.schema_file, prompt_file=args.prompt_file)
    report = analyze(args.annotation_jsonl, keys, allowed_map)
    report["allowed_values_source"] = schema_source

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = _to_pretty_json_with_compact_scalar_lists(report, level=0, indent=2)
    args.out_json.write_text(rendered + "\n", encoding="utf-8")

    print(f"allowed_values_source={schema_source}")
    print(f"annotation_jsonl={args.annotation_jsonl}")
    print(f"rows={report['rows']}, events={report['events']}, parse_errors={report['parse_errors']}")
    print(f"saved={args.out_json}")
    print("")
    for k in keys:
        d = report["keys"].get(k, {})
        print(f"[{k}]")
        print(f"  allowed_count={d.get('allowed_count', 0)}")
        print(f"  token_total={d.get('token_total_count', 0)}")
        print(f"  token_unique={d.get('token_unique_count', 0)}")
        print(f"  unknown_token_count={d.get('unknown_token_count', 0)}")
        print(f"  unknown_ratio={d.get('unknown_ratio', 0.0):.4f}")
        print(
            f"  value_coverage={d.get('observed_known_value_count', 0)}/{d.get('allowed_count', 0)}"
            f" ({d.get('coverage_ratio', 0.0):.4f})"
        )
        print(f"  top_unknown_tokens={d.get('top_unknown_tokens', [])[:10]}")
        print("")


if __name__ == "__main__":
    main()
