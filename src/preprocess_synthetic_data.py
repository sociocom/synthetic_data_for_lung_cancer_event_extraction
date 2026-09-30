#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from src.list_value_schema import DEFAULT_LIST_VALUE_SCHEMA_PATH, load_allowed_values_from_schema
    from src.normalize import extract_json_from_text_lenient
except ModuleNotFoundError:
    from list_value_schema import DEFAULT_LIST_VALUE_SCHEMA_PATH, load_allowed_values_from_schema
    from normalize import extract_json_from_text_lenient


LIST_TOKEN_SPLIT_RE = re.compile(r"[,、，]+")


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield obj


def _parse_annotation(value: Any) -> List[Dict[str, Any]]:
    parsed: Any = value
    if isinstance(value, str):
        parsed = extract_json_from_text_lenient(value)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"annotation must be list/dict/string JSON, got {type(value)}")

    out: List[Dict[str, Any]] = []
    for ev in parsed:
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _load_progress(progress_jsonl: str, id_key: str, progress_key: str) -> List[Tuple[str, Dict[str, Any]]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for obj in iter_jsonl(progress_jsonl):
        rid = obj.get(id_key)
        txt = obj.get(progress_key)
        if rid is None or txt is None:
            continue
        rid_s = str(rid)
        if not str(txt).strip():
            continue
        rows.append((rid_s, obj))
    return rows


def _load_annotations(annotations_jsonl: str, id_key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(annotations_jsonl):
        rid = obj.get(id_key)
        if rid is None:
            continue
        out[str(rid)] = obj
    return out


def _load_allowed_keys_from_prompt(prompt_file: str) -> set[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        text = f.read()
    keys: set[str] = set()
    pat = re.compile(r"^\s*-\s*JSONキー:\s*(\S+)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            keys.add(m.group(1))
    if not keys:
        raise ValueError(f"No '- JSONキー: ...' entries found in prompt file: {prompt_file}")
    return keys


def _drop_unknown_keys(
    events: List[Dict[str, Any]], allowed_keys: set[str]
) -> Tuple[List[Dict[str, Any]], int, int]:
    out: List[Dict[str, Any]] = []
    dropped_keys = 0
    dropped_events = 0
    for ev in events:
        cleaned: Dict[str, Any] = {}
        for k, v in ev.items():
            if k in allowed_keys:
                cleaned[k] = v
            else:
                dropped_keys += 1
        if cleaned:
            out.append(cleaned)
        else:
            dropped_events += 1
    return out, dropped_keys, dropped_events


def _dedupe_events(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    for ev in events:
        sig = json.dumps(ev, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        out.append(ev)
    return out, dropped


def _infer_site_from_progress_text(progress_text: str, side: str) -> str | None:
    if side not in {"右", "左"}:
        return None

    candidates: set[str] = set()

    if side == "右":
        direct_patterns = [
            (r"(?:右肺上葉|右上葉)", "右肺上葉"),
            (r"(?:右肺中葉|右中葉)", "右肺中葉"),
            (r"(?:右肺下葉|右下葉)", "右肺下葉"),
            (r"(?:右肺門部|右肺門|#10R)", "右肺門部"),
        ]
        seg_re = re.compile(r"右S\s*(1\+2|[1-9]|10)")
    else:
        direct_patterns = [
            (r"(?:左肺上葉|左上葉)", "左肺上葉"),
            (r"(?:左肺下葉|左下葉)", "左肺下葉"),
            (r"(?:左肺門部|左肺門|#10L)", "左肺門部"),
        ]
        seg_re = re.compile(r"左S\s*(1\+2|[1-9]|10)")

    for pat, mapped in direct_patterns:
        if re.search(pat, progress_text):
            candidates.add(mapped)

    for m in seg_re.finditer(progress_text):
        seg = m.group(1).replace(" ", "")
        if side == "右":
            if seg in {"1", "2", "3"}:
                candidates.add("右肺上葉")
            elif seg in {"4", "5"}:
                candidates.add("右肺中葉")
            elif seg in {"6", "7", "8", "9", "10"}:
                candidates.add("右肺下葉")
        else:
            if seg in {"1", "1+2", "2", "3", "4", "5"}:
                candidates.add("左肺上葉")
            elif seg in {"6", "7", "8", "9", "10"}:
                candidates.add("左肺下葉")

    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _normalize_side_only_site(events: List[Dict[str, Any]], progress_text: str) -> Tuple[List[Dict[str, Any]], int]:
    right_repl = _infer_site_from_progress_text(progress_text, "右") or "肺"
    left_repl = _infer_site_from_progress_text(progress_text, "左") or "肺"

    changed = 0
    out: List[Dict[str, Any]] = []
    for ev in events:
        if "Site" not in ev or not isinstance(ev.get("Site"), str):
            out.append(ev)
            continue

        raw = str(ev.get("Site", ""))
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens:
            out.append(ev)
            continue

        new_tokens: List[str] = []
        touched = False
        for t in tokens:
            if t == "右肺":
                new_tokens.append(right_repl)
                touched = True
            elif t == "左肺":
                new_tokens.append(left_repl)
                touched = True
            else:
                new_tokens.append(t)

        if touched:
            seen: set[str] = set()
            deduped: List[str] = []
            for t in new_tokens:
                if t in seen:
                    continue
                seen.add(t)
                deduped.append(t)
            new_ev = dict(ev)
            new_ev["Site"] = ", ".join(deduped)
            out.append(new_ev)
            changed += 1
        else:
            out.append(ev)

    return out, changed


def _split_value_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for elem in value:
            out.extend(_split_value_tokens(elem))
        return out
    s = str(value).strip()
    if not s:
        return []
    return [tok.strip() for tok in LIST_TOKEN_SPLIT_RE.split(s) if tok.strip()]


def _drop_out_of_candidate_values(
    events: List[Dict[str, Any]],
    allowed_values: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], int, int, int, Counter[str]]:
    out: List[Dict[str, Any]] = []
    dropped_tokens = 0
    dropped_keys = 0
    dropped_events = 0
    per_key_dropped_tokens: Counter[str] = Counter()

    for ev in events:
        cleaned: Dict[str, Any] = {}
        for key, value in ev.items():
            allowed = allowed_values.get(key)
            if not allowed:
                cleaned[key] = value
                continue

            tokens = _split_value_tokens(value)
            if not tokens:
                dropped_keys += 1
                continue

            allowed_set = set(allowed)
            kept_tokens: List[str] = []
            seen: set[str] = set()
            key_dropped = 0
            for tok in tokens:
                if tok in allowed_set:
                    if tok not in seen:
                        kept_tokens.append(tok)
                        seen.add(tok)
                else:
                    key_dropped += 1

            if key_dropped > 0:
                dropped_tokens += key_dropped
                per_key_dropped_tokens[key] += key_dropped

            if kept_tokens:
                cleaned[key] = ", ".join(kept_tokens)
            else:
                dropped_keys += 1

        if cleaned:
            out.append(cleaned)
        else:
            dropped_events += 1

    return out, dropped_tokens, dropped_keys, dropped_events, per_key_dropped_tokens


def _drop_timestamp_only_events(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    dropped = 0
    for ev in events:
        if isinstance(ev, dict) and set(ev.keys()) == {"Timestamp"}:
            dropped += 1
            continue
        out.append(ev)
    return out, dropped


def _save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Clean synthetic JSONL: normalize annotation JSON, drop invalid keys, normalize Site, and optionally drop out-of-candidate list values.")
    p.add_argument("--progress_jsonl", required=True, type=str)
    p.add_argument("--annotations_jsonl", required=True, type=str)
    p.add_argument("--out_progress_jsonl", required=True, type=str)
    p.add_argument("--out_annotations_jsonl", required=True, type=str)
    p.add_argument("--id_key", default="id", type=str)
    p.add_argument("--progress_key", default="progress_note", type=str)
    p.add_argument("--annotation_key", default="annotation", type=str)
    p.add_argument(
        "--dedupe_events",
        action="store_true",
        help="Remove duplicated events within each annotation list (exact dict match).",
    )
    p.add_argument(
        "--drop_unknown_keys",
        action="store_true",
        help="Drop keys not listed in --allowed_keys_prompt_file from each event.",
    )
    p.add_argument(
        "--allowed_keys_prompt_file",
        default="",
        type=str,
        help="Prompt TOML path used to read allowed keys ('- JSONキー: ...'). Required with --drop_unknown_keys.",
    )
    p.add_argument(
        "--normalize_side_only_site",
        action="store_true",
        help="Normalize Site values 右肺/左肺 using note-level clues; fallback to 肺 when not inferable.",
    )
    p.add_argument(
        "--drop_out_of_candidate_values",
        action="store_true",
        help="Drop list-value tokens that are not present in the list-value schema.",
    )
    p.add_argument(
        "--list_value_schema_file",
        default=str(DEFAULT_LIST_VALUE_SCHEMA_PATH),
        type=str,
        help="List-value schema JSON path used when --drop_out_of_candidate_values is set.",
    )
    p.add_argument(
        "--drop_timestamp_only_events",
        action="store_true",
        help="Drop events that only contain Timestamp after other cleaning steps.",
    )
    args = p.parse_args()

    allowed_keys: set[str] = set()
    if args.drop_unknown_keys:
        if not args.allowed_keys_prompt_file:
            raise ValueError("--allowed_keys_prompt_file is required when --drop_unknown_keys is set.")
        allowed_keys = _load_allowed_keys_from_prompt(args.allowed_keys_prompt_file)

    allowed_values: Dict[str, List[str]] = {}
    if args.drop_out_of_candidate_values:
        if not args.list_value_schema_file:
            raise ValueError("--list_value_schema_file is required when --drop_out_of_candidate_values is set.")
        allowed_values = load_allowed_values_from_schema(Path(args.list_value_schema_file))

    progress_rows = _load_progress(args.progress_jsonl, args.id_key, args.progress_key)
    ann_map = _load_annotations(args.annotations_jsonl, args.id_key)

    clean_progress: List[Dict[str, Any]] = []
    clean_annotations: List[Dict[str, Any]] = []

    n_progress = len(progress_rows)
    n_ann = len(ann_map)
    n_missing_ann = 0
    n_parse_error = 0
    n_empty = 0
    n_unknown_keys_dropped = 0
    n_events_dropped_after_unknown = 0
    n_rows_touched_unknown = 0
    n_rows_touched_out_of_candidate = 0
    n_out_of_candidate_tokens_dropped = 0
    n_keys_dropped_after_out_of_candidate = 0
    n_events_dropped_after_out_of_candidate = 0
    per_key_out_of_candidate_dropped_tokens: Counter[str] = Counter()
    n_events_dedup_dropped = 0
    n_rows_touched_dedupe = 0
    n_rows_touched_side_site = 0
    n_site_values_normalized = 0
    n_rows_touched_timestamp_only_drop = 0
    n_timestamp_only_events_dropped = 0

    for rid, p_row in progress_rows:
        a_row = ann_map.get(rid)
        if a_row is None:
            n_missing_ann += 1
            continue

        raw_ann = a_row.get(args.annotation_key)
        if raw_ann is None:
            n_parse_error += 1
            continue

        try:
            events = _parse_annotation(raw_ann)
        except Exception:
            n_parse_error += 1
            continue

        if args.drop_unknown_keys:
            events, dropped_keys, dropped_events = _drop_unknown_keys(events, allowed_keys)
            n_unknown_keys_dropped += dropped_keys
            n_events_dropped_after_unknown += dropped_events
            if dropped_keys > 0 or dropped_events > 0:
                n_rows_touched_unknown += 1

        if args.normalize_side_only_site:
            progress_text = str(p_row.get(args.progress_key, ""))
            events, site_changed = _normalize_side_only_site(events, progress_text)
            n_site_values_normalized += site_changed
            if site_changed > 0:
                n_rows_touched_side_site += 1

        if args.drop_out_of_candidate_values:
            events, dropped_tokens, dropped_keys, dropped_events, per_key_dropped = _drop_out_of_candidate_values(events, allowed_values)
            n_out_of_candidate_tokens_dropped += dropped_tokens
            n_keys_dropped_after_out_of_candidate += dropped_keys
            n_events_dropped_after_out_of_candidate += dropped_events
            per_key_out_of_candidate_dropped_tokens.update(per_key_dropped)
            if dropped_tokens > 0 or dropped_keys > 0 or dropped_events > 0:
                n_rows_touched_out_of_candidate += 1

        if args.drop_timestamp_only_events:
            events, dropped = _drop_timestamp_only_events(events)
            n_timestamp_only_events_dropped += dropped
            if dropped > 0:
                n_rows_touched_timestamp_only_drop += 1

        if args.dedupe_events:
            events, dropped = _dedupe_events(events)
            n_events_dedup_dropped += dropped
            if dropped > 0:
                n_rows_touched_dedupe += 1

        if len(events) == 0:
            n_empty += 1
            continue

        a_clean = dict(a_row)
        a_clean[args.annotation_key] = json.dumps(events, ensure_ascii=False)

        clean_progress.append(p_row)
        clean_annotations.append(a_clean)

    _save_jsonl(args.out_progress_jsonl, clean_progress)
    _save_jsonl(args.out_annotations_jsonl, clean_annotations)

    print(f"progress_in={n_progress}")
    print(f"annotation_in={n_ann}")
    print(f"clean_rows={len(clean_progress)}")
    print(f"dropped_missing_annotation={n_missing_ann}")
    print(f"dropped_parse_error={n_parse_error}")
    print(f"dropped_empty_annotation={n_empty}")
    print(f"rows_touched_unknown_key_drop={n_rows_touched_unknown}")
    print(f"dropped_unknown_keys={n_unknown_keys_dropped}")
    print(f"dropped_events_after_unknown_key_drop={n_events_dropped_after_unknown}")
    print(f"rows_touched_side_site_normalization={n_rows_touched_side_site}")
    print(f"normalized_side_site_values={n_site_values_normalized}")
    print(f"rows_touched_out_of_candidate_value_drop={n_rows_touched_out_of_candidate}")
    print(f"dropped_out_of_candidate_tokens={n_out_of_candidate_tokens_dropped}")
    print(f"dropped_keys_after_out_of_candidate_value_drop={n_keys_dropped_after_out_of_candidate}")
    print(f"dropped_events_after_out_of_candidate_value_drop={n_events_dropped_after_out_of_candidate}")
    print(f"per_key_dropped_out_of_candidate_tokens={json.dumps(dict(sorted(per_key_out_of_candidate_dropped_tokens.items())), ensure_ascii=False)}")
    print(f"rows_touched_timestamp_only_drop={n_rows_touched_timestamp_only_drop}")
    print(f"dropped_timestamp_only_events={n_timestamp_only_events_dropped}")
    print(f"rows_touched_dedupe={n_rows_touched_dedupe}")
    print(f"dropped_duplicated_events={n_events_dedup_dropped}")
    print(f"out_progress={args.out_progress_jsonl}")
    print(f"out_annotations={args.out_annotations_jsonl}")


if __name__ == "__main__":
    main()
