# ../data/processed/case_reports_all_flags.csv (0/1 flags) と
# ../data/raw/case_reports_all.jsonl (本文) を id で結合し、条件に一致するレコードだけを
# JSONL として書き出す。
#
# 例:
#   uv run python make_filtered_jsonl.py \
#     --flags-csv ../data/processed/case_reports_all_flags.csv \
#     --raw-jsonl  ../data/raw/case_reports_all.jsonl \
#     --expr "lung_cancer AND old AND (year OR years)" \
#     --out ../data/processed/case_reports_peripheral_neuropathy.jsonl

# AND/OR/NOT や括弧も使える（count_bin.py と同じ式）。
#   --expr "cancer AND (peripheral_neuropathy OR ADE)"

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _norm_col_name(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def _col_map(df: pd.DataFrame) -> dict[str, str]:
    return {_norm_col_name(c): c for c in df.columns}


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    """Resolve a user-specified column name to an actual DF column name (forgiving)."""
    target_n = _norm_col_name(name)
    cmap = _col_map(df)

    if target_n in cmap:
        return cmap[target_n]

    # fallback: substring match
    for n, actual in cmap.items():
        if target_n in n:
            return actual

    raise KeyError(
        f"Column '{name}' not found. Available columns include: {list(df.columns)[:30]}"
    )


class _Token:
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value


def _tokenize(expr: str) -> list[_Token]:
    s = expr.strip()
    if not s:
        raise ValueError("Empty expression")

    tokens: list[_Token] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(_Token(ch, ch))
            i += 1
            continue

        j = i
        while j < len(s) and (not s[j].isspace()) and s[j] not in "()":
            j += 1
        word = s[i:j]
        w = word.lower()
        if w in {"and", "or", "not"}:
            tokens.append(_Token(w.upper(), word))
        else:
            tokens.append(_Token("IDENT", word))
        i = j

    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]):
        self.toks = tokens
        self.pos = 0

    def _peek(self) -> _Token | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _eat(self, kind: str) -> _Token:
        t = self._peek()
        if t is None or t.kind != kind:
            got = None if t is None else t.kind
            raise ValueError(f"Parse error: expected {kind}, got {got}")
        self.pos += 1
        return t

    def parse(self):
        node = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"Parse error: unexpected token '{self._peek().value}'")
        return node

    def _parse_or(self):
        node = self._parse_and()
        while True:
            t = self._peek()
            if t is None or t.kind != "OR":
                break
            self._eat("OR")
            rhs = self._parse_and()
            node = ("OR", node, rhs)
        return node

    def _parse_and(self):
        node = self._parse_unary()
        while True:
            t = self._peek()
            if t is None or t.kind != "AND":
                break
            self._eat("AND")
            rhs = self._parse_unary()
            node = ("AND", node, rhs)
        return node

    def _parse_unary(self):
        t = self._peek()
        if t is not None and t.kind == "NOT":
            self._eat("NOT")
            inner = self._parse_unary()
            return ("NOT", inner)
        return self._parse_primary()

    def _parse_primary(self):
        t = self._peek()
        if t is None:
            raise ValueError("Parse error: unexpected end")
        if t.kind == "(":
            self._eat("(")
            node = self._parse_or()
            self._eat(")")
            return node
        if t.kind == "IDENT":
            ident = self._eat("IDENT").value
            return ("IDENT", ident)
        raise ValueError(f"Parse error: unexpected token '{t.value}'")


def _eval_ast(df: pd.DataFrame, ast) -> pd.Series:
    op = ast[0]
    if op == "IDENT":
        col_name = _resolve_col(df, ast[1])
        s = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(int)
        return s == 1
    if op == "NOT":
        return ~_eval_ast(df, ast[1])
    if op == "AND":
        return _eval_ast(df, ast[1]) & _eval_ast(df, ast[2])
    if op == "OR":
        return _eval_ast(df, ast[1]) | _eval_ast(df, ast[2])
    raise ValueError(f"Unknown AST node: {op}")


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                continue
            yield obj


def _build_mask(flags_df: pd.DataFrame, expr: str) -> pd.Series:
    tokens = _tokenize(expr)
    ast = _Parser(tokens).parse()
    return _eval_ast(flags_df, ast)


def build_filtered_jsonl(
    flags_csv: str,
    raw_jsonl: str,
    out_jsonl: str,
    expr: str,
    *,
    encoding: str | None = None,
    attach_flags: bool = False,
    strict_id_check: bool = False,
) -> dict[str, Any]:
    flags = pd.read_csv(flags_csv, encoding=encoding)
    if "id" not in flags.columns:
        raise KeyError("flags CSV must include 'id' column")

    # Make id comparable to raw-jsonl's `id` (string based)
    flags = flags.copy()
    flags["id"] = flags["id"].astype(str).str.strip()

    mask = _build_mask(flags, expr)
    sel = flags.loc[mask].copy()

    selected_ids = set(sel["id"].tolist())

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build fast lookup for flags by id (optional)
    flags_by_id: dict[str, dict[str, Any]] = {}
    if attach_flags:
        # keep only flag columns (exclude id)
        flag_cols = [c for c in sel.columns if c != "id"]
        for _, row in sel.iterrows():
            _id = str(row["id"])
            flags_by_id[_id] = {c: int(pd.to_numeric(row[c], errors="coerce") or 0) for c in flag_cols}

    written = 0
    seen_ids = set()
    with out_path.open("w", encoding="utf-8") as out_f:
        for obj in iter_jsonl(raw_jsonl):
            _id = str(obj.get("id", "")).strip()
            if not _id:
                continue
            if _id in selected_ids:
                if attach_flags:
                    obj = dict(obj)
                    obj["flags"] = flags_by_id.get(_id, {})
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                seen_ids.add(_id)

    if strict_id_check:
        missing = selected_ids - seen_ids
        if missing:
            raise RuntimeError(
                f"{len(missing)} ids were selected by flags but not found in raw jsonl. Example: {sorted(list(missing))[:5]}"
            )

    return {
        "expr": expr,
        "selected_rows": int(mask.sum()),
        "written": written,
        "out": str(out_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Join flags CSV with raw case-reports JSONL by id, filter by expression, and write a new JSONL.\n"
            "Expression grammar supports AND/OR/NOT and parentheses. IDENT tokens refer to CSV columns.\n"
            "(Use underscores instead of spaces, e.g. peripheral_neuropathy)"
        )
    )
    p.add_argument(
        "--flags-csv",
        type=str,
        default="../data/processed/case_reports_all_flags.csv",
        help="Input flags CSV path",
    )
    p.add_argument(
        "--raw-jsonl",
        type=str,
        default="../data/raw/case_reports_all.jsonl",
        help="Input raw JSONL path",
    )
    p.add_argument(
        "--out",
        type=str,
        default="../data/processed/case_reports_peripheral_neuropathy.jsonl",
        help="Output JSONL path",
    )
    p.add_argument(
        "--expr",
        type=str,
        default="peripheral_neuropathy",
        help="Boolean expression over 0/1 columns, e.g. 'cancer AND peripheral_neuropathy'",
    )
    p.add_argument("--encoding", type=str, default=None, help="Flags CSV encoding (optional)")
    p.add_argument(
        "--attach-flags",
        action="store_true",
        help="Attach selected row's flag columns into output JSON under key 'flags'",
    )
    p.add_argument(
        "--strict-id-check",
        action="store_true",
        help="Fail if some selected ids are not found in raw jsonl",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    report = build_filtered_jsonl(
        flags_csv=args.flags_csv,
        raw_jsonl=args.raw_jsonl,
        out_jsonl=args.out,
        expr=args.expr,
        encoding=args.encoding,
        attach_flags=args.attach_flags,
        strict_id_check=args.strict_id_check,
    )

    print(f"expr\t{report['expr']}")
    print(f"selected_rows\t{report['selected_rows']}")
    print(f"written\t{report['written']}")
    print(f"out\t{report['out']}")


if __name__ == "__main__":
    main()
