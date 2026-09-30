# ../data/processed/case_reports_all_flags.csvを読み込んで、それぞれの単語でマッチしたものがいくつあるかを確認することができる
# ANDやORを使うことで、組み合わせの条件でも検索可能
# uv run python count_bin.py --expr "cancer AND peripheral_neuropathy"

import argparse
import pandas as pd


def _find_column_index(df: pd.DataFrame, target: str) -> int:
    """Find column index for `target` in a forgiving way (case/space/underscore-insensitive)."""
    norm = lambda s: " ".join(str(s).strip().lower().replace("_", " ").split())
    target_n = norm(target)

    for i, c in enumerate(df.columns):
        if norm(c) == target_n:
            return i

    for i, c in enumerate(df.columns):
        if target_n in norm(c):
            return i

    raise KeyError(
        f"Start column '{target}' not found. Available columns include: {list(df.columns)[:30]}"
    )


def _norm_col_name(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def _col_map(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized column names to actual column names."""
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
    """Evaluate AST to a boolean mask (Series[bool])."""
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


def count_expression(
    csv_path: str,
    expr: str,
    encoding: str | None = None,
) -> int:
    """Count rows matching a boolean expression over 0/1 flag columns.

    Grammar (case-insensitive operators):
      expr  := or_expr
      or_expr := and_expr (OR and_expr)*
      and_expr := unary (AND unary)*
      unary := NOT unary | primary
      primary := IDENT | '(' expr ')'

    IDENT is a column name token (use underscores instead of spaces, e.g. adverse_drug_event).
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    tokens = _tokenize(expr)
    ast = _Parser(tokens).parse()
    mask = _eval_ast(df, ast)
    return int(mask.sum())


def count_ones_from_column(
    csv_path: str,
    start_column: str = "lung cancer",
    encoding: str | None = None,
) -> tuple[pd.Series, int]:
    """Count 1s for each column from `start_column` to the end.

    Returns:
      - counts: Series indexed by column name
      - total: sum of counts across columns
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    start_idx = _find_column_index(df, start_column)

    sub = df.iloc[:, start_idx:]
    # Coerce to numeric 0/1-ish, treat non-numeric/missing as 0
    sub_num = sub.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    counts = sub_num.eq(1).sum(axis=0)
    total = int(counts.sum())
    return counts, total


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Count 1s in binary flag columns, and optionally count boolean expressions.")
    p.add_argument("--csv", type=str, default="../data/processed/case_reports_all_flags.csv", help="Input CSV path")
    p.add_argument(
        "--start_column",
        type=str,
        default="cancer",
        help="Start counting columns from this column (inclusive)",
    )
    p.add_argument("--encoding", type=str, default=None, help="CSV encoding (optional)")
    p.add_argument(
        "--expr",
        action="append",
        default=[],
        help="Boolean expression over columns to count (repeatable), e.g. 'cancer AND ADE AND ADR'",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # 1) Keep existing behavior: per-column counts from `start_column`.
    counts, total = count_ones_from_column(
        csv_path=args.csv,
        start_column=args.start_column,
        encoding=args.encoding,
    )

    for col, v in counts.items():
        print(f"{col} {int(v)}")
    print(f"TOTAL {total}")

    # 2) Then print expression counts (if any).
    for expr in args.expr:
        n = count_expression(args.csv, expr=expr, encoding=args.encoding)
        print(f"EXPR {expr} {n}")


if __name__ == "__main__":
    main()