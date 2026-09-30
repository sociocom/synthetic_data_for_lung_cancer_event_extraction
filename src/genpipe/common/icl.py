# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


ICL_GLOB = "[0-9][0-9]_*.txt"


def _make_rng(seed: int, record_id: str) -> random.Random:
    h = hashlib.md5(record_id.encode("utf-8")).hexdigest()
    rid_int = int(h[:8], 16)
    return random.Random(seed ^ rid_int)


def _read_context_as_text(p: Path) -> str:
    """Read a context file and return a text representation for prompting.

    - .txt: returned as-is
    - .json: pretty-printed JSON (ensure_ascii=False)

    This keeps the ICL interface text-only while allowing structured outputs
    (e.g. annotations) to be stored as JSON.
    """

    suf = p.suffix.lower()
    if suf in {".txt", ".md"}:
        return p.read_text(encoding="utf-8").rstrip()

    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        return json.dumps(obj, ensure_ascii=False, indent=2).rstrip()

    # Fallback: treat as text
    return p.read_text(encoding="utf-8").rstrip()


def _glob_many(d: Path, patterns: str | Iterable[str]) -> List[Path]:
    """Glob with one or many patterns and return a sorted, de-duplicated list."""
    if isinstance(patterns, str):
        patterns = [patterns]
    seen: dict[str, Path] = {}
    for pat in patterns:
        for p in d.glob(pat):
            # key by full path string to dedupe
            seen[str(p)] = p
    return sorted(seen.values())


@dataclass(frozen=True)
class NoICL:
    def context_for(self, record_id: str) -> str:
        return ""


@dataclass(frozen=True)
class SingleTextICL:
    context_dir: str
    n: int = 3
    seed: int = 42
    glob: str | Iterable[str] = ICL_GLOB

    def _load_texts(self) -> List[str]:
        d = Path(self.context_dir)
        if not d.exists():
            raise FileNotFoundError(f"Context dir not found: {self.context_dir}")

        paths = _glob_many(d, self.glob)
        if not paths:
            raise FileNotFoundError(f"No context files matched {self.glob} in {self.context_dir}")

        return [_read_context_as_text(p) for p in paths]

    def context_for(self, record_id: str) -> str:
        texts = self._load_texts()
        if len(texts) < self.n:
            raise ValueError(f"Need at least {self.n} context files, got {len(texts)}")

        rng = _make_rng(self.seed, record_id)
        chosen = rng.sample(texts, k=self.n)

        parts = []
        for i, txt in enumerate(chosen, start=1):
            parts.append(f"【例{i}】\n{txt}")
        return "\n\n".join(parts)


@dataclass(frozen=True)
class PairedICL:
    icl_input_dir: str
    icl_output_dir: str
    n: int = 3
    seed: int = 42
    input_label: str | None = None
    output_label: str | None = None
    input_glob: str | Iterable[str] = ICL_GLOB
    # annotations等は.jsonで持つケースがあるので、出力側はデフォルトでtxt/jsonを許可
    # NOTE: pathlib.Path.glob は brace 展開 {..} 非対応なので、複数パターンで指定する
    output_glob: str | Iterable[str] = ("[0-9][0-9]_*.txt", "[0-9][0-9]_*.json")

    def __post_init__(self) -> None:
        # dataclass(frozen=True)なのでobject.__setattr__は使わず、検証だけ行う
        if not self.input_label or not self.output_label:
            raise ValueError(
                "PairedICL requires explicit input_label/output_label (no defaults). "
                "Example: PairedICL(..., input_label='退院サマリ', output_label='診療録')"
            )

    def _prefix(self, p: Path) -> str:
        name = p.name
        if "_" not in name:
            raise ValueError(f"Bad context filename (missing '_'): {p}")
        return name.split("_", 1)[0]

    def _load_pairs(self) -> List[Tuple[str, str]]:
        in_dir = Path(self.icl_input_dir)
        out_dir = Path(self.icl_output_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"ICL input dir not found: {self.icl_input_dir}")
        if not out_dir.exists():
            raise FileNotFoundError(f"ICL output dir not found: {self.icl_output_dir}")

        in_paths = _glob_many(in_dir, self.input_glob)
        out_paths = _glob_many(out_dir, self.output_glob)
        if not in_paths:
            raise FileNotFoundError(f"No ICL input files matched {self.input_glob} in {self.icl_input_dir}")
        if not out_paths:
            raise FileNotFoundError(f"No ICL output files matched {self.output_glob} in {self.icl_output_dir}")

        in_map: dict[str, List[Path]] = {}
        out_map: dict[str, List[Path]] = {}
        for p in in_paths:
            in_map.setdefault(self._prefix(p), []).append(p)
        for p in out_paths:
            out_map.setdefault(self._prefix(p), []).append(p)

        common = sorted(set(in_map.keys()) & set(out_map.keys()))
        if not common:
            raise ValueError("No matching prefixes between ICL input and output dirs.")

        pairs: List[Tuple[str, str]] = []
        for k in common:
            in_list = sorted(in_map[k])
            out_list = sorted(out_map[k])
            m = min(len(in_list), len(out_list))
            for i in range(m):
                inp = _read_context_as_text(in_list[i])
                outp = _read_context_as_text(out_list[i])
                pairs.append((inp, outp))

        return pairs

    def context_for(self, record_id: str) -> str:
        pairs = self._load_pairs()
        if len(pairs) < self.n:
            raise ValueError(f"Need at least {self.n} paired contexts, got {len(pairs)}")

        rng = _make_rng(self.seed, record_id)
        chosen = rng.sample(pairs, k=self.n)

        parts = []
        for i, (inp, outp) in enumerate(chosen, start=1):
            parts.append(f"【例{i}の{self.input_label}】\n{inp}\n\n【例{i}の{self.output_label}】\n{outp}")
        return "\n\n".join(parts)
