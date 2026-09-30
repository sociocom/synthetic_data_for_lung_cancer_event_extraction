# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import heapq
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line is not an object at {path}:{ln}")
            yield obj


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    out_parent = Path(path).parent
    if str(out_parent) not in ["", "."]:
        out_parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: str, objs: Iterable[Dict[str, Any]]) -> None:
    """Overwrite-write JSONL file."""
    out_parent = Path(path).parent
    if str(out_parent) not in ["", "."]:
        out_parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _iter_jsonl_with_line(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line is not an object at {path}:{ln}")
            yield ln, obj


def sort_jsonl_inplace_by_numeric_id(
    path: str,
    id_key: str = "id",
    chunk_size: int = 200_000,
    tmp_dir: Optional[str] = None,
) -> None:
    """Sort JSONL file by numeric id_key ascending and replace the original file atomically.

    This uses an external sort (chunk sort + k-way merge) to avoid loading the entire file.
    Lines missing id_key are treated as errors.
    """

    p = Path(path)
    if not p.exists():
        return

    tmp_files: List[str] = []

    def _flush_chunk(chunk: List[Tuple[int, Dict[str, Any]]]) -> None:
        chunk.sort(key=lambda t: t[0])
        fd, tpath = tempfile.mkstemp(prefix="jsonl_sort_chunk_", suffix=".jsonl", dir=tmp_dir)
        os.close(fd)
        with open(tpath, "w", encoding="utf-8") as f:
            for _, obj in chunk:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp_files.append(tpath)

    chunk: List[Tuple[int, Dict[str, Any]]] = []
    for ln, obj in _iter_jsonl_with_line(path):
        if id_key not in obj:
            raise ValueError(f"Missing '{id_key}' at {path}:{ln}")
        try:
            nid = int(str(obj[id_key]))
        except ValueError as e:
            raise ValueError(f"Non-numeric '{id_key}' at {path}:{ln}: {obj[id_key]!r}") from e
        chunk.append((nid, obj))
        if len(chunk) >= chunk_size:
            _flush_chunk(chunk)
            chunk = []

    if chunk:
        _flush_chunk(chunk)

    # If there was only one chunk, we can just replace by rewriting its file.
    fd, out_tmp = tempfile.mkstemp(prefix="jsonl_sort_out_", suffix=".jsonl", dir=str(p.parent))
    os.close(fd)

    def _iter_chunk_file(tpath: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for ln, obj in _iter_jsonl_with_line(tpath):
            if id_key not in obj:
                raise ValueError(f"Missing '{id_key}' in temp chunk {tpath}:{ln}")
            nid = int(str(obj[id_key]))
            yield nid, obj

    try:
        iters = [_iter_chunk_file(t) for t in tmp_files]
        # heapq.merge merges sorted iterators
        merged = heapq.merge(*iters, key=lambda t: t[0])
        with open(out_tmp, "w", encoding="utf-8") as f:
            for _, obj in merged:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        os.replace(out_tmp, path)
    finally:
        # cleanup
        try:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        except OSError:
            pass
        for t in tmp_files:
            try:
                os.remove(t)
            except OSError:
                pass


def load_done_ids(output_jsonl: str, id_key: str = "id") -> set[str]:
    done: set[str] = set()
    p = Path(output_jsonl)
    if not p.exists():
        return done

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in existing output at {output_jsonl}:{ln}")
            if isinstance(obj, dict) and id_key in obj:
                done.add(str(obj[id_key]))
    return done
