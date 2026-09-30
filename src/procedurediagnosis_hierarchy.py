#!/usr/bin/env python3
"""
ProcedureDiagnosis hierarchy utilities.

This module defines candidate labels and descendant compatibility used by
evaluation mode `pred_child_of_gold`:
- exact match is always correct
- pred is also correct when pred is a strict descendant of gold
- reverse direction (pred is broader than gold) is not correct
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Optional


PROCEDUREDIAGNOSIS_CANDIDATES: Final[list[str]] = [
    "右上葉肺腺癌",
    "右中葉肺腺癌",
    "右下葉肺腺癌",
    "左上葉肺腺癌",
    "左下葉肺腺癌",
    "右上葉肺扁平上皮癌",
    "右中葉肺扁平上皮癌",
    "右下葉肺扁平上皮癌",
    "左上葉肺扁平上皮癌",
    "左下葉肺扁平上皮癌",
    "右上葉肺小細胞癌",
    "右中葉肺小細胞癌",
    "右下葉肺小細胞癌",
    "左上葉肺小細胞癌",
    "左下葉肺小細胞癌",
    "右上葉肺大細胞癌",
    "右中葉肺大細胞癌",
    "右下葉肺大細胞癌",
    "左上葉肺大細胞癌",
    "左下葉肺大細胞癌",
    "右肺門部肺腺癌",
    "左肺門部肺腺癌",
    "右肺門部肺扁平上皮癌",
    "左肺門部肺扁平上皮癌",
    "右肺門部肺大細胞癌",
    "左肺門部肺大細胞癌",
    "右肺門部肺小細胞癌",
    "左肺門部肺小細胞癌",
    "右肺腺癌",
    "右肺扁平上皮癌",
    "右肺小細胞癌",
    "右肺大細胞癌",
    "左肺腺癌",
    "左肺扁平上皮癌",
    "左肺小細胞癌",
    "左肺大細胞癌",
    "肺腺癌",
    "肺扁平上皮癌",
    "肺小細胞癌",
    "肺大細胞癌",
    "右上葉肺癌",
    "右中葉肺癌",
    "右下葉肺癌",
    "左上葉肺癌",
    "左下葉肺癌",
    "右肺門部肺癌",
    "左肺門部肺癌",
    "右肺癌",
    "左肺癌",
    "肺癌",
    "肺癌疑い",
    "肺癌骨転移",
    "その他",
]


def canonicalize_procedurediagnosis(label: str) -> str:
    """Normalize label token to the scorer's canonical style."""
    # normalize.py converts "癌" to "がん" for ProcedureDiagnosis.
    return str(label).strip().replace("癌", "がん")


_CANDIDATES_CANONICAL: Final[set[str]] = {
    canonicalize_procedurediagnosis(v) for v in PROCEDUREDIAGNOSIS_CANDIDATES
}
_SPECIAL_EXACT_ONLY: Final[set[str]] = {
    canonicalize_procedurediagnosis("肺癌疑い"),
    canonicalize_procedurediagnosis("肺癌骨転移"),
    canonicalize_procedurediagnosis("その他"),
}

_HIST_GROUP = "腺がん|扁平上皮がん|小細胞がん|大細胞がん"
_RE_SIDE_REGION_HIST = re.compile(rf"^([左右])(上葉|中葉|下葉)肺({_HIST_GROUP})$")
_RE_HILUM_HIST = re.compile(rf"^([左右])肺門部肺({_HIST_GROUP})$")
_RE_SIDE_HIST = re.compile(rf"^([左右])肺({_HIST_GROUP})$")
_RE_LUNG_HIST = re.compile(rf"^肺({_HIST_GROUP})$")
_RE_SIDE_REGION_GENERIC = re.compile(r"^([左右])(上葉|中葉|下葉)肺がん$")
_RE_HILUM_GENERIC = re.compile(r"^([左右])肺門部肺がん$")
_RE_SIDE_GENERIC = re.compile(r"^([左右])肺がん$")


@dataclass(frozen=True)
class _PDNode:
    side: Optional[str]
    region: Optional[str]
    hist: Optional[str]
    special: Optional[str]


@lru_cache(maxsize=None)
def _parse_label(canonical_label: str) -> Optional[_PDNode]:
    if canonical_label in _SPECIAL_EXACT_ONLY:
        return _PDNode(side=None, region=None, hist=None, special=canonical_label)
    if canonical_label == "肺がん":
        return _PDNode(side=None, region=None, hist=None, special=None)

    m = _RE_SIDE_REGION_HIST.match(canonical_label)
    if m:
        side, region, hist = m.groups()
        return _PDNode(side=side, region=region, hist=hist, special=None)

    m = _RE_HILUM_HIST.match(canonical_label)
    if m:
        side, hist = m.groups()
        return _PDNode(side=side, region="肺門部", hist=hist, special=None)

    m = _RE_SIDE_HIST.match(canonical_label)
    if m:
        side, hist = m.groups()
        return _PDNode(side=side, region=None, hist=hist, special=None)

    m = _RE_LUNG_HIST.match(canonical_label)
    if m:
        (hist,) = m.groups()
        return _PDNode(side=None, region=None, hist=hist, special=None)

    m = _RE_SIDE_REGION_GENERIC.match(canonical_label)
    if m:
        side, region = m.groups()
        return _PDNode(side=side, region=region, hist=None, special=None)

    m = _RE_HILUM_GENERIC.match(canonical_label)
    if m:
        (side,) = m.groups()
        return _PDNode(side=side, region="肺門部", hist=None, special=None)

    m = _RE_SIDE_GENERIC.match(canonical_label)
    if m:
        (side,) = m.groups()
        return _PDNode(side=side, region=None, hist=None, special=None)

    return None


def is_known_procedurediagnosis_candidate(label: str) -> bool:
    return canonicalize_procedurediagnosis(label) in _CANDIDATES_CANONICAL


def is_pred_child_or_equal_procedurediagnosis(pred_label: str, gold_label: str) -> bool:
    """
    Return True when pred is equal to or a descendant of gold.

    Unknown / malformed labels are treated as non-compatible unless exact.
    """
    pred = canonicalize_procedurediagnosis(pred_label)
    gold = canonicalize_procedurediagnosis(gold_label)

    if pred == gold:
        return True

    if pred not in _CANDIDATES_CANONICAL or gold not in _CANDIDATES_CANONICAL:
        return False

    p = _parse_label(pred)
    g = _parse_label(gold)
    if p is None or g is None:
        return False

    # Special nodes are exact-only (already handled above).
    if p.special is not None or g.special is not None:
        return False

    # Descendant-or-equal by constrained attributes:
    # pred may add detail, but must not violate any explicit gold attribute.
    side_ok = (g.side is None) or (p.side == g.side)
    region_ok = (g.region is None) or (p.region == g.region)
    hist_ok = (g.hist is None) or (p.hist == g.hist)
    return side_ok and region_ok and hist_ok


__all__ = [
    "PROCEDUREDIAGNOSIS_CANDIDATES",
    "canonicalize_procedurediagnosis",
    "is_known_procedurediagnosis_candidate",
    "is_pred_child_or_equal_procedurediagnosis",
]
