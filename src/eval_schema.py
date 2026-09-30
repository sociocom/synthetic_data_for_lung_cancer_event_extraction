from __future__ import annotations

try:
    from src.schema import KEY_TYPES
except ModuleNotFoundError:
    from schema import KEY_TYPES


# Evaluation-only aliases. Training/inference schema remains unchanged.
EVAL_KEY_ALIASES = {
    "TBB": "TBB_TBLB",
    "TBLB": "TBB_TBLB",
}


def canonicalize_eval_key(key: str) -> str:
    return EVAL_KEY_ALIASES.get(key, key)


def build_eval_key_types() -> dict[str, str]:
    merged = {k: v for k, v in KEY_TYPES.items() if k not in EVAL_KEY_ALIASES}
    merged["TBB_TBLB"] = KEY_TYPES["TBLB"]
    return merged


EVAL_KEY_TYPES = build_eval_key_types()
