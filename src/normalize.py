# normalize.py
# -*- coding: utf-8 -*-

import json
import re
from typing import Any, List, Tuple, Optional

try:
    from src.schema import (
        KEY_TYPES,
        NUMBER_KEYS,
        LIST_KEYS,
        DATE_KEYS,
        INTEGER_KEYS,
        NUMERIC_RE,
        LIST_SPLIT_RE,
        TRANS_TABLE,
        DATE_PATTERNS,
    )
except ModuleNotFoundError:
    # Fallback for direct script execution (e.g., python ../src/inference.py)
    from schema import (
        KEY_TYPES,
        NUMBER_KEYS,
        LIST_KEYS,
        DATE_KEYS,
        INTEGER_KEYS,
        NUMERIC_RE,
        LIST_SPLIT_RE,
        TRANS_TABLE,
        DATE_PATTERNS,
    )

Atom = Tuple[str, Any]

# ===== Added, per-key token normalization =====
def normalize_token_for_key(key: str, tok: str) -> str:
    t = tok.strip()
    if not t:
        return t
    if key == "ProcedureDiagnosis":
        t = t.replace("癌", "がん")
    return t


def normalize_date(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s2 = s.translate(TRANS_TABLE)
    for pat in DATE_PATTERNS:
        m = pat.match(s2)
        if not m:
            continue
        parts = list(m.groups())
        if len(parts) == 3:
            y, mo, d = parts
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        if len(parts) == 2:
            y, mo = parts
            return f"{int(y):04d}-{int(mo):02d}"
        if len(parts) == 1:
            y = parts[0]
            return f"{int(y):04d}"
    return s2


def split_list_tokens(text: Any) -> List[str]:
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    s = s.translate(TRANS_TABLE)
    parts = LIST_SPLIT_RE.split(s)
    toks: List[str] = []
    for p in parts:
        t = p.strip()
        if t:
            toks.append(t)
    return toks


def normalize_atoms(key: str, raw_value: Any) -> List[Atom]:
    if raw_value is None:
        if key in ("Timestamp", "Site"):
            return []
        return [("str", "None")]

    if key in DATE_KEYS:
        ts = normalize_date(raw_value)
        if ts == "":
            return []
        return [("str", ts)]

    if key in NUMBER_KEYS:
        if isinstance(raw_value, (int, float)):
            v = float(raw_value)
            if key in INTEGER_KEYS:
                return [("num_int", int(round(v)))]
            return [("num_float", round(v, 1))]

        if isinstance(raw_value, str):
            s = raw_value.strip().translate(TRANS_TABLE)
            if s == "":
                if key in ("Timestamp", "Site"):
                    return []
                return [("str", "")]
            if NUMERIC_RE.match(s):
                v = float(s)
                if key in INTEGER_KEYS:
                    return [("num_int", int(round(v)))]
                return [("num_float", round(v, 1))]
            return [("str", s)]

        return [("str", str(raw_value).strip())]

    if key in LIST_KEYS:
        atoms: List[Atom] = []
        if isinstance(raw_value, list):
            for elem in raw_value:
                for tok in split_list_tokens(elem):
                    atoms.append(("str", normalize_token_for_key(key, tok)))
        else:
            for tok in split_list_tokens(raw_value):
                atoms.append(("str", normalize_token_for_key(key, tok)))

        if not atoms:
            s = str(raw_value).strip()
            if s == "":
                if key in ("Timestamp", "Site"):
                    return []
                return [("str", "")]
            atoms = [("str", normalize_token_for_key(key, s))]
        return atoms

    return [("str", str(raw_value).strip())]


# ===== Lenient JSON extraction, drop only broken key,value pairs =====
PRIM_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return i

def _read_json_string(s: str, i: int):
    # s[i] == '"'
    n = len(s)
    j = i + 1
    esc = False
    while j < n:
        ch = s[j]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            return s[i:j+1], j + 1
        j += 1
    return None, i

def _looks_like_valid_primitive(seg: str) -> bool:
    t = seg.strip()
    if t in ("true", "false", "null"):
        return True
    if PRIM_NUM_RE.match(t):
        return True
    return False

def _consume_until_next_key_or_end(s: str, i: int):
    # consume broken region until we see , "key" or } at top level
    n = len(s)
    in_str = False
    esc = False
    depth = 0
    j = i
    while j < n:
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth = max(0, depth - 1)
            elif depth == 0 and ch == "}":
                return j, "}"
            elif depth == 0 and ch == ",":
                k = _skip_ws(s, j + 1)
                if k < n and s[k] == '"':
                    return k, ","
        j += 1
    return n, None

def _read_value_segment(s: str, i: int):
    """
    returns (val_seg, next_i, delim, bad_value)
    next_i points to start of next key (quote) or to after }.
    """
    n = len(s)
    i = _skip_ws(s, i)
    if i >= n:
        return "", i, None, True

    ch = s[i]

    # string
    if ch == '"':
        seg, j = _read_json_string(s, i)
        if seg is None:
            return s[i:], n, None, True
        j = _skip_ws(s, j)
        if j < n and s[j] == ",":
            k = _skip_ws(s, j + 1)
            if k < n and s[k] == '"':
                return seg, k, ",", False
            # comma not followed by key, treat broken and consume
            nxt, delim = _consume_until_next_key_or_end(s, j + 1)
            return s[i:nxt].strip(), nxt, delim, True
        if j < n and s[j] == "}":
            return seg, j + 1, "}", False
        return seg, j, None, False

    # object or array
    if ch in "{[":
        open_ch = ch
        close_ch = "}" if ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        seg = s[i:j+1]
                        k = _skip_ws(s, j + 1)
                        if k < n and s[k] == ",":
                            kk = _skip_ws(s, k + 1)
                            if kk < n and s[kk] == '"':
                                return seg, kk, ",", False
                            nxt, delim = _consume_until_next_key_or_end(s, k + 1)
                            return s[i:nxt].strip(), nxt, delim, True
                        if k < n and s[k] == "}":
                            return seg, k + 1, "}", False
                        return seg, k, None, False
            j += 1
        return s[i:], n, None, True

    # primitive or broken token
    j = i
    while j < n and s[j] not in ",}":
        j += 1
    seg = s[i:j].strip()

    if j < n and s[j] == ",":
        k = _skip_ws(s, j + 1)
        if k < n and s[k] == '"':
            # normal key separator
            ok = _looks_like_valid_primitive(seg)
            return seg, k, ",", (not ok)
        # comma not followed by key, broken like 2.8,4.0 or 18万,....
        nxt, delim = _consume_until_next_key_or_end(s, j + 1)
        return s[i:nxt].strip(), nxt, delim, True

    if j < n and s[j] == "}":
        ok = _looks_like_valid_primitive(seg)
        return seg, j + 1, "}", (not ok)

    ok = _looks_like_valid_primitive(seg)
    return seg, j, None, (not ok)

def sanitize_object_text(obj_text: str) -> str:
    s = obj_text.strip()
    if not s.startswith("{"):
        return "{ }"
    n = len(s)
    i = 1
    kept: List[str] = []

    while True:
        i = _skip_ws(s, i)
        if i >= n:
            break
        if s[i] == "}":
            break
        if s[i] == ",":
            i += 1
            continue
        if s[i] != '"':
            break

        key_str, j = _read_json_string(s, i)
        if key_str is None:
            break
        i = _skip_ws(s, j)
        if i >= n or s[i] != ":":
            break
        i += 1

        val_seg, next_i, delim, bad = _read_value_segment(s, i)
        i = next_i

        if (not bad) and val_seg.strip() != "":
            kept.append(f"{key_str}: {val_seg.strip()}")

        if delim == "}":
            break

    return "{ " + ", ".join(kept) + " }"

def salvage_first_json_array(text: str) -> str:
    t = text.strip()
    start = t.find("[")
    if start == -1:
        raise ValueError("No '[' found in prediction text")
    depth = 0
    end = None
    in_str = False
    esc = False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end is None:
        raise ValueError("Could not find matching ']' for JSON array")
    return t[start:end+1]

def scan_object_texts_from_array(array_text: str) -> List[str]:
    s = array_text
    objs: List[str] = []
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(s[start:i+1])
                    start = None
    return objs

def extract_json_from_text_lenient(text: str):
    t = text.strip()

    # 1, if whole text is valid JSON already, use it
    try:
        return json.loads(t)
    except Exception:
        pass

    # 2, extract first array, then try parse as is
    arr = None
    try:
        arr = salvage_first_json_array(t)
        try:
            return json.loads(arr)
        except Exception:
            pass
    except Exception:
        # If the closing ']' is missing, salvage complete objects from the tail
        # after the first '[' so truncated generations can still be partially used.
        start = t.find("[")
        if start != -1:
            arr_part = t[start:]
            out = []
            for obj_text in scan_object_texts_from_array(arr_part):
                fixed = sanitize_object_text(obj_text)
                try:
                    out.append(json.loads(fixed))
                except Exception:
                    continue
            return out
        raise

    # 3, object-wise salvage, keep only parsable key,value pairs per object
    if arr is None:
        arr = salvage_first_json_array(t)
    out = []
    for obj_text in scan_object_texts_from_array(arr):
        fixed = sanitize_object_text(obj_text)
        try:
            out.append(json.loads(fixed))
        except Exception:
            # if even sanitized object fails, drop the whole object
            continue
    return out
