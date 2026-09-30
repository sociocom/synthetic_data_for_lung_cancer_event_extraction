# schema.py
# -*- coding: utf-8 -*-

import re

# ===== Keys and types =====
KEY_TYPES = {
    "Timestamp": "date",
    "ProcedureDiagnosis": "list",
    "Site": "list",
    "TumorSize": "number",
    "TNM_T": "list",
    "TNM_N": "list",
    "SitesOfMetastasis": "list",
    "TNM_M": "list",
    "Stage": "list",
    "Recurrence": "list",
    "SiteOfRecurrence": "list",
    "HistologicalGrading": "list",
    "Reaction": "list",
    "ToxicityGrade": "list",
    "Cytology": "list",
    "TBB": "list",
    "TBLB": "list",
    "EBUS_TBNA": "list",
    "PDL1": "number",
    "PETCT": "number",
    "Oncomine": "list",
    "TTF1": "list",
    "p40": "list",
    "p63": "list",
    "CK5_6": "list",
    "NapsinA": "list",
    "EGFR": "list",
    "ALK": "list",
    "ROS1": "list",
    "RET": "list",
    "BRAF": "list",
    "KRAS": "list",
    "HER2": "list",
    "PathologicalExamination": "list",
    "AssessedAnatomicSite": "list",
    "NTRK": "list",
    "NGS": "list",
    "ROSE": "list",
    "Treatment": "list",
    "RadiationDose": "number",
    "RadiationFractions": "number",
}

NUMBER_KEYS = {k for k, t in KEY_TYPES.items() if t == "number"}
LIST_KEYS = {k for k, t in KEY_TYPES.items() if t == "list"}
DATE_KEYS = {k for k, t in KEY_TYPES.items() if t == "date"}

# Integer treated keys
INTEGER_KEYS = {"RadiationFractions", "TumorSize"}

NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

# Split list-like strings by comma or plus, also accept Japanese variants.
LIST_SPLIT_RE = re.compile(r"[,+]+")

TRANS_TABLE = str.maketrans({
    "、": ",",
    "，": ",",
    "＋": "+",
    "､": ",",
})

DATE_PATTERNS = [
    re.compile(r"^\s*(\d{4})[\/\.](\d{1,2})[\/\.](\d{1,2})\s*$"),
    re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$"),
    re.compile(r"^\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*$"),
    re.compile(r"^\s*(\d{4})[\/\.-](\d{1,2})\s*$"),
    re.compile(r"^\s*(\d{4})年\s*(\d{1,2})月\s*$"),
    re.compile(r"^\s*(\d{4})\s*$"),
]
