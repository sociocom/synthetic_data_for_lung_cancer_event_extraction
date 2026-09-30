# -*- coding: utf-8 -*-

"""Backward-compatible shim.

The SFT implementation has been modularized under `trainpipe.sft.*`.
Import from `trainpipe` or `trainpipe.sft` modules instead of this file.

This file remains to avoid breaking existing scripts/imports.
"""

from __future__ import annotations

from trainpipe.sft.prompting import STRUCTURED_TEMPLATE
from trainpipe.sft.args import TrainSFTArgs
from trainpipe.sft.run import run_sft

__all__ = [
    "STRUCTURED_TEMPLATE",
    "TrainSFTArgs",
    "run_sft",
]
