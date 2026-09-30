#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Backward-compatible wrapper for inference CLI.

This module keeps the historical entrypoint (`src/inference.py`) while the
implementation lives in `inferpipe` modules.
"""

from __future__ import annotations

from inferpipe.cli.run_inference import main


if __name__ == "__main__":
    main()
