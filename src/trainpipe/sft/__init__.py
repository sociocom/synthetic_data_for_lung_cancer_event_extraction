"""Modular SFT implementation.

Public API:
  - TrainSFTArgs
  - TrainSFTPreparedArgs
  - run_sft
  - run_sft_prepared
  - STRUCTURED_TEMPLATE
"""

from trainpipe.sft.args import TrainSFTArgs, TrainSFTPreparedArgs
from trainpipe.sft.run import run_sft, run_sft_prepared
from trainpipe.sft.prompting import STRUCTURED_TEMPLATE

__all__ = [
    "TrainSFTArgs",
    "TrainSFTPreparedArgs",
    "run_sft",
    "run_sft_prepared",
    "STRUCTURED_TEMPLATE",
]
