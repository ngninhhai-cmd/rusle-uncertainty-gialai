"""
run_all.py
===========
Main workflow runner for Paper 1.

Chạy toàn bộ pipeline:
    1. Load factor data
    2. Uncertainty propagation
    3. Resolution sensitivity
    4. Sensitivity analysis (Sobol + correlated MC)
    5. LISA
    6. Generate figures
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rusle_uncertainty.config import Config
from rusle_uncertainty import io_utils as iou
from rusle_uncertainty import factors as fac
from rusle_uncertainty import uncertainty as unc
from rusle_uncertainty import sensitivity as sens
from rusle_uncertainty import resolution as res_mod
from rusle_uncertainty import lisa as lisa_mod
from rusle_uncertainty import visualization as viz
from rusle_uncertainty import report as rep


def setup_logging(output_root):
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )