"""Generate summary tables and reports."""
import pandas as pd
import numpy as np


def summarize_factor(arr):
    """Summary statistics for a raster array."""
    valid = arr[~np.isnan(arr)]
    return {
        "min": float(np.min(valid)), "max": float(np.max(valid)),
        "mean": float(np.mean(valid)), "std": float(np.std(valid)),
        "n": int(len(valid)),
    }


def classify_TCVN(ser):
    """Classify SER by TCVN 5299:2009 (5 classes)."""
    cls = np.full(ser.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(ser)
    cls[valid & (ser <= 1)] = 1
    cls[valid & (ser > 1) & (ser <= 5)] = 2
    cls[valid & (ser > 5) & (ser <= 10)] = 3
    cls[valid & (ser > 10) & (ser <= 50)] = 4
    cls[valid & (ser > 50)] = 5
    return cls


def tcvn_summary(ser_class, pixel_ha):
    """Summary of TCVN classes."""
    labels = {
        1: "I (≤1)", 2: "II (1–5)", 3: "III (5–10)",
        4: "IV (10–50)", 5: "V (>50)",
    }
    rows = []
    total = int(np.sum(~np.isnan(ser_class)))
    for code in [1, 2, 3, 4, 5]:
        n = int(np.sum(ser_class == code))
        rows.append({
            "Class": labels[code], "N_pixels": n,
            "Area_ha": n * pixel_ha,
            "Proportion": 100 * n / total if total > 0 else 0,
        })
    return pd.DataFrame(rows)