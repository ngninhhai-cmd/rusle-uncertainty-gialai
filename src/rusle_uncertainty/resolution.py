"""Multi-resolution sensitivity analysis."""
import numpy as np
import pandas as pd


def compute_S_res(stats_df, res_high=250, res_low=30, metric="Max"):
    """S_res = (SER_high - SER_low) / SER_low × 100%."""
    v_high = stats_df.loc[stats_df["Resolution_m"] == res_high, metric].values[0]
    v_low = stats_df.loc[stats_df["Resolution_m"] == res_low, metric].values[0]
    return (v_high - v_low) / v_low * 100


def compute_S_range(stats_df, res_target=150, metric="Max"):
    """S_range = (SER_target - min(other)) / min(other) × 100%."""
    target = stats_df.loc[stats_df["Resolution_m"] == res_target, metric].values[0]
    others = stats_df.loc[stats_df["Resolution_m"] != res_target, metric]
    min_other = others.min()
    return (target - min_other) / min_other * 100


def summarize_variation(stats_df, metrics=("Mean", "Median", "SD", "P95", "P99", "Max")):
    """Summarize variation across resolutions."""
    summary = {}
    for m in metrics:
        vals = stats_df[m].values
        summary[m] = {
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "variation_pct": float((np.max(vals) - np.min(vals)) / np.mean(vals) * 100),
        }
    return summary


def sensitivity_indices(stats_df, metrics=("Max", "P99", "P95")):
    """S_res and S_range for multiple metrics."""
    return pd.DataFrame([{
        "Metric": metric,
        "S_res_pct": compute_S_res(stats_df, metric=metric),
        "S_range_pct": compute_S_range(stats_df, metric=metric),
    } for metric in metrics])