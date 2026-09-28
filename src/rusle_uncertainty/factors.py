"""RUSLE factor derivation."""
import numpy as np


# ============================================================
# R FACTOR
# ============================================================
def compute_R(annual_rainfall_mm):
    """R = 0.548257 * P - 59.5 (Nguyen Trong Ha, 1996)."""
    return 0.548257 * annual_rainfall_mm - 59.5


# ============================================================
# K FACTOR
# ============================================================
K_TABLE = {
    1: [0.05, 0.03, 0.02], 2: [0.12, 0.10, 0.08],
    3: [0.27, 0.24, 0.19], 4: [0.38, 0.34, 0.29],
    5: [0.48, 0.42, 0.33], 6: [0.60, 0.52, 0.42],
    7: [0.28, 0.25, 0.21], 8: [0.27, 0.25, 0.21],
    9: [0.37, 0.32, 0.26], 10: [0.14, 0.13, 0.12],
    11: [0.25, 0.23, 0.19], 12: [0.165, 0.165, 0.165],
}

METRIC_FACTOR = 0.1317
VAN_BEMMELEN = 1.724


def classify_usda_strict(sand, silt, clay):
    """USDA classification (12 classes). Returns (class_id, fallback_mask)."""
    result = np.full(sand.shape, -9999, dtype=np.int32)
    rules = [
        (1, (sand >= 85)),
        (2, (sand >= 70) & (sand < 85) & (clay < 15)),
        (3, (sand >= 50) & (sand < 70) & (clay <= 20) & (silt <= 50)),
        (4, (sand >= 23) & (sand < 50) & (silt >= 28) & (silt < 50) & (clay >= 7) & (clay < 27)),
        (5, (sand < 50) & (silt >= 50) & (clay < 27)),
        (6, (sand < 20) & (silt >= 80) & (clay < 12)),
        (7, (sand >= 20) & (sand < 45) & (clay >= 27) & (clay < 40)),
        (8, (sand >= 45) & (sand < 80) & (clay >= 20) & (clay < 35)),
        (9, (sand < 20) & (silt >= 40) & (silt < 73) & (clay >= 27) & (clay < 40)),
        (10, (sand >= 45) & (sand < 65) & (clay >= 35) & (clay < 55)),
        (11, (sand < 20) & (silt >= 40) & (silt < 60) & (clay >= 40) & (clay < 60)),
        (12, (sand < 45) & (silt < 40) & (clay >= 40)),
    ]
    for cls_id, mask in rules:
        result[mask] = cls_id
    fallback_mask = (result == -9999)
    return result, fallback_mask


def compute_K(sand_pct, silt_pct, clay_pct, soc_pct, valid_mask,
              apply_fallback=True):
    """Compute metric K-factor via pixel-wise OM interpolation."""
    sum_ssc = sand_pct + silt_pct + clay_pct
    mask_norm = (sum_ssc > 0) & valid_mask

    sand_n = np.where(mask_norm, sand_pct / sum_ssc * 100, np.nan)
    silt_n = np.where(mask_norm, silt_pct / sum_ssc * 100, np.nan)
    clay_n = np.where(mask_norm, clay_pct / sum_ssc * 100, np.nan)

    sand_f = np.where(valid_mask, sand_n, -9999)
    silt_f = np.where(valid_mask, silt_n, -9999)
    clay_f = np.where(valid_mask, clay_n, -9999)
    tex_class, fallback_mask = classify_usda_strict(sand_f, silt_f, clay_f)

    if apply_fallback:
        tex_class[fallback_mask & valid_mask] = 4

    om = np.where(valid_mask, soc_pct * VAN_BEMMELEN, np.nan)
    K_us = np.full(sand_pct.shape, np.nan, dtype=np.float32)

    for tex_id, (K0, K2, K4) in K_TABLE.items():
        mask = (tex_class == tex_id) & valid_mask
        if not np.any(mask):
            continue
        om_v = om[mask]
        K_v = np.full(om_v.shape, np.nan, dtype=np.float32)
        if tex_id == 12:
            K_v[:] = K0
        else:
            c1 = om_v >= 4
            K_v[c1] = K4
            c2 = (om_v >= 2) & (om_v < 4)
            K_v[c2] = K2 + (K4 - K2) / 2.0 * (om_v[c2] - 2.0)
            c3 = (om_v >= 0.5) & (om_v < 2)
            K_v[c3] = K0 + (K2 - K0) / 1.5 * (om_v[c3] - 0.5)
            c4 = om_v < 0.5
            K_v[c4] = K0
        K_us[mask] = K_v

    return K_us * METRIC_FACTOR, tex_class, fallback_mask


# ============================================================
# LS FACTOR
# ============================================================
def compute_LS(flow_acc, slope_rad, cell_size, cap_percentile=99):
    """Moore & Burch (1986) with percentile cap."""
    FA = np.where(flow_acc < 1, 1, flow_acc)
    L = ((FA * cell_size) / 22.13) ** 0.4
    S = (np.sin(slope_rad) / 0.0896) ** 1.3
    LS = L * S
    if cap_percentile is not None:
        cap_val = np.nanpercentile(LS, cap_percentile)
        LS = np.where(LS > cap_val, cap_val, LS)
    return LS


# ============================================================
# C FACTOR
# ============================================================
C_VALUES = {
    "water": 0.0, "forest": 0.005, "builtup": 0.02,
    "agriculture": 0.30, "other": 0.80,
}


def assign_C(lulc_array, lulc_mapping):
    """Assign C values from LULC classes."""
    C = np.full(lulc_array.shape, np.nan, dtype=np.float32)
    for cls_id, c_val in lulc_mapping.items():
        C[lulc_array == cls_id] = c_val
    return C


# ============================================================
# SER
# ============================================================
def compute_SER(R, K, LS, C, P=1.0):
    """SER = R × K × LS × C × P."""
    mask = (~np.isnan(R) & ~np.isnan(K) & ~np.isnan(LS) & ~np.isnan(C))
    SER = np.full(R.shape, np.nan, dtype=np.float32)
    SER[mask] = R[mask] * K[mask] * LS[mask] * C[mask] * P
    return np.where(SER < 0, np.nan, SER)