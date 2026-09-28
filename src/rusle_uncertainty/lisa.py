"""Local Moran's I (LISA) spatial cluster analysis."""
import numpy as np
import pandas as pd


CLUSTER_LABELS = {
    0: "Not significant", 1: "High-High (HH)",
    2: "Low-High (LH) - outlier", 3: "Low-Low (LL)",
    4: "High-Low (HL) - outlier",
}


def compute_lisa(ser_flat, valid_mask, shape,
                 n_permutations=999, significance=0.05,
                 weight_type="queen", seed=42):
    """Compute LISA on valid pixels."""
    from libpysal.weights import lat2W
    from esda.moran import Moran_Local

    rows, cols = shape
    rook = (weight_type == "rook")
    w_full = lat2W(rows, cols, rook=rook)
    valid_idx = np.where(valid_mask.flatten())[0]
    w_subset = w_full.subset(valid_idx.tolist())
    w_subset.transform = "r"

    ser_valid = ser_flat[valid_idx]
    ser_std = (ser_valid - np.mean(ser_valid)) / np.std(ser_valid)

    lisa = Moran_Local(ser_std, w_subset, permutations=n_permutations, seed=seed)

    cluster = np.zeros(len(ser_valid), dtype=np.int32)
    sig = lisa.p_sim < significance
    cluster[sig & (lisa.q == 1)] = 1
    cluster[sig & (lisa.q == 2)] = 2
    cluster[sig & (lisa.q == 3)] = 3
    cluster[sig & (lisa.q == 4)] = 4

    return cluster, lisa, valid_idx


def summarize_clusters(cluster, pixel_ha):
    """Summarize LISA cluster statistics."""
    rows = []
    for code in [1, 3, 4, 2, 0]:
        n = int(np.sum(cluster == code))
        rows.append({
            "Code": code, "Cluster_type": CLUSTER_LABELS[code],
            "N_pixels": n, "Area_ha": n * pixel_ha,
            "Percent": 100 * n / len(cluster),
        })
    return pd.DataFrame(rows)


def top_pixels(cluster, lisa, valid_idx, ser_flat, shape,
               cluster_code, top_n=5):
    """Top-N pixels by |I_i| for a given cluster type."""
    mask = (cluster == cluster_code)
    if not np.any(mask):
        return pd.DataFrame()

    I_vals = lisa.Is[mask]
    p_vals = lisa.p_sim[mask]
    ser_vals = ser_flat[valid_idx][mask]
    gidx = valid_idx[mask]
    order = np.argsort(-np.abs(I_vals))[:top_n]

    rows = []
    for i in order:
        r, c = np.unravel_index(gidx[i], shape)
        rows.append({
            "Row": int(r), "Col": int(c),
            "I_i": float(I_vals[i]),
            "p_value": float(p_vals[i]),
            "SER": float(ser_vals[i]),
        })
    return pd.DataFrame(rows)