"""
SER_by_K_and_C.py
Tính phân bố SER theo cấp TCVN × nhóm đất (K) và nhóm sử dụng đất (C).
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7"
OUT_DIR = os.path.join(BASE, "Paper1_Corrected")
os.makedirs(OUT_DIR, exist_ok=True)
NODATA = -9999

# Files
SER_FILE = os.path.join(BASE, "SER_FINAL", "SER_250m.tif")
K_FILE = os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif")  # K_median
C_FILE = os.path.join(BASE, "SER_FINAL", "metric", "C_30m_utm.tif")

# Đọc SER
with rasterio.open(SER_FILE) as src:
    ser = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else NODATA
    ser = np.where(ser == nd, np.nan, ser)
    ref_profile = src.profile.copy()
    ref_shape = ser.shape
    ref_transform = src.transform
    ref_crs = src.crs
    pixel_ha = (abs(ref_transform[0]) ** 2) / 10_000

print(f"SER: shape={ref_shape}, valid={np.sum(~np.isnan(ser)):,}, "
      f"pixel_ha={pixel_ha:.4f}")

# Đọc K (resample nếu cần)
with rasterio.open(K_FILE) as src:
    k = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else NODATA
    k = np.where(k == nd, np.nan, k)
    if src.shape != ref_shape:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(source=k, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        k = dst

# Đọc C (resample nếu cần, dùng nearest)
with rasterio.open(C_FILE) as src:
    c = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else NODATA
    c = np.where(c == nd, np.nan, c)
    if src.shape != ref_shape:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(source=c, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.nearest,
                  src_nodata=np.nan, dst_nodata=np.nan)
        c = dst

# Phân loại TCVN 5299:2009
def classify_TCVN(ser):
    cls = np.full(ser.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(ser)
    cls[valid & (ser <= 1)] = 1
    cls[valid & (ser > 1) & (ser <= 5)] = 2
    cls[valid & (ser > 5) & (ser <= 10)] = 3
    cls[valid & (ser > 10) & (ser <= 50)] = 4
    cls[valid & (ser > 50)] = 5
    return cls

ser_class = classify_TCVN(ser)

# Phân loại K theo 3 nhóm đất (dựa trên K_mean đã biết)
# Thực tế cần đọc texture_class_int để biết nhóm đất
# Tạm phân loại theo giá trị K
def classify_K(k):
    """Phân loại theo dải K: Loam (cao), Clay Loam (trung), Clay (thấp)."""
    cls = np.full(k.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(k)
    cls[valid & (k < 0.0250)] = 3     # Clay
    cls[valid & (k >= 0.0250) & (k < 0.0350)] = 7  # Clay Loam
    cls[valid & (k >= 0.0350)] = 4     # Loam
    return cls

k_class = classify_K(k)

# Phân loại C
def classify_C(c):
    cls = np.full(c.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(c)
    cls[valid & (c == 0)] = 1       # Water
    cls[valid & (c == 0.005)] = 2   # Forest
    cls[valid & (c == 0.02)] = 3    # Built-up
    cls[valid & (c == 0.30)] = 4    # Agriculture
    cls[valid & (c == 0.80)] = 5    # Other
    return cls

c_class = classify_C(c)

# Mask valid trên tất cả các lớp
valid_all = ~np.isnan(ser_class) & ~np.isnan(k_class) & ~np.isnan(c_class)
print(f"Valid all: {np.sum(valid_all):,}")

# ============================================================
# Bảng chéo 1: SER_class × K_class
# ============================================================
print("\n" + "=" * 70)
print("SER TCVN × K (nhóm đất)")
print("=" * 70)

K_NAMES = {4: "Loam", 7: "Clay Loam", 3: "Clay"}
TCVN_NAMES = {1: "I (≤1)", 2: "II (1–5)", 3: "III (5–10)",
              4: "IV (10–50)", 5: "V (>50)"}

rows = []
for tcvn in [1, 2, 3, 4, 5]:
    row = {"TCVN": TCVN_NAMES[tcvn]}
    total = 0
    for kc in [4, 7, 3]:
        n = int(np.sum(valid_all & (ser_class == tcvn) & (k_class == kc)))
        area = n * pixel_ha
        row[K_NAMES[kc]] = area
        total += area
    row["Total"] = total
    rows.append(row)

df_k = pd.DataFrame(rows)
print(df_k.to_string(index=False))
df_k.to_csv(os.path.join(OUT_DIR, "SER_TCVN_by_K.csv"),
            index=False, encoding="utf-8-sig")

# ============================================================
# Bảng chéo 2: SER_class × C_class
# ============================================================
print("\n" + "=" * 70)
print("SER TCVN × C (nhóm sử dụng đất)")
print("=" * 70)

C_NAMES = {1: "Water", 2: "Forest", 3: "Built-up",
           4: "Agriculture", 5: "Other"}

rows = []
for tcvn in [1, 2, 3, 4, 5]:
    row = {"TCVN": TCVN_NAMES[tcvn]}
    total = 0
    for cc in [1, 2, 3, 4, 5]:
        n = int(np.sum(valid_all & (ser_class == tcvn) & (c_class == cc)))
        area = n * pixel_ha
        row[C_NAMES[cc]] = area
        total += area
    row["Total"] = total
    rows.append(row)

df_c = pd.DataFrame(rows)
print(df_c.to_string(index=False))
df_c.to_csv(os.path.join(OUT_DIR, "SER_TCVN_by_C.csv"),
            index=False, encoding="utf-8-sig")

# ============================================================
# Bảng 3: SER_mean và SER_max theo K và C
# ============================================================
print("\n" + "=" * 70)
print("SER statistics by K class")
print("=" * 70)

rows = []
for kc in [4, 7, 3]:
    mask = valid_all & (k_class == kc)
    if not np.any(mask):
        continue
    ser_v = ser[mask]
    rows.append({
        "K class": K_NAMES[kc],
        "N pixels": int(np.sum(mask)),
        "Area (ha)": np.sum(mask) * pixel_ha,
        "SER_mean": float(np.mean(ser_v)),
        "SER_median": float(np.median(ser_v)),
        "SER_max": float(np.max(ser_v)),
    })

df_k_stats = pd.DataFrame(rows)
print(df_k_stats.to_string(index=False))
df_k_stats.to_csv(os.path.join(OUT_DIR, "SER_stats_by_K.csv"),
                  index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("SER statistics by C class")
print("=" * 70)

rows = []
for cc in [1, 2, 3, 4, 5]:
    mask = valid_all & (c_class == cc)
    if not np.any(mask):
        continue
    ser_v = ser[mask]
    rows.append({
        "C class": C_NAMES[cc],
        "N pixels": int(np.sum(mask)),
        "Area (ha)": np.sum(mask) * pixel_ha,
        "SER_mean": float(np.mean(ser_v)),
        "SER_median": float(np.median(ser_v)),
        "SER_max": float(np.max(ser_v)),
    })

df_c_stats = pd.DataFrame(rows)
print(df_c_stats.to_string(index=False))
df_c_stats.to_csv(os.path.join(OUT_DIR, "SER_stats_by_C.csv"),
                  index=False, encoding="utf-8-sig")

print(f"\n✅ Đã lưu kết quả tại: {OUT_DIR}")