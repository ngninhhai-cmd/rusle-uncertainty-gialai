"""
Sobol_proper.py
Tính Sobol indices đúng cách với SALib.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7"
OUT_DIR = os.path.join(BASE, "Paper1_Corrected")
os.makedirs(OUT_DIR, exist_ok=True)
NODATA = -9999

FILES = {
    'R':  os.path.join(BASE, "SER_FINAL", "metric", "R_30m_utm.tif"),
    'K':  os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif"),
    'LS': os.path.join(BASE, "SER_FINAL", "LS_250m_capped.tif"),
    'C':  os.path.join(BASE, "SER_FINAL", "metric", "C_30m_utm.tif"),
}
SER_FILE = os.path.join(BASE, "SER_FINAL", "SER_250m.tif")

# Đọc reference
with rasterio.open(SER_FILE) as ref:
    ref_shape = ref.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

# Đọc và resample các factor
arrays = {}
for name, path in FILES.items():
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        arr = np.where(arr == nd, np.nan, arr)
        if src.shape != ref_shape:
            dst = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(source=arr, destination=dst,
                     src_transform=src.transform, src_crs=src.crs,
                     dst_transform=ref_transform, dst_crs=ref_crs,
                     resampling=Resampling.bilinear,
                     src_nodata=np.nan, dst_nodata=np.nan)
            arr = dst
        arrays[name] = arr

# Mask valid
mask = np.ones(ref_shape, dtype=bool)
for arr in arrays.values():
    mask &= ~np.isnan(arr)

n_valid = np.sum(mask)
print(f"Valid pixels: {n_valid:,}")

# Lấy bounds
bounds = {}
for name, arr in arrays.items():
    v = arr[mask]
    bounds[name] = [float(np.min(v)), float(np.max(v))]
    print(f"  {name}: [{bounds[name][0]:.4f}, {bounds[name][1]:.4f}]")

# Định nghĩa problem
problem = {
    'num_vars': 4,
    'names': ['R', 'K', 'LS', 'C'],
    'bounds': [bounds['R'], bounds['K'], bounds['LS'], bounds['C']],
}

# Sinh mẫu Sobol
N = 4096
print(f"\nSinh {N} base samples...")
param_values = sobol_sample.sample(problem, N, seed=42)
print(f"  Shape: {param_values.shape}")

# Tính SER cho mỗi mẫu
Y = (param_values[:, 0] * param_values[:, 1] *
     param_values[:, 2] * param_values[:, 3])

# Phân tích Sobol
Si = sobol_analyze.analyze(problem, Y, print_to_console=False, seed=42)

print(f"\n{'='*60}")
print("KẾT QUẢ SOBOL PROPER")
print(f"{'='*60}")
print(f"  {'Factor':<8} {'S1':>10} {'S1_conf':>10} {'ST':>10} {'ST_conf':>10}")
print("  " + "-" * 52)
for i, name in enumerate(problem['names']):
    print(f"  {name:<8} {Si['S1'][i]:>10.4f} {Si['S1_conf'][i]:>10.4f} "
          f"{Si['ST'][i]:>10.4f} {Si['ST_conf'][i]:>10.4f}")

print(f"\n  Sum S1 = {Si['S1'].sum():.4f}")
print(f"  Sum ST = {Si['ST'].sum():.4f}")

# Lưu
sobol_df = pd.DataFrame({
    'Factor': problem['names'],
    'S1': Si['S1'],
    'S1_conf': Si['S1_conf'],
    'ST': Si['ST'],
    'ST_conf': Si['ST_conf'],
})
sobol_df.to_csv(os.path.join(OUT_DIR, 'Sobol_proper.csv'),
                index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu: Sobol_proper.csv")