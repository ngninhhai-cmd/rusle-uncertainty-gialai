"""
save_LS_250m.py
Lưu file LS 250m capped từ LS 30m capped (EPSG:3406).
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7"
LS_30M = os.path.join(BASE, "SER_FINAL", "metric", "LS_30m_utm.tif")
REF_250M = os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif")
OUT = os.path.join(BASE, "SER_FINAL", "LS_250m_capped.tif")

NODATA = -9999

# Đọc reference 250m (để lấy extent, transform, shape)
with rasterio.open(REF_250M) as ref:
    ref_profile = ref.profile.copy()
    ref_shape = ref.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

# Đọc LS 30m
with rasterio.open(LS_30M) as src:
    ls = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else NODATA
    ls = np.where(ls == nd, np.nan, ls)

# Resample về 250m
ls_250 = np.full(ref_shape, np.nan, dtype=np.float32)
reproject(
    source=ls,
    destination=ls_250,
    src_transform=src.transform,
    src_crs=src.crs,
    dst_transform=ref_transform,
    dst_crs=ref_crs,
    resampling=Resampling.bilinear,
    src_nodata=np.nan,
    dst_nodata=np.nan,
)

ls_250 = np.where(np.isnan(ls_250), NODATA, ls_250)

# Lưu file
ref_profile.update(dtype=rasterio.float32, nodata=NODATA, compress='LZW')
with rasterio.open(OUT, 'w', **ref_profile) as dst:
    dst.write(ls_250, 1)

valid = ls_250[ls_250 != NODATA]
print(f"✅ Đã lưu: {OUT}")
print(f"   Min  = {np.min(valid):.4f}")
print(f"   Max  = {np.max(valid):.4f}")
print(f"   Mean = {np.mean(valid):.4f}")
print(f"   Std  = {np.std(valid):.4f}")