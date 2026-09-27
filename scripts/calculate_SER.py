"""
=====================================================================
RUSLE HOÀN CHỈNH – P = 1, LS CAP, MULTI-RESOLUTION
=====================================================================
Xử lý triệt để 3 vấn đề:
  1. P = 1 (đúng mô tả bài báo, không dùng P_adjusted)
  2. LS được cap ở percentile 99 để loại bỏ outlier do flow accumulation
  3. Tính SER chính ở 250m + so sánh 6 độ phân giải (30–250m)
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# =====================================================================
# CẤU HÌNH
# =====================================================================
BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7"
OUT_DIR = os.path.join(BASE, "SER_FINAL")
os.makedirs(OUT_DIR, exist_ok=True)

# File nguồn 30m
SRC_30M = {
    'R' : os.path.join(BASE, "2_R_factor_30m.tif"),
    'LS': os.path.join(BASE, "3_LS_factor_30m.tif"),
    'C' : os.path.join(BASE, "4_C_factor_30m.tif"),
}
# K đã hiệu chỉnh (hệ số 0.1317)
K_FILE = os.path.join(BASE, "6_K_factor_250m.tif")

NODATA = -9999
TARGET_CRS = 'EPSG:3406'  # VN-2000 / UTM 49N (mét)
LS_CAP_PERCENTILE = 99
RESOLUTIONS = [30, 90, 120, 150, 200, 250]


# =====================================================================
# BƯỚC 1: ĐỌC VÀ XỬ LÝ LS – CAP OUTLIER
# =====================================================================
print("=" * 70)
print("BƯỚC 1: ĐỌC VÀ CAP LS")
print("=" * 70)

with rasterio.open(SRC_30M['LS']) as src:
    ls_raw = src.read(1).astype(np.float32)
    ls_meta = src.meta.copy()
    ls_crs = src.crs
    ls_transform = src.transform
    nd = src.nodata if src.nodata is not None else NODATA
    ls_raw = np.where(ls_raw == nd, np.nan, ls_raw)
    ls_raw = np.where(ls_raw < -1e30, np.nan, ls_raw)

valid_ls = ls_raw[~np.isnan(ls_raw)]
ls_min = float(np.min(valid_ls))
ls_max_raw = float(np.max(valid_ls))
ls_mean_raw = float(np.mean(valid_ls))
ls_cap_value = float(np.percentile(valid_ls, LS_CAP_PERCENTILE))

print(f"  LS gốc  : min={ls_min:.4f} | max={ls_max_raw:.4f} | mean={ls_mean_raw:.4f}")
print(f"  Cap tại P{LS_CAP_PERCENTILE} = {ls_cap_value:.4f}")

# Cap LS
ls_capped = np.where(ls_raw > ls_cap_value, ls_cap_value, ls_raw)
valid_capped = ls_capped[~np.isnan(ls_capped)]
print(f"  LS cap  : min={np.min(valid_capped):.4f} | max={np.max(valid_capped):.4f} | "
      f"mean={np.mean(valid_capped):.4f}")
n_capped = int(np.sum(ls_raw > ls_cap_value))
print(f"  Số pixel bị cap: {n_capped:,} ({n_capped/valid_ls.size*100:.4f}%)")

# Lưu LS đã cap
LS_CAPPED_FILE = os.path.join(OUT_DIR, "3_LS_factor_30m_capped.tif")
ls_meta.update(dtype=rasterio.float32, nodata=NODATA, compress='LZW')
with rasterio.open(LS_CAPPED_FILE, 'w', **ls_meta) as dst:
    dst.write(np.where(np.isnan(ls_capped), NODATA, ls_capped).astype(np.float32), 1)
print(f"  ✅ Đã lưu: {LS_CAPPED_FILE}\n")


# =====================================================================
# BƯỚC 2: REPROJECT TẤT CẢ VỀ EPSG:3406 (MÉT)
# =====================================================================
print("=" * 70)
print("BƯỚC 2: REPROJECT VỀ EPSG:3406 (MÉT)")
print("=" * 70)

METRIC_DIR = os.path.join(OUT_DIR, "metric")
os.makedirs(METRIC_DIR, exist_ok=True)


def reproject_to_metric(src_file, dst_file, res=30, method=Resampling.bilinear):
    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
            resolution=res
        )
        profile = src.profile.copy()
        profile.update({
            'crs': TARGET_CRS, 'transform': transform,
            'width': width, 'height': height,
            'dtype': 'float32', 'nodata': NODATA,
            'compress': 'LZW',
        })
        dst_arr = np.full((height, width), np.nan, dtype=np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        src_arr = src.read(1).astype(np.float32)
        src_arr = np.where(src_arr == nd, np.nan, src_arr)

        reproject(
            source=src_arr, destination=dst_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=TARGET_CRS,
            resampling=method, src_nodata=np.nan, dst_nodata=np.nan,
        )
        with rasterio.open(dst_file, 'w', **profile) as dst:
            dst.write(np.where(np.isnan(dst_arr), NODATA, dst_arr).astype(np.float32), 1)
    return dst_file


metric_files = {}
for name, src in SRC_30M.items():
    if name == 'LS':
        src = LS_CAPPED_FILE
    out = os.path.join(METRIC_DIR, f'{name}_30m_utm.tif')
    method = Resampling.nearest if name == 'C' else Resampling.bilinear
    print(f"  Reproject {name} ...")
    reproject_to_metric(src, out, res=30, method=method)
    metric_files[name] = out

# K → 250m UTM
k_out = os.path.join(METRIC_DIR, 'K_250m_utm.tif')
print(f"  Reproject K ...")
reproject_to_metric(K_FILE, k_out, res=250, method=Resampling.bilinear)
metric_files['K'] = k_out

# Lấy reference bounds từ K (250m)
with rasterio.open(metric_files['K']) as src:
    ref_bounds = src.bounds
    ref_crs = src.crs
print(f"\n  Reference bounds: {ref_bounds}")
print(f"  Reference CRS   : {ref_crs}\n")


# =====================================================================
# BƯỚC 3: HÀM TÍNH SER CHO MỘT ĐỘ PHÂN GIẢI
# =====================================================================
def build_profile(target_res, bounds, crs):
    width = int(np.ceil((bounds.right - bounds.left) / target_res))
    height = int(np.ceil((bounds.top - bounds.bottom) / target_res))
    transform = from_bounds(bounds.left, bounds.bottom,
                             bounds.right, bounds.top, width, height)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': NODATA,
        'width': width, 'height': height, 'count': 1,
        'crs': crs, 'transform': transform, 'compress': 'LZW',
    }
    return profile, (height, width)


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        arr = np.where(arr == nd, np.nan, arr)
        arr = np.where(arr < -1e30, np.nan, arr)
        return arr, src.transform, src.crs


def resample_to_profile(src_arr, src_transform, src_crs, profile, shape,
                         method=Resampling.bilinear):
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=profile['transform'], dst_crs=profile['crs'],
        resampling=method, src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst


def compute_SER(target_res):
    print("=" * 70)
    print(f"TÍNH SER Ở {target_res} m")
    print("=" * 70)

    profile, shape = build_profile(target_res, ref_bounds, ref_crs)
    print(f"  Shape: {shape[0]}×{shape[1]}")

    factors = {}

    # R, C: resample từ 30m
    for name in ['R', 'C']:
        arr, tr, crs = read_raster(metric_files[name])
        method = Resampling.nearest if name == 'C' else Resampling.bilinear
        factors[name] = resample_to_profile(arr, tr, crs, profile, shape, method)

    # LS: resample từ 30m đã cap
    arr, tr, crs = read_raster(metric_files['LS'])
    factors['LS'] = resample_to_profile(arr, tr, crs, profile, shape,
                                         Resampling.bilinear)

    # K: resample từ 250m
    arr, tr, crs = read_raster(metric_files['K'])
    factors['K'] = resample_to_profile(arr, tr, crs, profile, shape,
                                        Resampling.bilinear)

    # P = 1 (hằng số) – đúng mô tả bài báo
    factors['P'] = np.ones(shape, dtype=np.float32)

    # In thống kê
    for name in ['R', 'K', 'LS', 'C']:
        v = factors[name][~np.isnan(factors[name])]
        print(f"  {name:3s}: min={np.min(v):.4f} | max={np.max(v):.4f} | "
              f"mean={np.mean(v):.4f}")

    # Mask hợp lệ
    mask = np.ones(shape, dtype=bool)
    for name in ['R', 'K', 'LS', 'C']:
        mask &= ~np.isnan(factors[name])

    n_valid = int(np.sum(mask))
    print(f"  Pixel hợp lệ: {n_valid:,} / {mask.size:,} "
          f"({n_valid/mask.size*100:.2f}%)")

    # Tính SER = R × K × LS × C × 1
    SER = np.full(shape, NODATA, dtype=np.float32)
    if n_valid > 0:
        SER[mask] = (factors['R'][mask] * factors['K'][mask] *
                     factors['LS'][mask] * factors['C'][mask] * 1.0)

    valid_ser = SER[SER != NODATA]
    if valid_ser.size == 0:
        print("  ⚠️ Không có pixel SER hợp lệ!\n")
        return None

    stats = {
        'Resolution_m': target_res,
        'Shape': f"{shape[0]}x{shape[1]}",
        'N_valid': n_valid,
        'Valid_pct': round(n_valid / mask.size * 100, 2),
        'Min': round(float(np.min(valid_ser)), 4),
        'Max': round(float(np.max(valid_ser)), 4),
        'Mean': round(float(np.mean(valid_ser)), 4),
        'Median': round(float(np.median(valid_ser)), 4),
        'StdDev': round(float(np.std(valid_ser)), 4),
    }
    print(f"  SER: min={stats['Min']:.4f} | max={stats['Max']:.4f} | "
          f"mean={stats['Mean']:.4f} | median={stats['Median']:.4f} | "
          f"std={stats['StdDev']:.4f}")

    # Lưu file
    out_file = os.path.join(OUT_DIR, f"SER_{target_res}m.tif")
    profile.update(dtype=rasterio.float32, nodata=NODATA)
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(SER, 1)
    print(f"  ✅ Đã lưu: {out_file}\n")
    return stats


# =====================================================================
# BƯỚC 4: TÍNH SER CHO TẤT CẢ ĐỘ PHÂN GIẢI
# =====================================================================
print("=" * 70)
print("BƯỚC 3: TÍNH SER ĐA ĐỘ PHÂN GIẢI")
print("=" * 70 + "\n")

all_stats = []
for res in RESOLUTIONS:
    stats = compute_SER(res)
    if stats is not None:
        all_stats.append(stats)

df = pd.DataFrame(all_stats)

# Lưu bảng
csv_file = os.path.join(OUT_DIR, "Table_SER_MultiRes.csv")
xlsx_file = os.path.join(OUT_DIR, "Table_SER_MultiRes.xlsx")
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
df.to_excel(xlsx_file, index=False, engine='openpyxl')

print("=" * 70)
print("BẢNG SER THEO ĐỘ PHÂN GIẢI (P = 1, LS CAP P99)")
print("=" * 70)
print(df.to_string(index=False))
print()


# =====================================================================
# BƯỚC 5: CHỈ SỐ ĐỘ NHẠY
# =====================================================================
print("=" * 70)
print("CHỈ SỐ ĐỘ NHẠY ĐỘ PHÂN GIẢI")
print("=" * 70)

max_30  = df.loc[df['Resolution_m'] == 30, 'Max'].values[0]
max_150 = df.loc[df['Resolution_m'] == 150, 'Max'].values[0]
max_250 = df.loc[df['Resolution_m'] == 250, 'Max'].values[0]

other_max = df.loc[df['Resolution_m'] != 150, 'Max']
min_other = other_max.min()

S_res   = (max_250 - max_30) / max_30 * 100
S_range = (max_150 - min_other) / min_other * 100

print(f"  SER max @ 30m  = {max_30:.4f}")
print(f"  SER max @ 150m = {max_150:.4f}")
print(f"  SER max @ 250m = {max_250:.4f}")
print(f"  Min other (≠150m) = {min_other:.4f}")
print(f"  S_res   = {S_res:.2f}%")
print(f"  S_range = {S_range:.2f}%\n")

summary_sens = pd.DataFrame([{
    'SER_max_30m': max_30,
    'SER_max_150m': max_150,
    'SER_max_250m': max_250,
    'Min_other': min_other,
    'S_res_pct': round(S_res, 2),
    'S_range_pct': round(S_range, 2),
}])
summary_sens.to_csv(os.path.join(OUT_DIR, "Sensitivity_Indices.csv"),
                     index=False, encoding='utf-8-sig')


# =====================================================================
# BƯỚC 6: BIỂU ĐỒ
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
res = df['Resolution_m'].values

# (a) Mean
axes[0].plot(res, df['Mean'], 'o-', color='#2E86AB', lw=2, ms=8)
axes[0].set_xlabel('Độ phân giải (m)', fontweight='bold')
axes[0].set_ylabel('SER trung bình (t.ha⁻¹.năm⁻¹)', fontweight='bold')
axes[0].set_title('(a) SER trung bình', fontweight='bold')
axes[0].grid(alpha=0.3)
for x, y in zip(res, df['Mean']):
    axes[0].annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=9)

# (b) StdDev
axes[1].plot(res, df['StdDev'], 's-', color='#E67E22', lw=2, ms=8)
axes[1].set_xlabel('Độ phân giải (m)', fontweight='bold')
axes[1].set_ylabel('Độ lệch chuẩn', fontweight='bold')
axes[1].set_title('(b) Độ lệch chuẩn SER', fontweight='bold')
axes[1].grid(alpha=0.3)
for x, y in zip(res, df['StdDev']):
    axes[1].annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=9)

# (c) Max
axes[2].plot(res, df['Max'], '^-', color='#C0392B', lw=2, ms=8)
axes[2].set_xlabel('Độ phân giải (m)', fontweight='bold')
axes[2].set_ylabel('SER tối đa (t.ha⁻¹.năm⁻¹)', fontweight='bold')
axes[2].set_title('(c) SER tối đa', fontweight='bold')
axes[2].grid(alpha=0.3)
for x, y in zip(res, df['Max']):
    axes[2].annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_SER_MultiRes.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Fig_SER_MultiRes.png")

# Biểu đồ sensitivity indices
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(['S_res\n(250m vs 30m)', 'S_range\n(150m vs min)'],
              [S_res, S_range],
              color=['#C0392B', '#E67E22'],
              edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, [S_res, S_range]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.2f}%', ha='center', va='bottom',
            fontsize=13, fontweight='bold')
ax.set_ylabel('Chỉ số độ nhạy (%)', fontweight='bold')
ax.set_title('Chỉ số độ nhạy độ phân giải', fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_Sensitivity_Indices.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Fig_Sensitivity_Indices.png\n")

print("=" * 70)
print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
print("=" * 70)