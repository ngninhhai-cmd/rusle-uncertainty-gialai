"""
Paper1_fix_numbers.py
======================
Sửa và tính lại toàn bộ số liệu cho Paper 1:
1. K_fixed weighted average từ Bảng 5
2. Resolution sensitivity statistics (mean, SD, median, P95, P99)
3. Uncertainty propagation trên dữ liệu capped LS
4. Proper Sobol/Shapley analysis
5. Kiểm tra diện tích
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH
# ============================================================
BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7"
SER_DIR = os.path.join(BASE, "SER_FINAL")
OUT_DIR = os.path.join(BASE, "Paper1_Corrected")
os.makedirs(OUT_DIR, exist_ok=True)

NODATA = -9999

# ============================================================
# 1. TÍNH LẠI K_FIXED WEIGHTED AVERAGE
# ============================================================
print("=" * 70)
print("1. TÍNH LẠI K_FIXED WEIGHTED AVERAGE")
print("=" * 70)

# Dữ liệu từ Table 5 (đã có trong bài)
K_TABLE = pd.DataFrame({
    'Textural_class': ['Loam', 'Clay Loam', 'Clay'],
    'Area_ha': [67463, 1502581, 2900],
    'K_cont': [0.0408, 0.0297, 0.0217],
    'K_fixed': [0.0448, 0.0329, 0.0217],  # K2 × 0.1317
})

K_TABLE['Weight'] = K_TABLE['Area_ha'] / K_TABLE['Area_ha'].sum()
K_TABLE['K_fixed_weighted'] = K_TABLE['K_fixed'] * K_TABLE['Weight']
K_TABLE['K_cont_weighted'] = K_TABLE['K_cont'] * K_TABLE['Weight']

K_fixed_mean = K_TABLE['K_fixed_weighted'].sum()
K_cont_mean = K_TABLE['K_cont_weighted'].sum()

print(f"\n  K_cont weighted mean  = {K_cont_mean:.4f}")
print(f"  K_fixed weighted mean = {K_fixed_mean:.4f}")
print(f"  Difference            = {K_cont_mean - K_fixed_mean:+.4f}")
print(f"  Reduction (%)         = {100*(K_fixed_mean - K_cont_mean)/K_fixed_mean:.2f}%")
print(f"\n  ⚠️ Giá trị cũ trong Abstract: 0.0358 (SAI)")
print(f"  ✅ Giá trị đúng: {K_fixed_mean:.4f}")
print(f"  ✅ Mức giảm đúng: ~{100*(K_fixed_mean - K_cont_mean)/K_fixed_mean:.0f}% (không phải 16%)")

K_TABLE.to_csv(os.path.join(OUT_DIR, 'K_fixed_recalculated.csv'),
               index=False, encoding='utf-8-sig')


# ============================================================
# 2. TÍNH LẠI RESOLUTION SENSITIVITY STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("2. TÍNH LẠI RESOLUTION SENSITIVITY STATISTICS")
print("=" * 70)

RESOLUTIONS = [30, 90, 120, 150, 200, 250]

res_stats = []
for res in RESOLUTIONS:
    ser_file = os.path.join(SER_DIR, f"SER_{res}m.tif")
    if not os.path.exists(ser_file):
        print(f"  ⚠️ Không tìm thấy: {ser_file}")
        continue

    with rasterio.open(ser_file) as src:
        ser = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA
        ser = np.where(ser == nodata, np.nan, ser)
        ser = np.where(ser < 0, np.nan, ser)
        cell_size = abs(src.transform[0])
        pixel_ha = (cell_size ** 2) / 10_000

    valid = ser[~np.isnan(ser)]

    # Statistics
    stats = {
        'Resolution_m': res,
        'N_pixels': len(valid),
        'Area_ha': len(valid) * pixel_ha,
        'Mean': np.mean(valid),
        'Median': np.median(valid),
        'SD': np.std(valid),
        'CV': np.std(valid) / np.mean(valid),
        'Min': np.min(valid),
        'Max': np.max(valid),
        'P50': np.percentile(valid, 50),
        'P90': np.percentile(valid, 90),
        'P95': np.percentile(valid, 95),
        'P99': np.percentile(valid, 99),
        'P99.9': np.percentile(valid, 99.9),
    }
    res_stats.append(stats)

res_df = pd.DataFrame(res_stats)
res_df.to_csv(os.path.join(OUT_DIR, 'Resolution_Sensitivity_Corrected.csv'),
              index=False, encoding='utf-8-sig')

print("\n  Bảng thống kê SER theo độ phân giải (đã sửa):")
print(res_df[['Resolution_m', 'Mean', 'Median', 'SD', 'P95', 'P99', 'Max']].round(4).to_string(index=False))

# Calculate variation
mean_range = 100 * (res_df['Mean'].max() - res_df['Mean'].min()) / res_df['Mean'].mean()
sd_range = 100 * (res_df['SD'].max() - res_df['SD'].min()) / res_df['SD'].mean()
median_range = 100 * (res_df['Median'].max() - res_df['Median'].min()) / res_df['Median'].mean()

print(f"\n  Mean variation across resolutions:   {mean_range:.2f}%")
print(f"  SD variation across resolutions:     {sd_range:.2f}%")
print(f"  Median variation across resolutions: {median_range:.2f}%")
print(f"\n  ⚠️ Giá trị cũ trong Abstract: 'mean and SD < 4%' (SAI)")
print(f"  ✅ Mean variation đúng:   {mean_range:.1f}%")
print(f"  ✅ SD variation đúng:     {sd_range:.1f}%")
print(f"  ✅ Median variation đúng: {median_range:.1f}%")


# ============================================================
# 3. TÍNH LẠI S_res VÀ S_range (dùng P99 thay max)
# ============================================================
print("\n" + "=" * 70)
print("3. TÍNH LẠI S_res VÀ S_range (dùng P99)")
print("=" * 70)

def get_metric(res_df, res, metric):
    return res_df.loc[res_df['Resolution_m'] == res, metric].values[0]

# S_res: using P99 (more robust than max)
S_res_P99 = 100 * (get_metric(res_df, 250, 'P99') - get_metric(res_df, 30, 'P99')) / get_metric(res_df, 30, 'P99')

# S_range: P99 of 150m vs min P99 of others
p99_150 = get_metric(res_df, 150, 'P99')
p99_others = [get_metric(res_df, r, 'P99') for r in RESOLUTIONS if r != 150]
S_range_P99 = 100 * (p99_150 - min(p99_others)) / min(p99_others)

print(f"\n  S_res (P99, 250m vs 30m)     = {S_res_P99:.2f}%")
print(f"  S_range (P99, 150m vs min)   = {S_range_P99:.2f}%")

# Also with max
S_res_max = 100 * (get_metric(res_df, 250, 'Max') - get_metric(res_df, 30, 'Max')) / get_metric(res_df, 30, 'Max')
max_150 = get_metric(res_df, 150, 'Max')
max_others = [get_metric(res_df, r, 'Max') for r in RESOLUTIONS if r != 150]
S_range_max = 100 * (max_150 - min(max_others)) / min(max_others)

print(f"\n  S_res (max, 250m vs 30m)     = {S_res_max:.2f}%")
print(f"  S_range (max, 150m vs min)   = {S_range_max:.2f}%")
print(f"\n  ⚠️ Lưu ý: S_range với max dùng min at 200m làm mẫu số (cherry-picked)")
print(f"  ✅ Nên báo cáo cả P99 và max")


# ============================================================
# 4. UNCERTAINTY PROPAGATION TRÊN DỮ LIỆU CAPPED LS
# ============================================================
print("\n" + "=" * 70)
print("4. UNCERTAINTY PROPAGATION TRÊN DỮ LIỆU CAPPED")
print("=" * 70)

# Đọc các file cần thiết
FILES = {
    'K_low': os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif"),  # placeholder
    'K_high': os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif"),
    'K_med': os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif"),
    'LS': os.path.join(BASE, "SER_FINAL", "LS_250m_capped.tif"),
    'R': os.path.join(BASE, "SER_FINAL", "metric", "R_30m_utm.tif"),
    'C': os.path.join(BASE, "SER_FINAL", "metric", "C_30m_utm.tif"),
    'SER': os.path.join(BASE, "SER_FINAL", "SER_250m.tif"),
}

# Kiểm tra file nào tồn tại
existing_files = {k: v for k, v in FILES.items() if os.path.exists(v)}
print(f"\n  Files tồn tại: {list(existing_files.keys())}")

# Nếu có đủ K_low, K_high, K_med và SER, tính uncertainty
# (Phần này cần file K_low_1019 và K_high_1019 từ PSU_lan truyen)

K_LOW = r"D:\8_PHD. CANDIDATE\PhD_GIS\RUSLE_CONFERENCE\PSU_lan truyen\K_low_1019.tif"
K_HIGH = r"D:\8_PHD. CANDIDATE\PhD_GIS\RUSLE_CONFERENCE\PSU_lan truyen\K_high_1019.tif"
K_MED = r"D:\8_PHD. CANDIDATE\PhD_GIS\RUSLE_CONFERENCE\PSU_lan truyen\K_median_1019.tif"

if all(os.path.exists(f) for f in [K_LOW, K_HIGH, K_MED]):
    print("\n  ✅ Tìm thấy đủ file K. Tính uncertainty...")

    with rasterio.open(K_LOW) as src:
        k_low = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        k_low = np.where(k_low == nd, np.nan, k_low)
    with rasterio.open(K_HIGH) as src:
        k_high = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        k_high = np.where(k_high == nd, np.nan, k_high)
    with rasterio.open(K_MED) as src:
        k_med = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        k_med = np.where(k_med == nd, np.nan, k_med)

    valid = ~np.isnan(k_low) & ~np.isnan(k_high) & ~np.isnan(k_med)
    print(f"  Số pixel hợp lệ: {np.sum(valid):,}")

    k_low_v = k_low[valid]
    k_high_v = k_high[valid]
    k_med_v = k_med[valid]

    W_K_pixel = (k_low_v - k_high_v) / k_med_v * 100
    W_K_mean = np.mean(W_K_pixel)

    print(f"\n  K_low mean  = {np.mean(k_low_v):.4f}")
    print(f"  K_high mean = {np.mean(k_high_v):.4f}")
    print(f"  K_med mean  = {np.mean(k_med_v):.4f}")
    print(f"  W_K mean (per-pixel) = {W_K_mean:.2f}%")
    print(f"  W_K from domain means = {100*(np.mean(k_low_v)-np.mean(k_high_v))/np.mean(k_med_v):.2f}%")
    print(f"\n  ⚠️ Hai giá trị này khác nhau – cần báo cáo cả hai")


# ============================================================
# 5. PROPER SOBOL ANALYSIS (nếu SALib có sẵn)
# ============================================================
print("\n" + "=" * 70)
print("5. PHÂN TÍCH SOBOL PROPER")
print("=" * 70)

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
    HAS_SALIB = True
    print("  ✅ SALib có sẵn")
except ImportError:
    HAS_SALIB = False
    print("  ⚠️ SALib không có – dùng Monte Carlo thủ công")

# Đọc dữ liệu để lấy bounds
FACTOR_FILES = {
    'R': os.path.join(BASE, "SER_FINAL", "metric", "R_30m_utm.tif"),
    'K': os.path.join(BASE, "SER_FINAL", "metric", "K_250m_utm.tif"),
    'LS': os.path.join(BASE, "SER_FINAL", "LS_250m_capped.tif"),
    'C': os.path.join(BASE, "SER_FINAL", "metric", "C_30m_utm.tif"),
}

# Lấy bounds từ SER_250m extent
with rasterio.open(os.path.join(SER_DIR, "SER_250m.tif")) as ref:
    ref_shape = ref.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

factor_arrays = {}
for name, path in FACTOR_FILES.items():
    if not os.path.exists(path):
        print(f"  ⚠️ Không tìm thấy: {path}")
        continue
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
        arr = np.where(arr == nd, np.nan, arr)
        # Resample if needed
        if src.shape != ref_shape:
            dst = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(source=arr, destination=dst,
                     src_transform=src.transform, src_crs=src.crs,
                     dst_transform=ref_transform, dst_crs=ref_crs,
                     resampling=Resampling.bilinear,
                     src_nodata=np.nan, dst_nodata=np.nan)
            arr = dst
        factor_arrays[name] = arr

# Lấy mẫu pixel hợp lệ
valid_mask = np.ones(ref_shape, dtype=bool)
for arr in factor_arrays.values():
    valid_mask &= ~np.isnan(arr)

n_valid = np.sum(valid_mask)
print(f"  Số pixel hợp lệ: {n_valid:,}")

# Lấy mẫu 50,000 pixel
np.random.seed(42)
sample_size = min(50000, n_valid)
idx = np.random.choice(np.where(valid_mask.flatten())[0], sample_size, replace=False)
rows, cols = np.unravel_index(idx, ref_shape)

df_factors = pd.DataFrame({name: arr[rows, cols] 
                           for name, arr in factor_arrays.items()})

print("\n  Thống kê các yếu tố:")
print(df_factors.describe().round(4).to_string())

# Tính correlation matrix
corr_matrix = df_factors.corr()
print("\n  Ma trận tương quan:")
print(corr_matrix.round(4).to_string())

# Nếu có SALib, chạy Sobol
if HAS_SALIB:
    problem = {
        'num_vars': 4,
        'names': ['R', 'K', 'LS', 'C'],
        'bounds': [[df_factors['R'].min(), df_factors['R'].max()],
                   [df_factors['K'].min(), df_factors['K'].max()],
                   [df_factors['LS'].min(), df_factors['LS'].max()],
                   [df_factors['C'].min(), df_factors['C'].max()]],
    }

    N_SAMPLES = 4096
    param_values = sobol_sample.sample(problem, N_SAMPLES, seed=42)
    Y = param_values[:, 0] * param_values[:, 1] * param_values[:, 2] * param_values[:, 3]

    Si = sobol_analyze.analyze(problem, Y, print_to_console=False, seed=42)

    print(f"\n  📊 Sobol indices (proper):")
    print(f"  {'Factor':<8} {'S1':>10} {'S1_conf':>10} {'ST':>10} {'ST_conf':>10}")
    print("  " + "-" * 50)
    for i, name in enumerate(problem['names']):
        print(f"  {name:<8} {Si['S1'][i]:>10.4f} {Si['S1_conf'][i]:>10.4f} "
              f"{Si['ST'][i]:>10.4f} {Si['ST_conf'][i]:>10.4f}")

    sobol_df = pd.DataFrame({
        'Factor': problem['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
    })
    sobol_df.to_csv(os.path.join(OUT_DIR, 'Sobol_proper.csv'),
                    index=False, encoding='utf-8-sig')
    print(f"\n  ✅ Sum S1 = {Si['S1'].sum():.4f}")

else:
    # Monte Carlo thủ công với correlation
    print("\n  Chạy Monte Carlo với correlation structure...")
    n_mc = 10000
    # Sử dụng Cholesky decomposition để tạo correlated samples
    cov = df_factors.cov().values
    L = np.linalg.cholesky(cov)
    
    np.random.seed(42)
    Z = np.random.randn(n_mc, 4)
    correlated = Z @ L.T + df_factors.mean().values
    
    SER_mc = correlated[:, 0] * correlated[:, 1] * correlated[:, 2] * correlated[:, 3]
    
    print(f"  SER MC: mean={np.mean(SER_mc):.4f}, std={np.std(SER_mc):.4f}")
    print(f"  ⚠️ Cần cài SALib để có Sobol indices chính xác:")
    print(f"     pip install SALib")


# ============================================================
# 6. KIỂM TRA DIỆN TÍCH
# ============================================================
print("\n" + "=" * 70)
print("6. KIỂM TRA DIỆN TÍCH")
print("=" * 70)

areas = {
    'SER_250m': res_df.loc[res_df['Resolution_m']==250, 'Area_ha'].values[0],
    'SER_30m': res_df.loc[res_df['Resolution_m']==30, 'Area_ha'].values[0],
    'K_factor': 1572944,  # từ bài
    'LS_factor': 1552217,
    'LULC': 1553356,
    'Natural_area': 1510800,
}

print("\n  Diện tích theo các nguồn khác nhau:")
for name, area in areas.items():
    print(f"  {name:<15}: {area:>12,.0f} ha")

print(f"\n  ⚠️ Diện tích không nhất quán giữa các lớp:")
print(f"     - Natural area: 1,510,800 ha")
print(f"     - SER valid: {areas['SER_250m']:,.0f} ha ({100*areas['SER_250m']/1510800:.1f}%)")
print(f"     - LULC: {areas['LULC']:,.0f} ha")
print(f"\n  → Cần giải thích rõ trong Methods")


print("\n" + "=" * 70)
print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
print("=" * 70)