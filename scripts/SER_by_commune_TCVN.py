"""
SER_by_commune_TCVN.py (FIXED)
================================
1. Phân loại SER 250m theo TCVN 5299:2009
2. Zonal statistics theo 77 xã (cột tenXa)
3. Xuất bảng và biểu đồ
"""

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH
# ============================================================
SER_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\SER_250m.tif"
SHP_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\5_province_boders\DiaPhan_75Xa_TayGL_2025.shp"
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\ByCommune"
os.makedirs(OUT_DIR, exist_ok=True)

NAME_COL = 'tenXa'  # ✅ Cột tên xã chính xác

TCVN = {
    1: (0, 1,      "No erosion (≤ 1)"),
    2: (1, 5,      "Slight (1–5)"),
    3: (5, 10,     "Moderate (5–10)"),
    4: (10, 50,    "Severe (10–50)"),
    5: (50, 1e9,   "Very severe (> 50)"),
}

# ============================================================
# 1. ĐỌC SER VÀ SHAPEFILE
# ============================================================
print("=" * 70)
print("ĐỌC DỮ LIỆU")
print("=" * 70)

with rasterio.open(SER_FILE) as src:
    ser = src.read(1).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else -9999
    ser = np.where(ser == nodata, np.nan, ser)
    ser = np.where(ser < 0, np.nan, ser)
    profile = src.profile.copy()
    crs = src.crs
    transform = src.transform
    pixel_ha = abs(transform[0] * transform[4]) / 10_000

print(f"  SER: shape={ser.shape}, valid={np.sum(~np.isnan(ser)):,}, "
      f"pixel_ha={pixel_ha:.4f}")

gdf = gpd.read_file(SHP_FILE)
print(f"  Shapefile: {len(gdf)} xã | CRS gốc: {gdf.crs}")

if gdf.crs != crs:
    gdf = gdf.to_crs(crs)
    print(f"  Đã reproject về: {crs}")

# Kiểm tra cột tenXa
if NAME_COL not in gdf.columns:
    print(f"  ⚠️ Không tìm thấy cột '{NAME_COL}'. Các cột hiện có:")
    print(f"     {list(gdf.columns)}")
    raise KeyError(f"Cần cột '{NAME_COL}' trong shapefile")
print(f"  ✅ Cột tên xã: {NAME_COL}")
print(f"  Số xã duy nhất: {gdf[NAME_COL].nunique()}")

# ============================================================
# 2. PHÂN LOẠI SER THEO TCVN
# ============================================================
print("\n" + "=" * 70)
print("PHÂN LOẠI SER THEO TCVN 5299:2009")
print("=" * 70)

ser_class = np.full_like(ser, np.nan, dtype=np.float32)
valid = ~np.isnan(ser)

for cls, (lo, hi, label) in TCVN.items():
    if lo > 0:
        mask_cls = valid & (ser > lo) & (ser <= hi)
    else:
        mask_cls = valid & (ser <= hi)
    ser_class[mask_cls] = cls

total_valid = int(np.sum(valid))
print(f"\n  Tổng pixel hợp lệ: {total_valid:,}")
print(f"  Tổng diện tích: {total_valid * pixel_ha:,.1f} ha")

print(f"\n  {'Cấp':<6}{'Ngưỡng':<22}{'Số pixel':>14}{'Diện tích (ha)':>18}{'Tỷ lệ (%)':>12}")
print("  " + "-" * 72)

provincial = []
for cls, (lo, hi, label) in TCVN.items():
    n = int(np.sum(ser_class == cls))
    area = n * pixel_ha
    pct = 100 * n / total_valid if total_valid > 0 else 0
    provincial.append({
        'Class': cls, 'Level': label, 'N_pixel': n,
        'Area_ha': round(area, 1), 'Percent': round(pct, 2)
    })
    print(f"  {cls:<6}{label:<22}{n:>14,}{area:>18,.1f}{pct:>12.2f}")

prov_df = pd.DataFrame(provincial)
prov_df.to_csv(os.path.join(OUT_DIR, 'TCVN_Provincial_Statistics.csv'),
               index=False, encoding='utf-8-sig')

# ============================================================
# 3. ZONAL STATISTICS THEO XÃ (đã sửa lỗi Dataset closed)
# ============================================================
print("\n" + "=" * 70)
print("ZONAL STATISTICS THEO 77 XÃ")
print("=" * 70)

results = []
errors = []

# ✅ Mở file raster MỘT LẦN, giữ mở suốt vòng lặp
with rasterio.open(SER_FILE) as src:
    for idx, row in gdf.iterrows():
        geom = [mapping(row['geometry'])]
        try:
            out_image, out_transform = mask(src, geom, crop=True, nodata=np.nan)
            out = out_image[0]
            valid_out = out[~np.isnan(out)]
            if valid_out.size == 0:
                continue

            total_pix = valid_out.size
            total_area = total_pix * pixel_ha
            row_result = {
                'Commune': row[NAME_COL],
                'N_pixel': total_pix,
                'Area_ha': round(total_area, 1),
                'SER_mean': round(float(np.mean(valid_out)), 4),
                'SER_median': round(float(np.median(valid_out)), 4),
                'SER_max': round(float(np.max(valid_out)), 4),
            }

            for cls, (lo, hi, label) in TCVN.items():
                if lo > 0:
                    n_cls = int(np.sum((valid_out > lo) & (valid_out <= hi)))
                else:
                    n_cls = int(np.sum(valid_out <= hi))
                row_result[f'Class{cls}_ha'] = round(n_cls * pixel_ha, 1)
                row_result[f'Class{cls}_pct'] = round(
                    100 * n_cls / total_pix if total_pix > 0 else 0, 2)

            results.append(row_result)

        except Exception as e:
            errors.append({'Commune': row[NAME_COL], 'Error': str(e)})
            continue

print(f"\n  ✅ Xử lý thành công: {len(results)} xã")
if errors:
    print(f"  ⚠️ Lỗi: {len(errors)} xã")
    for e in errors[:5]:
        print(f"     - {e['Commune']}: {e['Error'][:80]}")

commune_df = pd.DataFrame(results)
if len(commune_df) > 0:
    commune_df = commune_df.sort_values('SER_mean', ascending=False)

    commune_df.to_csv(
        os.path.join(OUT_DIR, 'TCVN_ByCommune_Statistics.csv'),
        index=False, encoding='utf-8-sig')

    print(f"\n  Top 10 xã có SER trung bình cao nhất:")
    print(commune_df[['Commune', 'SER_mean', 'SER_max',
                      'Class4_pct', 'Class5_pct']].head(10).to_string(index=False))
else:
    print("  ❌ Không có xã nào xử lý thành công!")

# ============================================================
# 4. XUẤT RASTER PHÂN LOẠI
# ============================================================
profile.update(dtype=rasterio.float32, nodata=-9999, compress='LZW')
CLS_FILE = os.path.join(OUT_DIR, 'SER_Class_TCVN_250m.tif')
with rasterio.open(CLS_FILE, 'w', **profile) as dst:
    dst.write(np.where(np.isnan(ser_class), -9999, ser_class).astype(np.float32), 1)
print(f"\n  ✅ Đã lưu: {CLS_FILE}")

# ============================================================
# 5. BIỂU ĐỒ
# ============================================================
if len(commune_df) > 0:
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    fig = plt.figure(figsize=(16, 10))
    colors = ['#2E7D32', '#FDD835', '#FB8C00', '#E53935', '#B71C1C']
    labels = [f'Class {i}\n{TCVN[i][2]}' for i in [1, 2, 3, 4, 5]]
    pcts = prov_df['Percent'].values

    # (a) Bar chart toàn tỉnh
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(labels, pcts, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax1.set_ylabel('Area (%)', fontweight='bold')
    ax1.set_title('(a) Provincial distribution by TCVN class',
                  fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', labelsize=9)

    # (b) Pie chart
    ax2 = fig.add_subplot(2, 2, 2)
    wedges, texts, autotexts = ax2.pie(
        pcts, labels=None, colors=colors, autopct='%1.2f%%',
        startangle=90, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax2.set_title('(b) Proportional distribution', fontweight='bold')
    ax2.legend(wedges, labels, loc='center left',
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

    # (c) Top 15 xã SER mean
    ax3 = fig.add_subplot(2, 2, 3)
    top15 = commune_df.head(15)
    bars = ax3.barh(top15['Commune'][::-1], top15['SER_mean'][::-1],
                    color='#C0392B', edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, top15['SER_mean'][::-1]):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Mean SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax3.set_title('(c) Top 15 communes by mean SER', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.tick_params(axis='y', labelsize=8)

    # (d) Stacked bar top 10 Class 4+5
    ax4 = fig.add_subplot(2, 2, 4)
    commune_df['Class45_pct'] = commune_df['Class4_pct'] + commune_df['Class5_pct']
    top10 = commune_df.sort_values('Class45_pct', ascending=False).head(10)

    x_pos = np.arange(len(top10))
    c4 = top10['Class4_pct'].values
    c5 = top10['Class5_pct'].values

    ax4.bar(x_pos, c4, 0.6, label='Class 4 (Severe)',
            color='#E53935', edgecolor='black')
    ax4.bar(x_pos, c5, 0.6, bottom=c4, label='Class 5 (Very severe)',
            color='#B71C1C', edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(top10['Commune'], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Area (%)', fontweight='bold')
    ax4.set_title('(d) Top 10 communes by Class 4+5 share', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig_SER_TCVN_Statistics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Fig_SER_TCVN_Statistics.png")

# ============================================================
# 6. KIỂM TRA VÙNG SER CAO
# ============================================================
print("\n" + "=" * 70)
print("KIỂM TRA VÙNG SER CAO (Class 4+5)")
print("=" * 70)

from rasterio.warp import reproject, Resampling

FACTORS = {
    'LS': r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\LS_250m_capped.tif",
    'R':  r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\metric\R_30m_utm.tif",
    'C':  r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\metric\C_30m_utm.tif",
    'K':  r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\metric\K_250m_utm.tif",
}

with rasterio.open(SER_FILE) as ref:
    ref_shape = ref.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

factor_arrays = {}
for name, path in FACTORS.items():
    if not os.path.exists(path):
        print(f"  ⚠️ Không tìm thấy: {path}")
        continue
    with rasterio.open(path) as src:
        if src.shape == ref_shape and src.transform == ref_transform:
            factor_arrays[name] = src.read(1).astype(np.float32)
        else:
            dst = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=src.read(1).astype(np.float32),
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata, dst_nodata=np.nan,
            )
            factor_arrays[name] = dst

high_mask = (ser_class == 4) | (ser_class == 5)
low_mask  = (ser_class == 1) | (ser_class == 2)

print(f"\n  So sánh factor giữa vùng SER cao (Class 4+5) và thấp (Class 1+2):")
print(f"\n  {'Factor':<8}{'High SER':>15}{'Low SER':>15}{'Difference':>15}")
print("  " + "-" * 55)

for name, arr in factor_arrays.items():
    high_vals = arr[high_mask & ~np.isnan(arr)]
    low_vals  = arr[low_mask & ~np.isnan(arr)]
    if high_vals.size > 0 and low_vals.size > 0:
        print(f"  {name:<8}{np.mean(high_vals):>15.4f}"
              f"{np.mean(low_vals):>15.4f}"
              f"{np.mean(high_vals) - np.mean(low_vals):>+15.4f}")

print("\n" + "=" * 70)
print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
print("=" * 70)