"""
LISA_SER_analysis_v2.py
========================
Tính Local Moran's I (LISA) cho SER 250 m.
Phiên bản sửa lỗi cho libpysal 2.x.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH
# ============================================================
SER_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\SER_250m.tif"
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\Paper1_Corrected\LISA"
os.makedirs(OUT_DIR, exist_ok=True)

NODATA = -9999
N_PERMUTATIONS = 999
SIGNIFICANCE = 0.05

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ============================================================
# 1. ĐỌC SER
# ============================================================
print("=" * 70)
print("1. ĐỌC SER")
print("=" * 70)

with rasterio.open(SER_FILE) as src:
    ser = src.read(1).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else NODATA
    transform = src.transform
    crs = src.crs
    cell_size = abs(transform[0])
    pixel_ha = (cell_size ** 2) / 10_000

valid_mask = (ser != nodata) & (ser >= 0) & np.isfinite(ser)
valid_idx = np.where(valid_mask.flatten())[0]

print(f"  SER shape: {ser.shape}")
print(f"  Cell size: {cell_size:.2f} m")
print(f"  Valid pixels: {len(valid_idx):,}")
print(f"  Valid area: {len(valid_idx) * pixel_ha:,.0f} ha")


# ============================================================
# 2. TẠO QUEEN CONTIGUITY WEIGHTS (KHÔNG DÙNG SUBSET)
# ============================================================
print("\n" + "=" * 70)
print("2. TẠO WEIGHT MATRIX (QUEEN CONTIGUITY)")
print("=" * 70)

from libpysal.weights import W

# Lấy tọa độ row/col của pixel valid
valid_rows, valid_cols = np.unravel_index(valid_idx, ser.shape)
n = len(valid_idx)

# Mapping từ (row, col) → index trong mảng valid
print(f"  Đang xây dựng mapping cho {n:,} pixel...")
coord_to_idx = {}
for i in range(n):
    coord_to_idx[(int(valid_rows[i]), int(valid_cols[i]))] = i

# Queen neighbors: 8 hướng
print(f"  Đang tìm neighbors...")
neighbors = {}
for i in range(n):
    r = int(valid_rows[i])
    c = int(valid_cols[i])
    neigh_list = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            key = (r + dr, c + dc)
            if key in coord_to_idx:
                neigh_list.append(coord_to_idx[key])
    neighbors[i] = neigh_list

# Tạo W object
w_subset = W(neighbors, silence_warnings=True)
w_subset.transform = 'r'

print(f"  ✅ Weight matrix: {w_subset.n:,} đơn vị")
print(f"  Mean neighbors: {w_subset.mean_neighbors:.2f}")
print(f"  Số pixel isolated: {sum(1 for v in neighbors.values() if len(v) == 0):,}")


# ============================================================
# 3. TÍNH LOCAL MORAN'S I
# ============================================================
print("\n" + "=" * 70)
print("3. TÍNH LOCAL MORAN'S I")
print("=" * 70)

from esda.moran import Moran_Local

ser_valid = ser.flatten()[valid_idx]
ser_std = (ser_valid - np.mean(ser_valid)) / np.std(ser_valid)

print(f"  SER mean = {np.mean(ser_valid):.4f}")
print(f"  SER std  = {np.std(ser_valid):.4f}")
print(f"  Đang chạy LISA với {N_PERMUTATIONS} permutations...")

lisa = Moran_Local(
    ser_std,
    w_subset,
    permutations=N_PERMUTATIONS,
    seed=42,
    keep_simulations=False,
)

n_sig = int(np.sum(lisa.p_sim < SIGNIFICANCE))
print(f"  ✅ Hoàn thành!")
print(f"  Số pixel có ý nghĩa (p < {SIGNIFICANCE}): {n_sig:,} "
      f"({100*n_sig/n:.2f}%)")


# ============================================================
# 4. PHÂN LOẠI CỤM
# ============================================================
print("\n" + "=" * 70)
print("4. PHÂN LOẠI CỤM LISA")
print("=" * 70)

# Quadrant codes:
# 1 = HH, 2 = LH (outlier), 3 = LL, 4 = HL (outlier)
cluster = np.zeros(n, dtype=np.int32)
sig = lisa.p_sim < SIGNIFICANCE
cluster[sig & (lisa.q == 1)] = 1
cluster[sig & (lisa.q == 2)] = 2
cluster[sig & (lisa.q == 3)] = 3
cluster[sig & (lisa.q == 4)] = 4

cluster_labels = {
    0: 'Not significant',
    1: 'High-High (HH)',
    2: 'Low-High (LH) - outlier',
    3: 'Low-Low (LL)',
    4: 'High-Low (HL) - outlier',
}

print(f"\n  {'Code':<6}{'Loại cụm':<25}{'Số pixel':>12}"
      f"{'Diện tích (ha)':>18}{'Tỷ lệ (%)':>12}")
print("  " + "-" * 73)

summary = []
for code in [1, 2, 3, 4, 0]:
    cnt = int(np.sum(cluster == code))
    area = cnt * pixel_ha
    pct = 100 * cnt / n if n > 0 else 0
    summary.append({
        'Code': code,
        'Cluster_type': cluster_labels[code],
        'N_pixels': cnt,
        'Area_ha': round(area, 2),
        'Percent': round(pct, 4),
    })
    print(f"  {code:<6}{cluster_labels[code]:<25}{cnt:>12,}"
          f"{area:>18,.1f}{pct:>12.4f}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, 'LISA_Cluster_Summary.csv'),
                  index=False, encoding='utf-8-sig')


# ============================================================
# 5. LƯU RASTER
# ============================================================
print("\n" + "=" * 70)
print("5. LƯU RASTER LISA")
print("=" * 70)

cluster_raster_flat = np.full(ser.size, NODATA, dtype=np.float32)
cluster_raster_flat[valid_idx] = cluster
cluster_raster = cluster_raster_flat.reshape(ser.shape)

profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': NODATA,
    'width': ser.shape[1],
    'height': ser.shape[0],
    'count': 1,
    'crs': crs,
    'transform': transform,
    'compress': 'LZW',
}
LISA_RASTER = os.path.join(OUT_DIR, 'LISA_Cluster_Map.tif')
with rasterio.open(LISA_RASTER, 'w', **profile) as dst:
    dst.write(cluster_raster, 1)
print(f"  ✅ Đã lưu: {LISA_RASTER}")


# ============================================================
# 6. VẼ HÌNH
# ============================================================
print("\n" + "=" * 70)
print("6. VẼ HÌNH")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- (a) Bản đồ cụm ---
ax = axes[0]
colors_map = ['#D3D3D3', '#C0392B', '#85C1E9', '#2E86AB', '#E67E22']
cmap = ListedColormap(colors_map)
display = np.where(cluster_raster == NODATA, np.nan, cluster_raster)
ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=4.5)

legend_elements = [
    Patch(facecolor='#C0392B', edgecolor='black', label='HH (High-High)'),
    Patch(facecolor='#2E86AB', edgecolor='black', label='LL (Low-Low)'),
    Patch(facecolor='#E67E22', edgecolor='black', label='HL (High-Low)'),
    Patch(facecolor='#85C1E9', edgecolor='black', label='LH (Low-High)'),
    Patch(facecolor='#D3D3D3', edgecolor='black', label='Not significant'),
]
ax.legend(handles=legend_elements, loc='lower right',
          fontsize=9, frameon=True, framealpha=0.95)
ax.set_title('(a) LISA cluster map of SER (250 m)', fontweight='bold')
ax.axis('off')

# --- (b) Bar chart ---
ax = axes[1]
order = [2, 4, 3, 1]  # LH, HL, LL, HH
colors_bar = {'1': '#C0392B', '2': '#85C1E9',
              '3': '#2E86AB', '4': '#E67E22'}

plot_data = summary_df[summary_df['Code'] > 0].copy()
plot_data = plot_data.set_index('Code').loc[order].reset_index()

bars = ax.barh(
    plot_data['Cluster_type'],
    plot_data['Area_ha'],
    color=[colors_bar[str(c)] for c in plot_data['Code']],
    edgecolor='black', linewidth=0.8, height=0.6,
)

max_area = plot_data['Area_ha'].max() if len(plot_data) > 0 else 1
for bar, val, pct in zip(bars, plot_data['Area_ha'], plot_data['Percent']):
    ax.text(bar.get_width() + max_area * 0.02,
            bar.get_y() + bar.get_height()/2,
            f'{val:,.0f} ha\n({pct:.2f}%)',
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Area (ha)', fontweight='bold')
ax.set_title('(b) Area by LISA cluster type', fontweight='bold')
ax.set_xlim(0, max_area * 1.35)
ax.grid(axis='x', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Fig_LISA_SER.png'))
plt.savefig(os.path.join(OUT_DIR, 'Fig_LISA_SER.pdf'))
plt.close()
print(f"  ✅ Đã lưu: Fig_LISA_SER.png / .pdf")


# ============================================================
# 7. BẢNG CHO PAPER
# ============================================================
print("\n" + "=" * 70)
print("7. BẢNG CHO PAPER")
print("=" * 70)

paper_table = summary_df[summary_df['Code'] > 0].copy()
paper_table = paper_table[['Cluster_type', 'N_pixels', 'Area_ha', 'Percent']]
paper_table.columns = ['Cluster type', 'N pixels',
                       'Area (ha)', 'Proportion (%)']
paper_table = paper_table.round(2)

paper_table.to_csv(os.path.join(OUT_DIR, 'Table_LISA_for_paper.csv'),
                   index=False, encoding='utf-8-sig')

print("\n  " + "=" * 70)
print("  Table X. LISA cluster statistics for SER (250 m)")
print("  " + "=" * 70)
print(paper_table.to_string(index=False))
print("  " + "=" * 70)


# ============================================================
# 8. TOP PIXELS THEO |I_i|
# ============================================================
print("\n" + "=" * 70)
print("8. TOP 5 PIXEL THEO |I_i| CHO MỖI LOẠI CỤM")
print("=" * 70)

for code in [1, 2, 3, 4]:
    mask_cl = (cluster == code)
    if not np.any(mask_cl):
        continue

    I_values = lisa.Is[mask_cl]
    p_values = lisa.p_sim[mask_cl]
    ser_values = ser_valid[mask_cl]
    global_idx = valid_idx[mask_cl]

    top5 = np.argsort(-np.abs(I_values))[:5]
    print(f"\n  --- {cluster_labels[code]} ---")
    print(f"  {'Row':>8}{'Col':>8}{'I_i':>12}{'p-value':>12}{'SER':>10}")
    print("  " + "-" * 50)
    for i in top5:
        gi = global_idx[i]
        r, c = np.unravel_index(gi, ser.shape)
        print(f"  {r:>8}{c:>8}{I_values[i]:>12.4f}"
              f"{p_values[i]:>12.4f}{ser_values[i]:>10.2f}")


# ============================================================
# 9. GLOBAL MORAN'S I (THAM KHẢO)
# ============================================================
print("\n" + "=" * 70)
print("9. GLOBAL MORAN'S I (THAM KHẢO)")
print("=" * 70)

from esda.moran import Moran
moran_global = Moran(ser_std, w_subset,
                     permutations=N_PERMUTATIONS, seed=42)
print(f"  Global Moran's I = {moran_global.I:.4f}")
print(f"  E[I] under null  = {moran_global.EI:.4f}")
print(f"  z-score          = {moran_global.z_sim:.2f}")
print(f"  p-value          = {moran_global.p_sim:.4f}")
print(f"\n  ⚠️ Chỉ tham khảo. Không báo cáo trong paper.")


print("\n" + "=" * 70)
print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
print("=" * 70)