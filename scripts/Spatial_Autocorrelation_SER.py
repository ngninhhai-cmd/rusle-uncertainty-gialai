"""
Spatial_Autocorrelation_SER.py
================================
Tính Moran's I, Geary's C (global) và LISA (local) cho SER 250 m.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from esda.geary import Geary   # ← SỬA DÒNG NÀY

OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\SpatialStats"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. ĐỌC SER 250 M
# ============================================================
SER_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\SER_250m.tif"

with rasterio.open(SER_FILE) as src:
    ser = src.read(1).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else -9999
    ser = np.where(ser == nodata, np.nan, ser)

# Lấy mẫu 50.000 pixel
np.random.seed(42)
valid_mask = ~np.isnan(ser)
rows, cols = np.where(valid_mask)
n_sample = min(50000, len(rows))
idx = np.random.choice(len(rows), n_sample, replace=False)

sample_rows = rows[idx]
sample_cols = cols[idx]
sample_vals = ser[sample_rows, sample_cols]

print(f"Số pixel mẫu: {len(sample_vals):,}")

df = pd.DataFrame({
    'row': sample_rows,
    'col': sample_cols,
    'SER': sample_vals,
})

# ============================================================
# 2. SPATIAL WEIGHTS
# ============================================================
coords = df[['row', 'col']].values
w = KNN.from_array(coords, k=8)
w.transform = 'r'
print(f"Trọng số không gian: {w.n} đơn vị")

# ============================================================
# 3. GLOBAL MORAN'S I
# ============================================================
moran = Moran(df['SER'].values, w)
print(f"\n📊 GLOBAL MORAN'S I")
print(f"   I      = {moran.I:.4f}")
print(f"   E[I]   = {moran.EI:.4f}")
print(f"   z-score= {moran.z_sim:.4f}")
print(f"   p-value= {moran.p_sim:.4f}")

# ============================================================
# 4. GLOBAL GEARY'S C
# ============================================================
geary = Geary(df['SER'].values, w)
print(f"\n📊 GLOBAL GEARY'S C")
print(f"   C      = {geary.C:.4f}")
print(f"   E[C]   = {geary.EC:.4f}")
print(f"   z-score= {geary.z_sim:.4f}")
print(f"   p-value= {geary.p_sim:.4f}")

# ============================================================
# 5. LOCAL MORAN'S I (LISA)
# ============================================================
lisa = Moran_Local(df['SER'].values, w, permutations=999)

quadrant_labels = {1: 'HH (Hot spot)', 2: 'LH', 3: 'LL (Cold spot)', 4: 'HL'}
sig = lisa.p_sim < 0.05
print(f"\n📊 LISA (Local Moran's I)")
print(f"   Số pixel có ý nghĩa (p < 0.05): {sig.sum():,}")
for q in [1, 2, 3, 4]:
    n_q = ((lisa.q == q) & sig).sum()
    pct = 100 * n_q / sig.sum() if sig.sum() > 0 else 0
    print(f"   {quadrant_labels[q]:20s}: {n_q:6,} ({pct:5.1f}%)")

df['LISA_I'] = lisa.Is
df['LISA_q'] = lisa.q
df['LISA_p'] = lisa.p_sim
df['LISA_sig'] = sig
df.to_csv(os.path.join(OUT_DIR, 'LISA_results.csv'),
          index=False, encoding='utf-8-sig')

# ============================================================
# 6. MORAN'S I THEO CÁC ĐỘ PHÂN GIẢI
# ============================================================
print(f"\n📊 MORAN'S I THEO ĐỘ PHÂN GIẢI")
resolutions = [30, 90, 120, 150, 200, 250]
moran_by_res = {}

for res in resolutions:
    ser_file = rf"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\SER_{res}m.tif"
    if not os.path.exists(ser_file):
        print(f"   ⚠️ Không tìm thấy: SER_{res}m.tif")
        continue
    with rasterio.open(ser_file) as src:
        s = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else -9999
        s = np.where(s == nd, np.nan, s)
    valid = ~np.isnan(s)
    r, c = np.where(valid)
    n = min(30000, len(r))
    idx = np.random.choice(len(r), n, replace=False)
    vals = s[r[idx], c[idx]]
    coords = np.column_stack([r[idx], c[idx]])
    w_res = KNN.from_array(coords, k=8)
    w_res.transform = 'r'
    m = Moran(vals, w_res)
    moran_by_res[res] = {'I': m.I, 'p': m.p_sim, 'z': m.z_sim}
    print(f"   {res:3d} m: I = {m.I:.4f}, p = {m.p_sim:.4f}, z = {m.z_sim:.2f}")

pd.DataFrame(moran_by_res).T.to_csv(
    os.path.join(OUT_DIR, 'Moran_by_resolution.csv'),
    encoding='utf-8-sig')

# ============================================================
# 7. BIỂU ĐỒ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Moran scatterplot
ax = axes[0]
z = (df['SER'] - df['SER'].mean()) / df['SER'].std()
wz = w.sparse @ z
ax.scatter(z, wz, s=2, alpha=0.2, c='#2E86AB')
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='gray', linewidth=0.8)
b, a = np.polyfit(z, wz, 1)
x_line = np.linspace(z.min(), z.max(), 100)
ax.plot(x_line, a + b * x_line, 'r-', linewidth=2,
        label=f"Slope = {b:.3f} (Moran's I)")
ax.set_xlabel('Standardized SER (z)', fontweight='bold')
ax.set_ylabel('Spatial lag of z', fontweight='bold')
ax.set_title(f"(a) Moran scatterplot (I = {moran.I:.3f})",
             fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# (b) Moran's I theo độ phân giải
ax = axes[1]
res_list = list(moran_by_res.keys())
I_list = [moran_by_res[r]['I'] for r in res_list]
ax.plot(res_list, I_list, 'o-', color='#C0392B',
        linewidth=2, markersize=10)
for x, y in zip(res_list, I_list):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9,
                fontweight='bold')
ax.set_xlabel('Resolution (m)', fontweight='bold')
ax.set_ylabel("Moran's I", fontweight='bold')
ax.set_title("(b) Moran's I across integration resolutions",
             fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(res_list)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Fig_Moran_Autocorrelation.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✅ Đã lưu: Fig_Moran_Autocorrelation.png")
print(f"\n✅ Hoàn thành! Kết quả tại: {OUT_DIR}")