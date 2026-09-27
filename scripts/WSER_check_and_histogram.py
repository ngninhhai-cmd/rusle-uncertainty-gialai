"""
WSER_Combined_SinglePanel_v2.py
================================
Sửa lỗi chồng chéo: legend và inset boxplot tách biệt.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CẤU HÌNH
# ============================================================
WSER_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\RUSLE_CONFERENCE\PSU_lan truyen\WSER_final_ok3.tif"
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\Paper1_Corrected"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================
print("Đang đọc W_SER...")
with rasterio.open(WSER_FILE) as src:
    wser = src.read(1).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else -9999
    wser = np.where(wser == nodata, np.nan, wser)
    wser = np.where(wser < -1e30, np.nan, wser)

valid = wser[~np.isnan(wser)]
print(f"  N = {len(valid):,}, Mean = {np.mean(valid):.2f}%, "
      f"Median = {np.median(valid):.2f}%")


# ============================================================
# VẼ BIỂU ĐỒ
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5))

# ---------------------------------------------
# 1. HISTOGRAM
# ---------------------------------------------
bins = np.arange(-60, 125, 5)
counts, bin_edges, patches = ax.hist(
    valid, bins=bins,
    color="#3498DB", edgecolor="black",
    linewidth=0.5, alpha=0.85, zorder=2,
)

# Đổi màu theo khoảng
for count, patch, left in zip(counts, patches, bin_edges[:-1]):
    if left < 0:
        patch.set_facecolor("#C0392B")
    elif left < 20:
        patch.set_facecolor("#27AE60")
    elif left < 30:
        patch.set_facecolor("#F1C40F")
    elif left < 50:
        patch.set_facecolor("#E67E22")
    else:
        patch.set_facecolor("#C0392B")

ax.set_xlabel("Relative uncertainty $W_{SER}$ (%)", fontweight="bold")
ax.set_ylabel("Number of pixels", fontweight="bold", color="#2E86AB")
ax.tick_params(axis="y", labelcolor="#2E86AB")
ax.set_xlim(-60, 120)
ax.set_ylim(0, max(counts) * 1.20)
ax.grid(alpha=0.25, linestyle=":", zorder=0)

# Mean + Median lines
mean_val = np.mean(valid)
median_val = np.median(valid)
ax.axvline(mean_val, color="navy", linestyle="--",
           linewidth=1.5, label=f"Mean = {mean_val:.2f}%", zorder=3)
ax.axvline(median_val, color="darkgreen", linestyle="--",
           linewidth=1.5, label=f"Median = {median_val:.2f}%", zorder=3)

# ---------------------------------------------
# 2. CUMULATIVE DISTRIBUTION (trục phụ)
# ---------------------------------------------
ax2 = ax.twinx()
sorted_vals = np.sort(valid)
cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
step = max(1, len(sorted_vals) // 3000)
ax2.plot(sorted_vals[::step], cumulative[::step],
         color="#8E44AD", linewidth=1.8, label="Cumulative", zorder=4)
ax2.set_ylabel("Cumulative frequency (%)", fontweight="bold", color="#8E44AD")
ax2.tick_params(axis="y", labelcolor="#8E44AD")
ax2.set_ylim(0, 100)

# ---------------------------------------------
# 3. INSET BOXPLOT — ĐẶT GÓC TRÊN TRÁI
# ---------------------------------------------
# Vị trí: [x0, y0, width, height] tính theo tỉ lệ trục
ax_inset = ax.inset_axes([0.03, 0.62, 0.28, 0.20])

ax_inset.boxplot(valid, vert=False, patch_artist=True,
                 widths=0.6, showfliers=False,
                 boxprops=dict(facecolor="#3498DB",
                               edgecolor="black", linewidth=1),
                 medianprops=dict(color="red", linewidth=1.5),
                 whiskerprops=dict(color="black", linewidth=1),
                 capprops=dict(color="black", linewidth=1))

ax_inset.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
ax_inset.set_xlim(-60, 120)
ax_inset.set_yticks([])
ax_inset.set_xlabel("$W_{SER}$ (%)", fontsize=8, labelpad=1)
ax_inset.tick_params(axis="x", labelsize=7)
ax_inset.grid(alpha=0.25, linestyle=":", axis="x")
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.6)

# ---------------------------------------------
# 4. LEGEND — ĐẶT GÓC TRÊN PHẢI (tách biệt khỏi inset)
# ---------------------------------------------
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

range_patches = [
    Patch(facecolor="#C0392B", edgecolor="black",
          label="$W_{SER}$ < 0 or > 50"),
    Patch(facecolor="#27AE60", edgecolor="black",
          label="$W_{SER}$ 0–20"),
    Patch(facecolor="#F1C40F", edgecolor="black",
          label="$W_{SER}$ 20–30"),
    Patch(facecolor="#E67E22", edgecolor="black",
          label="$W_{SER}$ 30–50"),
]

handles = range_patches + lines1 + lines2
labels = [h.get_label() for h in handles]

# Legend ở góc trên phải
ax.legend(handles, labels,
          loc="upper right",
          bbox_to_anchor=(1.0, 1.0),
          frameon=True,
          framealpha=0.95,
          fontsize=8,
          ncol=1,
          borderpad=0.5)

ax.set_title("Distribution of relative uncertainty $W_{SER}$ (%)",
             fontweight="bold", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Panel_b_v2.png"))
plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Panel_b_v2.pdf"))
plt.close()

print(f"\n✅ Đã lưu:")
print(f"   Fig_WSER_Panel_b_v2.png")
print(f"   Fig_WSER_Panel_b_v2.pdf")