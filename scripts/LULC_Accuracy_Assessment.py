"""
WSER_check_and_histogram.py
============================
1. Kiểm tra thống kê thực tế của WSER_final_ok3.tif
2. Vẽ histogram phân bố tần suất W_SER
3. Vẽ biểu đồ kết hợp: histogram + boxplot + cumulative
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
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# 1. ĐỌC VÀ KIỂM TRA THỐNG KÊ
# ============================================================
print("=" * 70)
print("KIỂM TRA WSER_final_ok3.tif")
print("=" * 70)

with rasterio.open(WSER_FILE) as src:
    wser = src.read(1).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else -9999
    wser = np.where(wser == nodata, np.nan, wser)
    wser = np.where(wser < -1e30, np.nan, wser)

valid = wser[~np.isnan(wser)]
n_total = wser.size
n_valid = len(valid)

print(f"  File: {os.path.basename(WSER_FILE)}")
print(f"  Total pixels: {n_total:,}")
print(f"  Valid pixels: {n_valid:,} ({100*n_valid/n_total:.2f}%)")
print(f"\n  Thống kê W_SER (%):")
print(f"    Min     = {np.min(valid):.4f}")
print(f"    Max     = {np.max(valid):.4f}")
print(f"    Mean    = {np.mean(valid):.4f}")
print(f"    Median  = {np.median(valid):.4f}")
print(f"    StdDev  = {np.std(valid):.4f}")

print(f"\n  Phân vị W_SER:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p:2d} = {np.percentile(valid, p):.4f}")

# Đếm pixel âm/dương
n_neg = int(np.sum(valid < 0))
n_pos = int(np.sum(valid >= 0))
print(f"\n  Pixel có W_SER < 0 (âm):  {n_neg:,} ({100*n_neg/n_valid:.2f}%)")
print(f"  Pixel có W_SER ≥ 0:       {n_pos:,} ({100*n_pos/n_valid:.2f}%)")

# Đếm theo khoảng
bins_pct = [(-999, 0), (0, 10), (10, 20), (20, 30), (30, 50), (50, 999)]
labels_pct = ["< 0", "0–10", "10–20", "20–30", "30–50", "> 50"]
print(f"\n  Phân bố theo khoảng W_SER:")
for (lo, hi), lab in zip(bins_pct, labels_pct):
    n = int(np.sum((valid >= lo) & (valid < hi)))
    print(f"    {lab:8s}: {n:>8,} ({100*n/n_valid:.2f}%)")


# ============================================================
# 2. VẼ HISTOGRAM
# ============================================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# --- (a) Histogram toàn bộ W_SER ---
ax1 = fig.add_subplot(gs[0, :])

# Chọn bins: từ -60 đến +120, bin width = 5
bins = np.arange(-60, 125, 5)
counts, bin_edges, patches = ax1.hist(
    valid, bins=bins, color="#3498DB",
    edgecolor="black", linewidth=0.6, alpha=0.85
)

# Đổi màu theo giá trị
for count, patch, left in zip(counts, patches, bin_edges[:-1]):
    if left < 0:
        patch.set_facecolor("#C0392B")  # Đỏ cho W_SER âm
    elif left < 20:
        patch.set_facecolor("#27AE60")  # Xanh lá cho bất định thấp
    elif left < 30:
        patch.set_facecolor("#F1C40F")  # Vàng cho bất định trung bình
    elif left < 50:
        patch.set_facecolor("#E67E22")  # Cam cho bất định cao
    else:
        patch.set_facecolor("#C0392B")  # Đỏ cho bất định rất cao

# Đường mean và median
ax1.axvline(np.mean(valid), color="navy", linestyle="--",
            linewidth=1.8, label=f"Mean = {np.mean(valid):.2f}%")
ax1.axvline(np.median(valid), color="green", linestyle="--",
            linewidth=1.8, label=f"Median = {np.median(valid):.2f}%")
ax1.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.6)

ax1.set_xlabel("Relative uncertainty W_SER (%)", fontweight="bold")
ax1.set_ylabel("Number of pixels", fontweight="bold")
ax1.set_title("(a) Distribution of W_SER (per-pixel relative uncertainty)",
              fontweight="bold")
ax1.legend(loc="upper right", frameon=True, fontsize=11)
ax1.grid(alpha=0.3, linestyle=":", axis="y")
ax1.set_xlim(-60, 120)

# Thêm chú thích cho các vùng
ax1.text(0.02, 0.95,
         f"N = {n_valid:,}\nMean = {np.mean(valid):.2f}%\n"
         f"Median = {np.median(valid):.2f}%\n"
         f"StdDev = {np.std(valid):.2f}%",
         transform=ax1.transAxes, fontsize=10,
         verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="gray", alpha=0.9))


# --- (b) Boxplot W_SER ---
ax2 = fig.add_subplot(gs[1, 0])

bp = ax2.boxplot(valid, vert=True, patch_artist=True,
                 widths=0.5, showfliers=False,
                 boxprops=dict(facecolor="#3498DB", edgecolor="black",
                               linewidth=1.2),
                 medianprops=dict(color="red", linewidth=2),
                 whiskerprops=dict(color="black", linewidth=1.2),
                 capprops=dict(color="black", linewidth=1.2))

ax2.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
ax2.axhline(np.mean(valid), color="navy", linestyle="--",
            linewidth=1.5, label=f"Mean = {np.mean(valid):.2f}%")

ax2.set_ylabel("W_SER (%)", fontweight="bold")
ax2.set_title("(b) Boxplot of W_SER", fontweight="bold")
ax2.set_xticks([])
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.3, linestyle=":", axis="y")


# --- (c) Cumulative distribution ---
ax3 = fig.add_subplot(gs[1, 1])

sorted_vals = np.sort(valid)
cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100

# Downsample để vẽ nhanh hơn
step = max(1, len(sorted_vals) // 5000)
ax3.plot(sorted_vals[::step], cumulative[::step],
         color="#2E86AB", linewidth=2)

# Đường tham chiếu
ax3.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax3.axvline(np.median(valid), color="green", linestyle="--",
            linewidth=1.5, label=f"Median = {np.median(valid):.2f}%")
ax3.axvline(np.mean(valid), color="navy", linestyle="--",
            linewidth=1.5, label=f"Mean = {np.mean(valid):.2f}%")
ax3.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)

ax3.set_xlabel("W_SER (%)", fontweight="bold")
ax3.set_ylabel("Cumulative frequency (%)", fontweight="bold")
ax3.set_title("(c) Cumulative distribution of W_SER", fontweight="bold")
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.3, linestyle=":")
ax3.set_xlim(-60, 120)
ax3.set_ylim(0, 100)

plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Distribution.png"))
plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Distribution.pdf"))
plt.close()
print(f"\n✅ Đã lưu: Fig_WSER_Distribution.png / .pdf")


# ============================================================
# 3. VẼ BẢN ĐỒ W_SER (nếu cần)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Mask NoData
wser_display = np.where(np.isnan(wser), np.nan, wser)

# Dùng colormap RdYlBu (đỏ-vàng-xanh dương)
cmap = plt.cm.RdYlBu_r
im = ax.imshow(wser_display, cmap=cmap, vmin=-60, vmax=120)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("W_SER (%)", fontweight="bold", fontsize=12)

ax.set_title("Spatial distribution of relative uncertainty W_SER (%)",
             fontweight="bold", fontsize=13)
ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Map.png"))
plt.savefig(os.path.join(OUT_DIR, "Fig_WSER_Map.pdf"))
plt.close()
print(f"✅ Đã lưu: Fig_WSER_Map.png / .pdf")


# ============================================================
# 4. XUẤT BẢNG THỐNG KÊ
# ============================================================
import pandas as pd

stats_table = pd.DataFrame([{
    "Metric": "W_SER",
    "Unit": "%",
    "N_valid": n_valid,
    "Min": round(float(np.min(valid)), 4),
    "Max": round(float(np.max(valid)), 4),
    "Mean": round(float(np.mean(valid)), 4),
    "Median": round(float(np.median(valid)), 4),
    "StdDev": round(float(np.std(valid)), 4),
    "P5": round(float(np.percentile(valid, 5)), 4),
    "P25": round(float(np.percentile(valid, 25)), 4),
    "P75": round(float(np.percentile(valid, 75)), 4),
    "P95": round(float(np.percentile(valid, 95)), 4),
    "N_negative": n_neg,
    "Pct_negative": round(100 * n_neg / n_valid, 4),
}])

stats_table.to_csv(os.path.join(OUT_DIR, "WSER_Statistics.csv"),
                   index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu: WSER_Statistics.csv")

# Bảng phân bố theo khoảng
range_table = []
for (lo, hi), lab in zip(bins_pct, labels_pct):
    n = int(np.sum((valid >= lo) & (valid < hi)))
    range_table.append({
        "Range_W_SER": lab,
        "N_pixels": n,
        "Percent": round(100 * n / n_valid, 2),
    })
range_table = pd.DataFrame(range_table)
range_table.to_csv(os.path.join(OUT_DIR, "WSER_Range_Distribution.csv"),
                   index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu: WSER_Range_Distribution.csv")

print("\n" + "=" * 70)
print("HOÀN THÀNH")
print("=" * 70)
print(f"Output: {OUT_DIR}")