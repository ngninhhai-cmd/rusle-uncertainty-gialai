"""
Fig_LISA_Panel_b.py
====================
Biểu đồ cột ngang cho LISA cluster statistics.
Panel (b) của Figure LISA.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CẤU HÌNH
# ============================================================
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\Paper1_Corrected"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# DỮ LIỆU LISA
# ============================================================
LISA_DATA = pd.DataFrame({
    "Cluster": ["High-High (HH)", "Low-Low (LL)",
                "High-Low (HL) outlier", "Low-High (LH) outlier",
                "Not significant"],
    "N_pixels": [29588, 74114, 1105, 6586, 127160],
    "Area_ha": [184925, 463213, 6906, 41163, 794750],
    "Proportion": [12.40, 31.07, 0.46, 2.76, 53.30],
})

# Chỉ lấy các cụm có ý nghĩa (loại Not significant)
sig = LISA_DATA[LISA_DATA["Cluster"] != "Not significant"].copy()
sig = sig.sort_values("Proportion", ascending=True)

# ============================================================
# MÀU SẮC
# ============================================================
COLORS = {
    "High-High (HH)": "#C0392B",           # Đỏ đậm
    "Low-Low (LL)": "#2E86AB",             # Xanh đậm
    "High-Low (HL) outlier": "#E67E22",    # Cam
    "Low-High (LH) outlier": "#85C1E9",    # Xanh nhạt
}


# ============================================================
# VẼ BIỂU ĐỒ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# --- (b1) Bar chart: Proportion ---
ax = axes[0]

bars = ax.barh(
    sig["Cluster"],
    sig["Proportion"],
    color=[COLORS[c] for c in sig["Cluster"]],
    edgecolor="black",
    linewidth=0.8,
    height=0.55,
)

# Thêm giá trị và diện tích
for bar, prop, area in zip(bars, sig["Proportion"], sig["Area_ha"]):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height()/2,
        f"{prop:.2f}%\n({area:,.0f} ha)",
        va="center", fontsize=9, fontweight="bold"
    )

ax.set_xlabel("Proportion of study area (%)", fontweight="bold")
ax.set_title("(a) LISA cluster proportions",
             fontweight="bold", fontsize=11)
ax.set_xlim(0, max(sig["Proportion"]) * 1.40)
ax.grid(axis="x", alpha=0.3, linestyle=":")

# Thêm annotation về "Not significant"
ax.text(
    0.98, 0.02,
    f"Not significant: 53.30%\n(794,750 ha)",
    transform=ax.transAxes,
    fontsize=9, style="italic",
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="lightyellow",
              edgecolor="gray", alpha=0.9)
)


# --- (b2) Bar chart: Area + Moran's I range ---
ax = axes[1]

# Vẽ bar chart area
bars = ax.barh(
    sig["Cluster"],
    sig["Area_ha"],
    color=[COLORS[c] for c in sig["Cluster"]],
    edgecolor="black",
    linewidth=0.8,
    height=0.55,
)

# Thêm diện tích
for bar, area, prop in zip(bars, sig["Area_ha"], sig["Proportion"]):
    ax.text(
        bar.get_width() + 5000,
        bar.get_y() + bar.get_height()/2,
        f"{area:,.0f} ha\n({prop:.2f}%)",
        va="center", fontsize=9, fontweight="bold"
    )

ax.set_xlabel("Area (ha)", fontweight="bold")
ax.set_title("(b) LISA cluster areas",
             fontweight="bold", fontsize=11)
ax.set_xlim(0, max(sig["Area_ha"]) * 1.35)
ax.grid(axis="x", alpha=0.3, linestyle=":")

# Thêm Moran's I range annotation
moran_text = (
    "Moran's I range:\n"
    "  HH:  0.51 – 46.31\n"
    "  LL:  0.25 – 0.48\n"
    "  HL/LH: outliers"
)
ax.text(
    0.98, 0.98,
    moran_text,
    transform=ax.transAxes,
    fontsize=8, style="italic",
    verticalalignment="top",
    horizontalalignment="right",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white",
              edgecolor="gray", alpha=0.9)
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_LISA_Panel_b.png"))
plt.savefig(os.path.join(OUT_DIR, "Fig_LISA_Panel_b.pdf"))
plt.close()
print("✅ Đã lưu: Fig_LISA_Panel_b.png / .pdf")


# ============================================================
# PHIÊN BẢN 2: BIỂU ĐỒ TRÒN (PIE) VỚI "NOT SIGNIFICANT" GỘP
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Gộp Not significant vào nhóm riêng
pie_data = LISA_DATA.copy()
colors_pie = [COLORS.get(c, "#D3D3D3") for c in pie_data["Cluster"]]
explode = [0.03 if c != "Not significant" else 0
           for c in pie_data["Cluster"]]

wedges, texts, autotexts = ax.pie(
    pie_data["Proportion"],
    labels=None,
    colors=colors_pie,
    autopct=lambda p: f"{p:.2f}%" if p > 1 else "",
    startangle=90,
    counterclock=False,
    explode=explode,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 9, "fontweight": "bold"}
)

# Legend
ax.legend(
    wedges,
    [f"{c} ({p:.2f}%)" for c, p in
     zip(pie_data["Cluster"], pie_data["Proportion"])],
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=9,
    frameon=True
)

ax.set_title("LISA cluster composition of the study area",
             fontweight="bold", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_LISA_Panel_b_pie.png"))
plt.savefig(os.path.join(OUT_DIR, "Fig_LISA_Panel_b_pie.pdf"))
plt.close()
print("✅ Đã lưu: Fig_LISA_Panel_b_pie.png / .pdf")