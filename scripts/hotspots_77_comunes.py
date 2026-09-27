"""
Figure_Commune_Hotspots.py
===========================
Vẽ biểu đồ cho mục "Commune-level erosion hotspots" với 77 xã.

Chiến lược trình bày:
- Không hiển thị cả 77 xã trong 1 bar chart (quá dài, chữ nhỏ)
- Sử dụng 4 panel phù hợp với chuẩn Q1:
    (a) Top 20 xã có SER_mean cao nhất (horizontal bar, màu gradient)
    (b) Scatter: SER_mean vs % Class 4+5 (xác định hotspots)
    (c) Phân bố SER_mean của 77 xã (histogram + KDE)
    (d) Box plot phân bố SER_mean theo nhóm nguy cơ
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# CẤU HÌNH
# ============================================================
CSV_FILE = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\ByCommune\TCVN_ByCommune_Statistics.csv"
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL\ByCommune"

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Đọc dữ liệu
df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
print(f"Đã đọc {len(df)} xã")
print(f"Cột: {df.columns.tolist()}")

# Tính % Class 4+5
df['Class45_pct'] = df['Class4_pct'] + df['Class5_pct']

# Sắp xếp theo SER_mean giảm dần
df = df.sort_values('SER_mean', ascending=False).reset_index(drop=True)


# ============================================================
# HÌNH 1: 4 PANEL TỔNG HỢP
# ============================================================
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

# ------------------------------------------------------------
# (a) Top 20 xã có SER_mean cao nhất – Horizontal Bar
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, :])  # Chiếm cả hàng trên

top20 = df.head(20).copy()
top20 = top20.sort_values('SER_mean', ascending=True)  # Đảo ngược để xếp từ dưới lên

# Colormap gradient đỏ
cmap = LinearSegmentedColormap.from_list('red_grad',
                                          ['#FFCDD2', '#B71C1C'])
colors = cmap(np.linspace(0.3, 1.0, len(top20)))

bars = ax1.barh(top20['Commune'], top20['SER_mean'],
                color=colors, edgecolor='black', linewidth=0.8)

# Thêm giá trị và % Class 4+5
for i, (bar, val, c45) in enumerate(zip(bars,
                                          top20['SER_mean'],
                                          top20['Class45_pct'])):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}  ({c45:.0f}%)',
             va='center', fontsize=9.5, fontweight='bold')

ax1.set_xlabel('Mean SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
ax1.set_title('(a) Top 20 communes by mean SER '
              '(value in parentheses: % area in Class 4+5)',
              fontweight='bold')
ax1.set_xlim(0, top20['SER_mean'].max() * 1.25)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.tick_params(axis='y', labelsize=10)

# ------------------------------------------------------------
# (b) Scatter: SER_mean vs % Class 4+5
# ------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])

# Phân loại theo mức nguy cơ
def classify_risk(row):
    if row['Class45_pct'] >= 60 and row['SER_mean'] >= 30:
        return 'Critical'
    elif row['Class45_pct'] >= 40 or row['SER_mean'] >= 25:
        return 'High'
    elif row['Class45_pct'] >= 20 or row['SER_mean'] >= 15:
        return 'Moderate'
    else:
        return 'Low'

df['Risk'] = df.apply(classify_risk, axis=1)

risk_colors = {
    'Critical': '#B71C1C',
    'High': '#E53935',
    'Moderate': '#FB8C00',
    'Low': '#2E7D32',
}
risk_order = ['Low', 'Moderate', 'High', 'Critical']

for risk in risk_order:
    sub = df[df['Risk'] == risk]
    ax2.scatter(sub['SER_mean'], sub['Class45_pct'],
                s=80, c=risk_colors[risk], alpha=0.75,
                edgecolors='black', linewidth=0.6,
                label=f'{risk} (n={len(sub)})', zorder=3)

# Đường tham chiếu
ax2.axvline(25, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax2.axhline(40, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# Annotate top 5 xã nổi bật
top5 = df.head(5)
for _, row in top5.iterrows():
    ax2.annotate(row['Commune'],
                 (row['SER_mean'], row['Class45_pct']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', alpha=0.8,
                           edgecolor='gray', linewidth=0.5))

ax2.set_xlabel('Mean SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
ax2.set_ylabel('% area in Class 4+5 (Severe + Very severe)',
               fontweight='bold')
ax2.set_title('(b) Commune risk matrix', fontweight='bold')
ax2.legend(loc='lower right', frameon=True, fontsize=9)
ax2.grid(alpha=0.3, linestyle='--')

# ------------------------------------------------------------
# (c) Histogram phân bố SER_mean của 77 xã
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])

n_bins = 15
counts, bins, patches = ax3.hist(df['SER_mean'], bins=n_bins,
                                  color='#FDD835', edgecolor='black',
                                  linewidth=0.8, alpha=0.85)

# Đổi màu theo giá trị
for count, patch, left in zip(counts, patches, bins[:-1]):
    if left >= 30:
        patch.set_facecolor('#B71C1C')
    elif left >= 20:
        patch.set_facecolor('#E53935')
    elif left >= 10:
        patch.set_facecolor('#FB8C00')
    else:
        patch.set_facecolor('#2E7D32')

# Đường mean và median
mean_val = df['SER_mean'].mean()
median_val = df['SER_mean'].median()
ax3.axvline(mean_val, color='navy', linestyle='--', linewidth=1.5,
            label=f'Mean = {mean_val:.1f}')
ax3.axvline(median_val, color='green', linestyle='--', linewidth=1.5,
            label=f'Median = {median_val:.1f}')

ax3.set_xlabel('Mean SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
ax3.set_ylabel('Number of communes', fontweight='bold')
ax3.set_title(f'(c) Distribution of commune-level mean SER (n={len(df)})',
              fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fontsize=9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.savefig(os.path.join(OUT_DIR, 'Fig_Commune_Hotspots.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'Fig_Commune_Hotspots.pdf'),
            bbox_inches='tight')
plt.close()
print("✅ Đã lưu: Fig_Commune_Hotspots.png / .pdf")


# ============================================================
# HÌNH 2: FOCUS – TOP 15 & BOTTOM 10 XÃ (dạng dumbbell)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 8))

# Lấy top 15 + bottom 10
focus = pd.concat([df.head(15), df.tail(10)]).reset_index(drop=True)
focus = focus.sort_values('SER_mean', ascending=True)

y_pos = np.arange(len(focus))

# Vẽ đường nối giữa SER_mean và SER_max (dumbbell)
for i, row in focus.iterrows():
    ax.plot([row['SER_mean'], row['SER_max']], [i, i],
            color='lightgray', linewidth=1.5, zorder=1)

# Điểm SER_mean (đỏ)
ax.scatter(focus['SER_mean'], y_pos, s=120, c='#C0392B',
           edgecolors='black', linewidth=0.8, zorder=3,
           label='Mean SER')
# Điểm SER_max (xám)
ax.scatter(focus['SER_max'], y_pos, s=80, c='#7F8C8D',
           edgecolors='black', linewidth=0.8, zorder=3,
           marker='s', label='Max SER')

# Nhãn trục Y
ax.set_yticks(y_pos)
ax.set_yticklabels(focus['Commune'], fontsize=10)

# Đường phân cách giữa top và bottom
ax.axhline(10.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.text(-30, 10.5, '◄ Bottom 10 communes  |  Top 15 communes ►',
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='lightyellow', edgecolor='gray'))

ax.set_xlabel('SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
ax.set_title('Commune-level SER: Top 15 (highest) and Bottom 10 (lowest)',
             fontweight='bold', fontsize=13)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Fig_Commune_Top15_Bottom10.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'Fig_Commune_Top15_Bottom10.pdf'),
            bbox_inches='tight')
plt.close()
print("✅ Đã lưu: Fig_Commune_Top15_Bottom10.png / .pdf")


# ============================================================
# BẢNG TÓM TẮT 4 NHÓM NGUY CƠ
# ============================================================
risk_summary = df.groupby('Risk').agg(
    N_communes=('Commune', 'count'),
    SER_mean_avg=('SER_mean', 'mean'),
    SER_max_avg=('SER_max', 'mean'),
    Class45_pct_avg=('Class45_pct', 'mean'),
    Total_area_ha=('Area_ha', 'sum'),
).reindex(risk_order)

risk_summary['%_area'] = 100 * risk_summary['Total_area_ha'] / df['Area_ha'].sum()

risk_summary.round(2).to_csv(
    os.path.join(OUT_DIR, 'Commune_Risk_Summary.csv'),
    encoding='utf-8-sig')

print("\n" + "=" * 70)
print("BẢNG TÓM TẮT THEO NHÓM NGUY CƠ")
print("=" * 70)
print(risk_summary.round(2).to_string())

print("\n✅ Hoàn thành!")
print(f"📁 Kết quả tại: {OUT_DIR}")