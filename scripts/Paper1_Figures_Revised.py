"""
Paper1_Figures_Revised.py
==========================
Vẽ lại 4 biểu đồ cho Paper 1:
1. Fig_Resolution_Sensitivity_Revised.png
2. Fig_Uncertainty_Revised.png
3. Fig_Sensitivity_CorrelatedMC.png
4. Fig_Fallback_Loam.png (đã có từ trước)
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

# ============================================================
# DỮ LIỆU (từ Resolution_Sensitivity_Corrected.csv)
# ============================================================
RES_DATA = pd.DataFrame({
    'Resolution': [30, 90, 120, 150, 200, 250],
    'Mean':   [23.7886, 23.8145, 23.8413, 23.8353, 23.8595, 23.9075],
    'Median': [5.0877, 6.2647, 6.7402, 7.1081, 7.6569, 8.1444],
    'SD':     [43.6398, 38.1363, 37.0199, 36.1609, 35.1778, 34.4998],
    'P95':    [111.329, 106.040, 103.642, 101.912, 99.518, 97.405],
    'P99':    [212.711, 176.627, 169.329, 164.200, 158.796, 155.006],
    'Max':    [868.188, 592.971, 622.632, 395.610, 468.605, 395.263],
})


# ============================================================
# FIGURE 1: RESOLUTION SENSITIVITY (4 panel)
# ============================================================
def plot_resolution_sensitivity():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    res = RES_DATA['Resolution'].values

    # (a) Mean
    ax = axes[0]
    ax.plot(res, RES_DATA['Mean'], 'o-', color='#2E86AB',
            lw=2, ms=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel('Resolution (m)', fontweight='bold')
    ax.set_ylabel('Mean SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax.set_title('(a) Mean — variation 0.50%', fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    for x, y in zip(res, RES_DATA['Mean']):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_ylim(23.7, 24.0)

    # (b) Median
    ax = axes[1]
    ax.plot(res, RES_DATA['Median'], 's-', color='#27AE60',
            lw=2, ms=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel('Resolution (m)', fontweight='bold')
    ax.set_ylabel('Median SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax.set_title('(b) Median — variation 60.1%', fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    for x, y in zip(res, RES_DATA['Median']):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)

    # (c) Standard deviation
    ax = axes[2]
    ax.plot(res, RES_DATA['SD'], '^-', color='#E67E22',
            lw=2, ms=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel('Resolution (m)', fontweight='bold')
    ax.set_ylabel('SD of SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax.set_title('(c) SD — variation 26.5%', fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    for x, y in zip(res, RES_DATA['SD']):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)

    # (d) P99 and Max
    ax = axes[3]
    ax.plot(res, RES_DATA['P99'], 'D-', color='#C0392B',
            lw=2, ms=8, markerfacecolor='white', markeredgewidth=2,
            label='P99')
    ax.plot(res, RES_DATA['Max'], 'v--', color='#7F8C8D',
            lw=2, ms=7, markerfacecolor='white', markeredgewidth=2,
            label='Maximum')
    ax.set_xlabel('Resolution (m)', fontweight='bold')
    ax.set_ylabel('SER upper tail (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax.set_title('(d) Upper tail — P99 vs Max', fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='best', fontsize=9)
    for x, y in zip(res, RES_DATA['P99']):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='#C0392B')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Resolution_Sensitivity_Revised.png'))
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Resolution_Sensitivity_Revised.pdf'))
    plt.close()
    print("✅ Fig_Resolution_Sensitivity_Revised.png")


# ============================================================
# FIGURE 2: UNCERTAINTY K AND SER (2 panel)
# ============================================================
def plot_uncertainty():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # (a) K uncertainty
    ax = axes[0]
    scenarios = ['Q₀.₀₅', 'Median', 'Q₀.₉₅']
    k_vals = [0.0347, 0.0302, 0.0276]
    colors = ['#C0392B', '#2E86AB', '#27AE60']
    bars = ax.bar(scenarios, k_vals, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.6)
    for bar, val in zip(bars, k_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('K (t ha h ha⁻¹ MJ⁻¹ mm⁻¹)', fontweight='bold')
    ax.set_title(f'(a) K uncertainty (W_K = 23.51%)', fontweight='bold')
    ax.set_ylim(0, 0.042)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # (b) SER uncertainty
    ax = axes[1]
    ser_vals = [22.73, 20.06, 17.97]
    bars = ax.bar(scenarios, ser_vals, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.6)
    for bar, val in zip(bars, ser_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('SER (t ha⁻¹ yr⁻¹)', fontweight='bold')
    ax.set_title(f'(b) SER uncertainty (W_SER = 23.43%)', fontweight='bold')
    ax.set_ylim(0, 26)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Uncertainty_Revised.png'))
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Uncertainty_Revised.pdf'))
    plt.close()
    print("✅ Fig_Uncertainty_Revised.png")


# ============================================================
# FIGURE 3: SENSITIVITY (CORRELATED MC)
# ============================================================
def plot_sensitivity_correlated():
    """
    Vẽ biểu đồ độ nhạy dựa trên correlated Monte Carlo.
    KHÔNG dùng Sobol vì independence assumption bị vi phạm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    factors = ['C', 'LS', 'R', 'K']
    contributions = [75.83, 17.69, 4.87, 1.72]
    colors_sens = ['#C0392B', '#E67E22', '#3498DB', '#27AE60']

    # (a) Bar chart
    ax = axes[0]
    bars = ax.barh(factors[::-1], contributions[::-1],
                   color=colors_sens[::-1],
                   edgecolor='black', linewidth=0.8, height=0.6)
    for bar, val in zip(bars, contributions[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Contribution to SER variance (%)', fontweight='bold')
    ax.set_title('(a) Variance contribution (correlated MC)',
                 fontweight='bold')
    ax.set_xlim(0, 90)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # (b) Cumulative contribution
    ax = axes[1]
    cumsum = np.cumsum(contributions)
    x_pos = np.arange(len(factors))
    bars = ax.bar(x_pos, contributions, color=colors_sens,
                  edgecolor='black', linewidth=0.8, width=0.6)
    ax2 = ax.twinx()
    ax2.plot(x_pos, cumsum, 'ko--', lw=2, ms=8,
             markerfacecolor='white', markeredgewidth=2,
             label='Cumulative')
    for i, (bar, val, cs) in enumerate(zip(bars, contributions, cumsum)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        ax2.text(i, cs + 3, f'{cs:.1f}%', ha='center',
                 fontsize=9, color='black', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(factors, fontweight='bold')
    ax.set_ylabel('Contribution (%)', fontweight='bold')
    ax2.set_ylabel('Cumulative (%)', fontweight='bold')
    ax.set_title('(b) Ranked contributions', fontweight='bold')
    ax.set_ylim(0, 90)
    ax2.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Sensitivity_CorrelatedMC.png'))
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Sensitivity_CorrelatedMC.pdf'))
    plt.close()
    print("✅ Fig_Sensitivity_CorrelatedMC.png")


# ============================================================
# FIGURE 4: FALLBACK LOAM CHECK
# ============================================================
def plot_fallback():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Fallback percentage
    ax = axes[0]
    scenarios = ['Q₀.₀₅', 'Q₀.₉₅']
    pct_fb = [0.4419, 0.0]
    colors = ['#C0392B', '#27AE60']
    bars = ax.bar(scenarios, pct_fb, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, pct_fb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('% pixels assigned by fallback', fontweight='bold')
    ax.set_title('(a) Fallback Loam frequency', fontweight='bold')
    ax.set_ylim(0, 0.55)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # (b) K mean with/without fallback
    ax = axes[1]
    x = np.arange(2)
    width = 0.35
    k_with = [0.0347, 0.0276]
    k_strict = [0.0346, 0.0276]
    bars1 = ax.bar(x - width/2, k_strict, width,
                   label='Strict (no fallback)',
                   color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x + width/2, k_with, width,
                   label='With fallback',
                   color='#C0392B', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontweight='bold')
    ax.set_ylabel('K mean (t ha h ha⁻¹ MJ⁻¹ mm⁻¹)', fontweight='bold')
    ax.set_title('(b) Effect on mean K', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.0004,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Fallback_Loam_Revised.png'))
    plt.savefig(os.path.join(OUT_DIR, 'Fig_Fallback_Loam_Revised.pdf'))
    plt.close()
    print("✅ Fig_Fallback_Loam_Revised.png")


# ============================================================
# CHẠY TẤT CẢ
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("VẼ LẠI CÁC BIỂU ĐỒ CHO PAPER 1")
    print("=" * 70)

    plot_resolution_sensitivity()
    plot_uncertainty()
    plot_sensitivity_correlated()
    plot_fallback()

    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
    print("=" * 70)