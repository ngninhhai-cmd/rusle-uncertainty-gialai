"""
Figure_7_SER_max_vs_resolution.py
==================================
Plot SER maximum versus data integration resolution under
two LS scenarios: uncapped vs. capped (MONRE river network).

Output: Figure_7_SER_max_vs_resolution.png / .pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\SER_FINAL"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
resolutions    = [30, 90, 120, 150, 200, 250]
ser_max_uncap  = [20118.04, 3877.18, 5060.31, 4627.74, 944.61, 1472.49]
ser_max_capped = [868.19, 592.97, 622.63, 395.61, 468.60, 395.26]

# ---------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(9, 6))

# Uncapped LS — dashed red
ax.plot(resolutions, ser_max_uncap, 's--', color='#C0392B',
        linewidth=2, markersize=9, label='Uncapped LS',
        markerfacecolor='white', markeredgewidth=2)

# Capped LS — solid blue
ax.plot(resolutions, ser_max_capped, 'o-', color='#2E86AB',
        linewidth=2, markersize=9, label='Capped LS (MONRE network)',
        markerfacecolor='white', markeredgewidth=2)

# Annotate spikes on the uncapped curve
ax.annotate('Spike\n150 m', xy=(150, 4627.74), xytext=(130, 9000),
            fontsize=11, fontweight='bold', color='#C0392B',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
ax.annotate('Spike\n250 m', xy=(250, 1472.49), xytext=(260, 6000),
            fontsize=11, fontweight='bold', color='#C0392B',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))

# Log scale on Y-axis (values span two orders of magnitude)
ax.set_yscale('log')
ax.set_ylim(100, 50000)

# Labels and title
ax.set_xlabel('Data integration resolution (m)', fontweight='bold')
ax.set_ylabel('SER maximum (t ha⁻¹ yr⁻¹) – log scale', fontweight='bold')
ax.set_title('SER maximum across integration resolutions: '
             'uncapped vs. capped LS', fontweight='bold')

ax.set_xticks(resolutions)
ax.grid(alpha=0.3, which='both', linestyle=':')
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_7_SER_max_vs_resolution.png'))
plt.savefig(os.path.join(OUT_DIR, 'Figure_7_SER_max_vs_resolution.pdf'))
plt.show()

print("✅ Figure 7 saved to:", OUT_DIR)