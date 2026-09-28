"""Publication-quality visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_fig(fig, path_png, save_pdf=True):
    """Save figure as PNG and PDF."""
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(str(path_png).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_resolution_sensitivity(stats_df, path_png):
    """4-panel resolution sensitivity figure."""
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    res = stats_df["Resolution_m"].values
    configs = [
        ("Mean", "o-", "#2E86AB", "Mean SER (t ha⁻¹ yr⁻¹)"),
        ("Median", "s-", "#27AE60", "Median SER (t ha⁻¹ yr⁻¹)"),
        ("SD", "^-", "#E67E22", "SD of SER (t ha⁻¹ yr⁻¹)"),
        ("P99", "D-", "#C0392B", "P99 SER (t ha⁻¹ yr⁻¹)"),
    ]
    for ax, (metric, style, color, ylabel) in zip(axes, configs):
        vals = stats_df[metric].values
        ax.plot(res, vals, style, color=color, lw=2, ms=8,
                markerfacecolor="white", markeredgewidth=2)
        ax.set_xlabel("Resolution (m)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        var = (vals.max() - vals.min()) / vals.mean() * 100
        ax.set_title(f"{metric} — variation {var:.2f}%", fontweight="bold")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xticks(res)
    plt.tight_layout()
    save_fig(fig, path_png)


def plot_uncertainty(K_vals, SER_vals, W_K, W_SER, path_png):
    """2-panel uncertainty figure."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scenarios = ["Q₀.₀₅", "Median", "Q₀.₉₅"]
    colors = ["#C0392B", "#2E86AB", "#27AE60"]
    for ax, vals, ylabel, title in zip(
        axes, [K_vals, SER_vals],
        ["K (t ha h ha⁻¹ MJ⁻¹ mm⁻¹)", "SER (t ha⁻¹ yr⁻¹)"],
        [f"K uncertainty (W_K = {W_K:.2f}%)",
         f"SER uncertainty (W_SER = {W_SER:.2f}%)"],
    ):
        bars = ax.bar(scenarios, vals, color=colors,
                      edgecolor="black", linewidth=0.8, width=0.6)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if max(vals) < 1 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    fmt, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    save_fig(fig, path_png)


def plot_lisa(cluster_raster, cluster_summary_df, path_png):
    """2-panel LISA figure."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    colors_map = ["#D3D3D3", "#C0392B", "#85C1E9", "#2E86AB", "#E67E22"]
    cmap = ListedColormap(colors_map)
    display = np.where(cluster_raster == -9999, np.nan, cluster_raster)
    ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=4.5)
    legend_elements = [
        Patch(facecolor="#C0392B", edgecolor="black", label="HH"),
        Patch(facecolor="#2E86AB", edgecolor="black", label="LL"),
        Patch(facecolor="#E67E22", edgecolor="black", label="HL"),
        Patch(facecolor="#85C1E9", edgecolor="black", label="LH"),
        Patch(facecolor="#D3D3D3", edgecolor="black", label="Not sig."),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_title("(a) LISA cluster map", fontweight="bold")
    ax.axis("off")

    ax = axes[1]
    sig = cluster_summary_df[cluster_summary_df["Code"] > 0].sort_values("Area_ha")
    colors_bar = ["#85C1E9", "#E67E22", "#2E86AB", "#C0392B"]
    bars = ax.barh(sig["Cluster_type"], sig["Area_ha"],
                   color=colors_bar, edgecolor="black", height=0.6)
    for bar, val, pct in zip(bars, sig["Area_ha"], sig["Percent"]):
        ax.text(bar.get_width() + sig["Area_ha"].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} ha ({pct:.2f}%)",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Area (ha)", fontweight="bold")
    ax.set_title("(b) Area by cluster type", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()
    save_fig(fig, path_png)