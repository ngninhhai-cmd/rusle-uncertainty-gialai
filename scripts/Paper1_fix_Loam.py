"""
check_fallback_Loam.py
========================
Kiểm tra số lượng pixel bị gán fallback vào nhóm Loam
trong quá trình phân loại USDA.

Áp dụng cho cả 3 kịch bản: Q0.05, median, Q0.95.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# ============================================================
# CẤU HÌNH
# ============================================================
BASE = r"D:\8_PHD. CANDIDATE\PhD_GIS\RUSLE_CONFERENCE\PSU_lan truyen"
OUT_DIR = r"D:\8_PHD. CANDIDATE\PhD_GIS\CON7\Paper1_Corrected"
os.makedirs(OUT_DIR, exist_ok=True)

NODATA = -32768

SCENARIOS = {
    'Q0.05': {
        'sand': os.path.join(BASE, 'sand_Q005.tif'),
        'silt': os.path.join(BASE, 'silt_Q005.tif'),
        'clay': os.path.join(BASE, 'clay_Q005.tif'),
        'soc':  os.path.join(BASE, 'SOC_Q005.tif'),
    },
    'median': {
        'sand': os.path.join(BASE, 'sand_mean.tif'),
        'silt': os.path.join(BASE, 'silt_mean.tif'),
        'clay': os.path.join(BASE, 'clay_mean.tif'),
        'soc':  os.path.join(BASE, 'soc_mean.tif'),
    },
    'Q0.95': {
        'sand': os.path.join(BASE, 'sand_Q095.tif'),
        'silt': os.path.join(BASE, 'silt_Q095.tif'),
        'clay': os.path.join(BASE, 'clay_Q095.tif'),
        'soc':  os.path.join(BASE, 'SOC_Q095.tif'),
    },
}

# Định nghĩa 12 nhóm USDA với tên
USDA_CLASSES = {
    1: 'Sand', 2: 'Loamy Sand', 3: 'Sandy Loam', 4: 'Loam',
    5: 'Silt Loam', 6: 'Silt', 7: 'Clay Loam', 8: 'Sandy Clay Loam',
    9: 'Silty Clay Loam', 10: 'Sandy Clay', 11: 'Silty Clay', 12: 'Clay',
}


# ============================================================
# HÀM PHÂN LOẠI USDA (trả về class ID, KHÔNG fallback)
# ============================================================
def classify_usda_strict(sand, silt, clay):
    """
    Phân loại USDA nghiêm ngặt.
    Trả về:
        result: mảng int (1-12), -9999 nếu không khớp bất kỳ điều kiện nào
        fallback_mask: mảng bool, True nếu pixel bị fallback
    """
    result = np.full(sand.shape, -9999, dtype=np.int32)

    # 1. Sand
    mask = (sand >= 85)
    result[mask] = 1
    # 2. Loamy Sand
    mask = (sand >= 70) & (sand < 85) & (clay < 15)
    result[mask] = 2
    # 3. Sandy Loam
    mask = (sand >= 50) & (sand < 70) & (clay <= 20) & (silt <= 50)
    result[mask] = 3
    # 4. Loam
    mask = (sand >= 23) & (sand < 50) & (silt >= 28) & (silt < 50) & (clay >= 7) & (clay < 27)
    result[mask] = 4
    # 5. Silt Loam
    mask = (sand < 50) & (silt >= 50) & (clay < 27)
    result[mask] = 5
    # 6. Silt
    mask = (sand < 20) & (silt >= 80) & (clay < 12)
    result[mask] = 6
    # 7. Clay Loam
    mask = (sand >= 20) & (sand < 45) & (clay >= 27) & (clay < 40)
    result[mask] = 7
    # 8. Sandy Clay Loam
    mask = (sand >= 45) & (sand < 80) & (clay >= 20) & (clay < 35)
    result[mask] = 8
    # 9. Silty Clay Loam
    mask = (sand < 20) & (silt >= 40) & (silt < 73) & (clay >= 27) & (clay < 40)
    result[mask] = 9
    # 10. Sandy Clay
    mask = (sand >= 45) & (sand < 65) & (clay >= 35) & (clay < 55)
    result[mask] = 10
    # 11. Silty Clay
    mask = (sand < 20) & (silt >= 40) & (silt < 60) & (clay >= 40) & (clay < 60)
    result[mask] = 11
    # 12. Clay
    mask = (sand < 45) & (silt < 40) & (clay >= 40)
    result[mask] = 12

    fallback_mask = (result == -9999)
    return result, fallback_mask


# ============================================================
# ĐỌC VÀ XỬ LÝ TỪNG KỊCH BẢN
# ============================================================
def process_scenario(name, files):
    print(f"\n{'='*70}")
    print(f"KỊCH BẢN: {name}")
    print(f"{'='*70}")

    # Đọc dữ liệu
    with rasterio.open(files['sand']) as src:
        sand_raw = src.read(1).astype(np.float32)
        shape = sand_raw.shape
    with rasterio.open(files['silt']) as src:
        silt_raw = src.read(1).astype(np.float32)
    with rasterio.open(files['clay']) as src:
        clay_raw = src.read(1).astype(np.float32)
    with rasterio.open(files['soc']) as src:
        soc_raw = src.read(1).astype(np.float32)

    # Mask hợp lệ
    valid = (sand_raw != NODATA) & (silt_raw != NODATA) & \
            (clay_raw != NODATA) & (soc_raw != NODATA)
    n_total = int(np.sum(valid))
    print(f"  Tổng pixel hợp lệ: {n_total:,}")

    # Chuyển đơn vị
    sand = np.where(valid, sand_raw / 10.0, np.nan)
    silt = np.where(valid, silt_raw / 10.0, np.nan)
    clay = np.where(valid, clay_raw / 10.0, np.nan)

    # ==========================================
    # KIỂM TRA TỔNG TRƯỚC KHI CHUẨN HÓA
    # ==========================================
    sum_ssc_before = sand + silt + clay
    sum_valid = sum_ssc_before[valid & ~np.isnan(sum_ssc_before)]

    print(f"\n  Tổng sand+silt+clay TRƯỚC chuẩn hóa:")
    print(f"    Min  = {np.min(sum_valid):.2f}%")
    print(f"    Max  = {np.max(sum_valid):.2f}%")
    print(f"    Mean = {np.mean(sum_valid):.2f}%")
    print(f"    Std  = {np.std(sum_valid):.2f}%")

    # Đếm pixel có tổng nằm ngoài [95, 105]
    n_outside = int(np.sum((sum_valid < 95) | (sum_valid > 105)))
    print(f"    Số pixel có tổng ngoài [95, 105]: {n_outside:,} "
          f"({100*n_outside/n_total:.2f}%)")

    # ==========================================
    # CHUẨN HÓA TỔNG = 100%
    # ==========================================
    sum_ssc = sand + silt + clay
    mask_norm = (sum_ssc > 0) & valid
    sand_n = np.where(mask_norm, sand / sum_ssc * 100, np.nan)
    silt_n = np.where(mask_norm, silt / sum_ssc * 100, np.nan)
    clay_n = np.where(mask_norm, clay / sum_ssc * 100, np.nan)

    # ==========================================
    # PHÂN LOẠI USDA STRICT
    # ==========================================
    sand_f = np.where(valid, sand_n, -9999)
    silt_f = np.where(valid, silt_n, -9999)
    clay_f = np.where(valid, clay_n, -9999)

    tex_class, fallback_mask = classify_usda_strict(sand_f, silt_f, clay_f)

    n_fallback = int(np.sum(fallback_mask & valid))
    n_classified = n_total - n_fallback

    print(f"\n  PHÂN LOẠI USDA (sau chuẩn hóa):")
    print(f"    Tổng pixel hợp lệ:        {n_total:,}")
    print(f"    Pixel phân loại được:     {n_classified:,} "
          f"({100*n_classified/n_total:.4f}%)")
    print(f"    Pixel bị FALLBACK Loam:   {n_fallback:,} "
          f"({100*n_fallback/n_total:.4f}%)")

    # ==========================================
    # PHÂN BỐ 12 NHÓM USDA (trước fallback)
    # ==========================================
    print(f"\n  Phân bố 12 nhóm USDA (trước fallback):")
    print(f"  {'ID':<5}{'Tên nhóm':<20}{'Số pixel':>14}{'Tỷ lệ (%)':>12}")
    print("  " + "-" * 51)

    class_dist = []
    for cls_id in range(1, 13):
        n = int(np.sum((tex_class == cls_id) & valid))
        pct = 100 * n / n_total if n_total > 0 else 0
        class_dist.append({
            'Class_ID': cls_id,
            'Class_Name': USDA_CLASSES[cls_id],
            'N_pixels': n,
            'Percent': pct,
        })
        print(f"  {cls_id:<5}{USDA_CLASSES[cls_id]:<20}{n:>14,}{pct:>12.4f}")

    # ==========================================
    # ÁP DỤNG FALLBACK (gán Loam cho pixel không khớp)
    # ==========================================
    tex_class_final = tex_class.copy()
    tex_class_final[fallback_mask & valid] = 4  # gán Loam

    # Phân bố sau fallback
    print(f"\n  Phân bố 12 nhóm USDA (SAU fallback):")
    print(f"  {'ID':<5}{'Tên nhóm':<20}{'Số pixel':>14}{'Tỷ lệ (%)':>12}")
    print("  " + "-" * 51)
    for cls_id in range(1, 13):
        n = int(np.sum((tex_class_final == cls_id) & valid))
        pct = 100 * n / n_total if n_total > 0 else 0
        print(f"  {cls_id:<5}{USDA_CLASSES[cls_id]:<20}{n:>14,}{pct:>12.4f}")

    # ==========================================
    # SO SÁNH K TRƯỚC VÀ SAU FALLBACK (nếu có thể)
    # ==========================================
    # Bảng K
    K_TABLE = {
        1: [0.05, 0.03, 0.02], 2: [0.12, 0.10, 0.08],
        3: [0.27, 0.24, 0.19], 4: [0.38, 0.34, 0.29],
        5: [0.48, 0.42, 0.33], 6: [0.60, 0.52, 0.42],
        7: [0.28, 0.25, 0.21], 8: [0.27, 0.25, 0.21],
        9: [0.37, 0.32, 0.26], 10: [0.14, 0.13, 0.12],
        11: [0.25, 0.23, 0.19], 12: [0.165, 0.165, 0.165],
    }

    # OM
    om = np.where(valid, soc_raw / 100.0 * 1.724, np.nan)

    def compute_K(tex_arr):
        """Tính K từ mảng texture (đã có fallback)."""
        K_us = np.full(shape, np.nan, dtype=np.float32)
        for tex_id, (K0, K2, K4) in K_TABLE.items():
            mask = (tex_arr == tex_id) & valid
            if not np.any(mask):
                continue
            om_v = om[mask]
            K_v = np.full(om_v.shape, np.nan, dtype=np.float32)

            if tex_id == 12:
                K_v[:] = K0
            else:
                c1 = (om_v >= 4)
                K_v[c1] = K4
                c2 = (om_v >= 2) & (om_v < 4)
                K_v[c2] = K2 + (K4 - K2) / 2.0 * (om_v[c2] - 2.0)
                c3 = (om_v >= 0.5) & (om_v < 2)
                K_v[c3] = K0 + (K2 - K0) / 1.5 * (om_v[c3] - 0.5)
                c4 = (om_v < 0.5)
                K_v[c4] = K0
            K_us[mask] = K_v
        return K_us * 0.1317

    # K với fallback (thực tế đã dùng)
    K_fallback = compute_K(tex_class_final)
    # K không fallback (chỉ pixel phân loại được)
    K_strict = compute_K(tex_class)

    K_fb_valid = K_fallback[~np.isnan(K_fallback)]
    K_st_valid = K_strict[~np.isnan(K_strict)]

    print(f"\n  SO SÁNH K:")
    print(f"    K với fallback (toàn vùng):")
    print(f"      N = {len(K_fb_valid):,}, mean = {np.mean(K_fb_valid):.4f}, "
          f"min = {np.min(K_fb_valid):.4f}, max = {np.max(K_fb_valid):.4f}")
    print(f"    K không fallback (chỉ pixel phân loại được):")
    print(f"      N = {len(K_st_valid):,}, mean = {np.mean(K_st_valid):.4f}, "
          f"min = {np.min(K_st_valid):.4f}, max = {np.max(K_st_valid):.4f}")

    diff_mean = np.mean(K_fb_valid) - np.mean(K_st_valid)
    diff_pct = 100 * diff_mean / np.mean(K_st_valid)
    print(f"    Chênh lệch mean: {diff_mean:+.6f} ({diff_pct:+.4f}%)")

    return {
        'Scenario': name,
        'N_total': n_total,
        'N_classified': n_classified,
        'N_fallback': n_fallback,
        'Pct_fallback': 100 * n_fallback / n_total if n_total > 0 else 0,
        'K_mean_with_fallback': float(np.mean(K_fb_valid)),
        'K_mean_strict': float(np.mean(K_st_valid)),
        'K_diff_pct': diff_pct,
        'Class_distribution': class_dist,
    }


# ============================================================
# CHẠY CHO CẢ 3 KỊCH BẢN
# ============================================================
if __name__ == '__main__':
    print("#" * 70)
    print("# KIỂM TRA FALLBACK LOAM TRONG PHÂN LOẠI USDA")
    print("#" * 70)

    results = []
    for name, files in SCENARIOS.items():
        # Kiểm tra file tồn tại
        missing = [k for k, v in files.items() if not os.path.exists(v)]
        if missing:
            print(f"\n⚠️ Kịch bản {name}: thiếu file {missing}")
            continue
        res = process_scenario(name, files)
        results.append(res)

    # ==========================================
    # TỔNG HỢP KẾT QUẢ
    # ==========================================
    if results:
        print("\n" + "=" * 70)
        print("TỔNG HỢP FALLBACK LOAM")
        print("=" * 70)

        summary = pd.DataFrame([{
            'Scenario': r['Scenario'],
            'N_total': r['N_total'],
            'N_classified': r['N_classified'],
            'N_fallback': r['N_fallback'],
            'Pct_fallback': round(r['Pct_fallback'], 4),
            'K_with_fallback': round(r['K_mean_with_fallback'], 4),
            'K_strict': round(r['K_mean_strict'], 4),
            'K_diff_pct': round(r['K_diff_pct'], 4),
        } for r in results])

        print("\n" + summary.to_string(index=False))

        # Lưu CSV
        csv_file = os.path.join(OUT_DIR, 'Fallback_Loam_Check.csv')
        summary.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Đã lưu: {csv_file}")

        # Lưu phân bố class chi tiết
        all_dist = []
        for r in results:
            for c in r['Class_distribution']:
                all_dist.append({
                    'Scenario': r['Scenario'],
                    'Class_ID': c['Class_ID'],
                    'Class_Name': c['Class_Name'],
                    'N_pixels': c['N_pixels'],
                    'Percent': round(c['Percent'], 4),
                })
        dist_df = pd.DataFrame(all_dist)
        dist_file = os.path.join(OUT_DIR, 'USDA_Class_Distribution.csv')
        dist_df.to_csv(dist_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu: {dist_file}")

        # ==========================================
        # BIỂU ĐỒ
        # ==========================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Bar chart: % fallback theo kịch bản
        ax = axes[0]
        scenarios = summary['Scenario'].values
        pct_fb = summary['Pct_fallback'].values
        colors = ['#C0392B', '#2E86AB', '#27AE60']
        bars = ax.bar(scenarios, pct_fb, color=colors,
                      edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, pct_fb):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(pct_fb) * 0.02,
                    f'{val:.4f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('% pixel bị fallback Loam', fontweight='bold')
        ax.set_xlabel('Kịch bản', fontweight='bold')
        ax.set_title('(a) Tỷ lệ pixel bị fallback Loam',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # (b) Bar chart: K_mean trước và sau fallback
        ax = axes[1]
        x = np.arange(len(scenarios))
        width = 0.35
        k_fb = summary['K_with_fallback'].values
        k_st = summary['K_strict'].values
        bars1 = ax.bar(x - width/2, k_st, width,
                       label='Không fallback (strict)',
                       color='#2E86AB', edgecolor='black')
        bars2 = ax.bar(x + width/2, k_fb, width,
                       label='Có fallback (thực tế)',
                       color='#C0392B', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontweight='bold')
        ax.set_ylabel('K mean (t ha h ha⁻¹ MJ⁻¹ mm⁻¹)', fontweight='bold')
        ax.set_title('(b) Ảnh hưởng của fallback đến K mean',
                     fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.0001,
                        f'{h:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'Fig_Fallback_Loam.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu: Fig_Fallback_Loam.png")

    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH – Kết quả tại: {OUT_DIR}")
    print("=" * 70)