"""
DFSAR Ice Detection Pipeline — Memory-Efficient Tiled Implementation
=====================================================================
Processes 2.4GB GeoTIFFs in 500-row chunks to stay within RAM limits
and maximize TIFF I/O read speed.

Chandrayaan-2 DFSAR L4-MOSAIC products:
  L-band: CPR (Circular Polarization Ratio), SRD, TRT
  S-band: Yamaguchi Y4R — EVN, VOL, ODD, HLX

Ice Detection (Putrevu et al. 2023 / Sinha et al. npj Space Exploration):
  HIGH CONF: CPR_L > 1.0 AND DOP < 0.87  (i.e. volume fraction > 13%)
  CANDIDATE: CPR_L > 0.8 AND DOP < 0.80
  FALSE POS: CPR_L > 1.0 AND DOP >= 0.87 (rough rocky terrain)
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from loguru import logger

# ── paths ──────────────────────────────────────────────────────────────────────
LBAND_DIR = Path(r'c:\Users\MRaza\Documents\Isro-BAH-RS\dfsar_data\L_band\data\derived\20250630')
SBAND_DIR = Path(r'c:\Users\MRaza\Documents\Isro-BAH-RS\dfsar_data\S_band\data\derived\20250630')
OUTPUT_DIR = Path(r'c:\Users\MRaza\Documents\Isro-BAH-RS\outputs\dfsar_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LBAND_CPR = str(LBAND_DIR / 'ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif')
LBAND_SRD = str(LBAND_DIR / 'ch2_sar_ndxl_20250630mpcpspwest_d_srd_xx_fp_xx_xxx.tif')
LBAND_TRT = str(LBAND_DIR / 'ch2_sar_ndxl_20250630mpcpspwest_d_trt_xx_fp_xx_xxx.tif')
SBAND_EVN = str(SBAND_DIR / 'ch2_sar_ndxl_20250630my4rspwest_d_evn_xx_fp_xx_xxx.tif')
SBAND_VOL = str(SBAND_DIR / 'ch2_sar_ndxl_20250630my4rspwest_d_vol_xx_fp_xx_xxx.tif')
SBAND_ODD = str(SBAND_DIR / 'ch2_sar_ndxl_20250630my4rspwest_d_odd_xx_fp_xx_xxx.tif')
SBAND_HLX = str(SBAND_DIR / 'ch2_sar_ndxl_20250630my4rspwest_d_hlx_xx_fp_xx_xxx.tif')

EPS             = 1e-10
PIXEL_SPACING_M = 25.0
PIXEL_AREA_M2   = PIXEL_SPACING_M ** 2


def get_profile():
    with rasterio.open(LBAND_CPR) as src:
        return src.profile.copy(), src.height, src.width


def run_tiled_pipeline():
    logger.info("=" * 70)
    logger.info("DFSAR ICE DETECTION — TILED PIPELINE (Row chunks)")
    logger.info("=" * 70)

    profile, H, W = get_profile()
    logger.info(f"Scene size: {H} × {W} pixels = {H*PIXEL_SPACING_M/1000:.1f} km × {W*PIXEL_SPACING_M/1000:.1f} km")
    
    ROW_CHUNK = 500
    n_chunks = (H + ROW_CHUNK - 1) // ROW_CHUNK
    logger.info(f"Row chunk size: {ROW_CHUNK} rows. Total chunks: {n_chunks}")

    out_profile = profile.copy()
    out_profile.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan,
                        'compress': 'lzw', 'predictor': 2})

    outputs = {
        'cpr_L':        str(OUTPUT_DIR / 'dfsar_cpr_L.tif'),
        'dop':          str(OUTPUT_DIR / 'dfsar_dop.tif'),
        'depth_index':  str(OUTPUT_DIR / 'dfsar_depth_index.tif'),
        'ice_highconf': str(OUTPUT_DIR / 'dfsar_ice_highconf.tif'),
        'ice_candidate':str(OUTPUT_DIR / 'dfsar_ice_candidate.tif'),
        'ice_falsepos': str(OUTPUT_DIR / 'dfsar_ice_falsepos.tif'),
        'subsurface_ice':str(OUTPUT_DIR / 'dfsar_subsurface_ice.tif'),
        'rgb_odd':      str(OUTPUT_DIR / 'dfsar_rgb_odd.tif'),
        'rgb_evn':      str(OUTPUT_DIR / 'dfsar_rgb_evn.tif'),
        'rgb_vol':      str(OUTPUT_DIR / 'dfsar_rgb_vol.tif'),
    }

    writers = {k: rasterio.open(v, 'w', **out_profile) for k, v in outputs.items()}

    total_valid    = 0
    total_highconf = 0
    total_candidate= 0
    total_falsepos = 0
    total_subsurface = 0
    cpr_vals_sample = []
    dop_vals_sample = []

    src_cpr = rasterio.open(LBAND_CPR)
    src_srd = rasterio.open(LBAND_SRD)
    src_trt = rasterio.open(LBAND_TRT)
    src_evn = rasterio.open(SBAND_EVN)
    src_vol = rasterio.open(SBAND_VOL)
    src_odd = rasterio.open(SBAND_ODD)
    src_hlx = rasterio.open(SBAND_HLX)

    tile_count = 0
    total_tiles = n_chunks

    try:
        for ty in range(n_chunks):
            row_off = ty * ROW_CHUNK
            tile_h  = min(ROW_CHUNK, H - row_off)
            win     = Window(0, row_off, W, tile_h)

            cpr = src_cpr.read(1, window=win).astype(np.float32)
            evn = src_evn.read(1, window=win).astype(np.float32)
            vol = src_vol.read(1, window=win).astype(np.float32)
            odd = src_odd.read(1, window=win).astype(np.float32)
            hlx = src_hlx.read(1, window=win).astype(np.float32)

            for arr in [cpr, evn, vol, odd, hlx]:
                arr[arr <= -9999] = np.nan
                arr[arr >= 9999]  = np.nan

            cpr = np.clip(cpr, 0.0, 3.0)

            total_pow = evn + vol + odd + hlx + EPS
            vol_frac  = vol / total_pow
            dop       = np.clip(1.0 - vol_frac, 0.0, 1.0)

            cpr_S = np.clip((evn + vol / 2.0) / (odd + vol / 2.0 + EPS), 0.0, 3.0)
            depth_index = cpr - cpr_S

            valid = np.isfinite(cpr) & np.isfinite(dop)

            high_conf  = valid & (cpr > 1.0) & (dop < 0.87)
            candidate  = valid & (cpr > 0.8) & (dop < 0.80)
            false_pos  = valid & (cpr > 1.0) & (dop >= 0.87)
            subsurface = (depth_index > 0.2) & high_conf

            total_valid     += int(valid.sum())
            total_highconf  += int(high_conf.sum())
            total_candidate += int(candidate.sum())
            total_falsepos  += int(false_pos.sum())
            total_subsurface+= int(subsurface.sum())

            if tile_count == 0:
                cpr_vals_sample = cpr[valid].flatten()[:10000]
                dop_vals_sample = dop[valid].flatten()[:10000]

            for key, arr in [
                ('cpr_L',         cpr),
                ('dop',           dop),
                ('depth_index',   depth_index),
                ('ice_highconf',  high_conf.astype(np.float32)),
                ('ice_candidate', candidate.astype(np.float32)),
                ('ice_falsepos',  false_pos.astype(np.float32)),
                ('subsurface_ice',subsurface.astype(np.float32)),
                ('rgb_odd',       odd),
                ('rgb_evn',       evn),
                ('rgb_vol',       vol),
            ]:
                writers[key].write(arr, 1, window=win)

            tile_count += 1
            if tile_count % 5 == 0 or tile_count == total_tiles:
                logger.info(f"  Processed {tile_count}/{total_tiles} chunks "
                            f"({tile_count/total_tiles*100:.0f}%) | "
                            f"ice_highconf so far: {total_highconf:,}")

    finally:
        for src in [src_cpr, src_srd, src_trt, src_evn, src_vol, src_odd, src_hlx]:
            src.close()
        for w in writers.values():
            w.close()

    logger.info("\n--- Sanity Checks ---")
    if len(cpr_vals_sample) > 0:
        med_cpr = float(np.nanmedian(cpr_vals_sample))
        med_dop = float(np.nanmedian(dop_vals_sample))
        logger.info(f"CPR_L median (sample): {med_cpr:.4f}  ({'PASS' if 0.2 <= med_cpr <= 1.5 else 'WARNING: outside [0.2, 1.5]'})")
        logger.info(f"DOP median (sample):   {med_dop:.4f}  ({'PASS' if 0.3 <= med_dop <= 0.99 else 'WARNING: outside [0.3, 0.99]'})")

    logger.info("\n" + "=" * 60)
    logger.info("SCENE-WIDE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Valid pixels:        {total_valid:>12,}")
    logger.info(f"  High-conf ice:       {total_highconf:>12,}  ({total_highconf/(total_valid+EPS)*100:.2f}%)")
    logger.info(f"  Candidate ice:       {total_candidate:>12,}  ({total_candidate/(total_valid+EPS)*100:.2f}%)")
    logger.info(f"  Rocky false-pos:     {total_falsepos:>12,}  ({total_falsepos/(total_valid+EPS)*100:.2f}%)")
    logger.info(f"  Subsurface ice:      {total_subsurface:>12,}  ({total_subsurface/(total_valid+EPS)*100:.2f}%)")

    ice_area_m2  = total_highconf * PIXEL_AREA_M2
    ice_area_km2 = ice_area_m2 / 1e6
    vol_low_m3   = ice_area_m2 * 5.0 * 0.05
    vol_high_m3  = ice_area_m2 * 5.0 * 0.10
    vol_low_km3  = vol_low_m3 / 1e9
    vol_high_km3 = vol_high_m3 / 1e9

    logger.info("\n" + "=" * 60)
    logger.info("ICE VOLUME ESTIMATE")
    logger.info("=" * 60)
    logger.info(f"  Ice-bearing area:    {ice_area_m2:>15,.0f} m²")
    logger.info(f"                       {ice_area_km2:>15.4f} km²")
    logger.info(f"  Depth assumed:       5 m")
    logger.info(f"  Ice fraction range:  5% – 10%")
    logger.info(f"  Volume (low):        {vol_low_m3:>15,.0f} m³  =  {vol_low_km3:.6f} km³")
    logger.info(f"  Volume (high):       {vol_high_m3:>15,.0f} m³  =  {vol_high_km3:.6f} km³")
    logger.info("=" * 60)

    vol_result = {
        'n_pixels': total_highconf, 'ice_area_km2': ice_area_km2,
        'vol_low_km3': vol_low_km3, 'vol_high_km3': vol_high_km3,
        'total_subsurface': total_subsurface,
    }

    logger.info("\n--- Generating visualization (downsampled reads)... ---")
    DS = 16

    def read_ds(path):
        with rasterio.open(path) as src:
            h, w = src.height, src.width
            out_h, out_w = max(1, h // DS), max(1, w // DS)
            return src.read(1, out_shape=(out_h, out_w),
                            resampling=rasterio.enums.Resampling.average).astype(np.float32)

    cpr_ds  = read_ds(outputs['cpr_L'])
    dop_ds  = read_ds(outputs['dop'])
    di_ds   = read_ds(outputs['depth_index'])
    hc_ds   = read_ds(outputs['ice_highconf'])
    fp_ds   = read_ds(outputs['ice_falsepos'])
    sub_ds  = read_ds(outputs['subsurface_ice'])
    odd_ds  = read_ds(outputs['rgb_odd'])
    evn_ds  = read_ds(outputs['rgb_evn'])
    vol_ds  = read_ds(outputs['rgb_vol'])

    def pct(a, p=99):
        v = a[np.isfinite(a)]
        return float(np.percentile(v, p)) if len(v) > 0 else 1.0

    odd_n = np.clip(odd_ds / (pct(odd_ds) + EPS), 0, 1)
    evn_n = np.clip(evn_ds / (pct(evn_ds) + EPS), 0, 1)
    vol_n = np.clip(vol_ds / (pct(vol_ds) + EPS), 0, 1)
    rgb   = np.stack([odd_n, evn_n, vol_n], axis=-1)

    cpr_vmax = min(pct(cpr_ds), 3.0)
    di_sym   = min(pct(np.abs(di_ds)), 2.0)

    fig, axes = plt.subplots(2, 3, figsize=(22, 15))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')

    def cb(fig, ax, im, label):
        c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        c.set_label(label, color='white', fontsize=9)
        c.ax.yaxis.set_tick_params(color='white')
        plt.setp(c.ax.yaxis.get_ticklabels(), color='white')

    ax = axes[0, 0]
    im = ax.imshow(cpr_ds, cmap='inferno', vmin=0, vmax=cpr_vmax)
    ax.contour(cpr_ds > 1.0, levels=[0.5], colors=['cyan'], linewidths=0.4)
    ax.set_title('L-band CPR (1.25 GHz)\nCyan contour: CPR > 1.0', color='white', fontsize=11)
    cb(fig, ax, im, 'CPR')

    ax = axes[0, 1]
    im = ax.imshow(dop_ds, cmap='viridis', vmin=0, vmax=1)
    ax.contour(dop_ds < 0.87, levels=[0.5], colors=['cyan'], linewidths=0.4)
    ax.set_title('DOP Proxy (1 − Vol fraction)\nCyan: DOP < 0.87 (ice-like)', color='white', fontsize=11)
    cb(fig, ax, im, 'DOP')

    ax = axes[0, 2]
    im = ax.imshow(di_ds, cmap='RdBu_r', vmin=-di_sym, vmax=di_sym)
    ax.contour(sub_ds > 0.5, levels=[0.5], colors=['lime'], linewidths=0.5)
    ax.set_title('Depth Index (CPR_L − CPR_S)\nLime: Subsurface ice', color='white', fontsize=11)
    cb(fig, ax, im, 'ΔDepth Index')

    ax = axes[1, 0]
    ax.imshow(cpr_ds, cmap='gray', alpha=0.5, vmin=0, vmax=cpr_vmax)
    rgba = np.zeros((*hc_ds.shape, 4))
    rgba[hc_ds > 0.5]  = [0.0, 1.0, 1.0, 0.85]
    rgba[sub_ds > 0.5] = [0.0, 1.0, 0.2, 1.0]
    rgba[fp_ds > 0.5]  = [1.0, 0.3, 0.0, 0.65]
    ax.imshow(rgba)
    legend = [Patch(fc='cyan',     label=f'High-Conf Ice  {total_highconf:,} px ({ice_area_km2:.2f} km²)'),
              Patch(fc='lime',     label=f'Subsurface Ice {total_subsurface:,} px'),
              Patch(fc='orangered',label=f'Rocky FP       {total_falsepos:,} px')]
    ax.legend(handles=legend, loc='lower left', fontsize=8,
              facecolor='#0d1117', edgecolor='gray', labelcolor='white')
    ax.set_title(f'Ice Detection Map\nVol: {vol_low_km3:.4f}–{vol_high_km3:.4f} km³  (5–10% ice, 5m depth)',
                 color='white', fontsize=10)

    ax = axes[1, 1]
    ax.imshow(rgb)
    legend2 = [Patch(fc='red',   label='Odd/Surface scatter'),
               Patch(fc='green', label='Even/Double bounce'),
               Patch(fc='blue',  label='Volume/Ice scatter')]
    ax.legend(handles=legend2, loc='lower left', fontsize=8,
              facecolor='#0d1117', edgecolor='gray', labelcolor='white')
    ax.set_title('Yamaguchi Y4R Decomp. RGB\nR=Odd  G=Even  B=Volume', color='white', fontsize=11)

    ax = axes[1, 2]
    ax.set_facecolor('#161b22')
    cpr_flat = cpr_ds[np.isfinite(cpr_ds)].flatten()
    if len(cpr_flat) > 0:
        ax.hist(cpr_flat, bins=200, color='#58a6ff', alpha=0.8, density=True)
        ax.axvline(1.0, color='cyan',  lw=1.5, linestyle='--', label='Ice threshold (CPR=1.0)')
        ax.axvline(float(np.nanmedian(cpr_flat)), color='orange', lw=1.5,
                   linestyle=':', label=f'Median CPR={np.nanmedian(cpr_flat):.3f}')
    ax.set_xlabel('CPR Value', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('L-band CPR Distribution\nScene-wide histogram', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='gray', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    for ax in axes.flat[:-1]:
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.axis('off') if ax != axes[1, 2] else None

    axes[1, 2].axis('on')

    fig.suptitle('Chandrayaan-2 DFSAR — Subsurface Ice Detection | Lunar South Polar Region\n'
                 'L-band 1.25 GHz CPR  +  S-band 2.5 GHz Yamaguchi Y4R  |  25 m/pixel  |  UPS Projection',
                 color='white', fontsize=13, fontweight='bold', y=1.005)

    plt.tight_layout()
    out_vis = str(OUTPUT_DIR / 'dfsar_ice_detection.png')
    plt.savefig(out_vis, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Visualization saved: {out_vis}")

    logger.info("\n" + "=" * 70)
    logger.info("DFSAR TILED PIPELINE COMPLETE")
    logger.info(f"All outputs: {OUTPUT_DIR}")
    logger.info("=" * 70)

    return vol_result


if __name__ == '__main__':
    run_tiled_pipeline()
