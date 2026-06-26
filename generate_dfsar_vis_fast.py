import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

outputs_dir = r"C:\Users\MRaza\Documents\Isro-BAH-RS\outputs\dfsar_results"
assets_dir = r"C:\Users\MRaza\Documents\Isro-BAH-RS\assets"

def make_vis():
    print("Generating DFSAR High-Res Crop Visualization...")
    # Open CPR to get dimensions
    with rasterio.open(os.path.join(outputs_dir, "dfsar_cpr_L.tif")) as src:
        H, W = src.height, src.width
        # Take a 4000x4000 crop (100km x 100km) from the center
        crop_h, crop_w = 4000, 4000
        row_off = H // 2 - crop_h // 2
        col_off = W // 2 - crop_w // 2
        win = Window(col_off, row_off, crop_w, crop_h)
        print(f"Reading crop: {win}")

        cpr_ds = src.read(1, window=win)
    
    with rasterio.open(os.path.join(outputs_dir, "dfsar_dop.tif")) as src:
        dop_ds = src.read(1, window=win)
    with rasterio.open(os.path.join(outputs_dir, "dfsar_ice_highconf.tif")) as src:
        hc_ds = src.read(1, window=win)
    with rasterio.open(os.path.join(outputs_dir, "dfsar_subsurface_ice.tif")) as src:
        sub_ds = src.read(1, window=win)
    with rasterio.open(os.path.join(outputs_dir, "dfsar_ice_falsepos.tif")) as src:
        fp_ds = src.read(1, window=win)
    
    # Read RGB
    with rasterio.open(os.path.join(outputs_dir, "dfsar_rgb_odd.tif")) as src:
        odd_ds = src.read(1, window=win)
    with rasterio.open(os.path.join(outputs_dir, "dfsar_rgb_evn.tif")) as src:
        evn_ds = src.read(1, window=win)
    with rasterio.open(os.path.join(outputs_dir, "dfsar_rgb_vol.tif")) as src:
        vol_ds = src.read(1, window=win)

    def pct(a, p=99):
        v = a[np.isfinite(a)]
        return float(np.percentile(v, p)) if len(v) > 0 else 1.0

    odd_n = np.clip(odd_ds / (pct(odd_ds) + 1e-10), 0, 1)
    evn_n = np.clip(evn_ds / (pct(evn_ds) + 1e-10), 0, 1)
    vol_n = np.clip(vol_ds / (pct(vol_ds) + 1e-10), 0, 1)
    rgb   = np.stack([odd_n, evn_n, vol_n], axis=-1)

    cpr_vmax = min(pct(cpr_ds), 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')

    # 1. CPR Map
    ax = axes[0]
    im = ax.imshow(cpr_ds, cmap='inferno', vmin=0, vmax=cpr_vmax)
    ax.contour(cpr_ds > 1.0, levels=[0.5], colors=['cyan'], linewidths=0.4)
    ax.set_title('L-band CPR (1.25 GHz)', color='white', fontsize=12)
    c = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    c.set_label('CPR', color='white')
    c.ax.yaxis.set_tick_params(color='white')
    plt.setp(c.ax.yaxis.get_ticklabels(), color='white')

    # 2. Ice Detection Map
    ax = axes[1]
    ax.imshow(cpr_ds, cmap='gray', alpha=0.5, vmin=0, vmax=cpr_vmax)
    rgba = np.zeros((*hc_ds.shape, 4))
    rgba[hc_ds > 0.5]  = [0.0, 1.0, 1.0, 0.85]
    rgba[sub_ds > 0.5] = [0.0, 1.0, 0.2, 1.0]
    rgba[fp_ds > 0.5]  = [1.0, 0.3, 0.0, 0.65]
    ax.imshow(rgba)
    legend = [Patch(fc='cyan',     label='High-Conf Ice'),
              Patch(fc='lime',     label='Subsurface Ice Anomaly'),
              Patch(fc='orangered',label='Rocky False Positive')]
    ax.legend(handles=legend, loc='lower left', fontsize=10,
              facecolor='#0d1117', edgecolor='gray', labelcolor='white')
    ax.set_title('Subsurface Ice Detection Map', color='white', fontsize=12)

    # 3. RGB Yamaguchi
    ax = axes[2]
    ax.imshow(rgb)
    legend2 = [Patch(fc='red',   label='Odd (Surface)'),
               Patch(fc='green', label='Even (Double)'),
               Patch(fc='blue',  label='Volume (Ice)')]
    ax.legend(handles=legend2, loc='lower left', fontsize=10,
              facecolor='#0d1117', edgecolor='gray', labelcolor='white')
    ax.set_title('Yamaguchi S-band RGB', color='white', fontsize=12)

    for ax in axes.flat:
        ax.axis('off')

    fig.suptitle('DFSAR 100km x 100km Crop - Subsurface Ice Detection\nCenter of Lunar South Pole Scene', 
                 color='white', fontsize=16, fontweight='bold', y=1.05)

    plt.tight_layout()
    out_vis = os.path.join(assets_dir, 'dfsar_ice_detection.png')
    plt.savefig(out_vis, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_vis}")

if __name__ == '__main__':
    make_vis()
