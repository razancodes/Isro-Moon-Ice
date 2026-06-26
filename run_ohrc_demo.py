"""
OHRC Data Analysis & LunarFM Demo
===================================

Processes actual Chandrayaan-2 OHRC PDS4 data from the lunar south pole.
Runs both classical CV analytics AND LunarFM embedding extraction.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(r'c:\Users\MRaza\Documents\Isro-BAH-RS')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'LunarFM-Science-Release' / 'src'))

from loguru import logger

# ============================================================
# CONFIGURATION
# ============================================================

# Use the Jan 2025 scene — smallest file (693 MB), higher sun elevation (3.3°),
# and covers the deepest south polar region (lat -89.67° to -89.20°)
SCENE_DIR = r'c:\Users\MRaza\Documents\Isro-BAH-RS\ohrc_data\data\calibrated\20250125'
SCENE_NAME = 'ch2_ohr_ncp_20250125T0328498909_d_img_d18'
IMG_PATH = os.path.join(SCENE_DIR, SCENE_NAME + '.img')
XML_PATH = os.path.join(SCENE_DIR, SCENE_NAME + '.xml')

OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), 'outputs', 'ohrc_demo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scene parameters (from XML)
LINES = 60534
SAMPLES = 12000
PIXEL_RES = 0.29  # m/pixel
SUN_ELEVATION = 3.346  # degrees
SPACECRAFT_ALT = 115.17  # km

logger.info("=" * 70)
logger.info("OHRC DATA ANALYSIS — Chandrayaan-2 South Pole")
logger.info("=" * 70)
logger.info(f"Scene: {SCENE_NAME}")
logger.info(f"Image: {LINES} x {SAMPLES} pixels ({LINES*SAMPLES:,} total)")
logger.info(f"Pixel resolution: {PIXEL_RES} m/pixel")
logger.info(f"Ground coverage: {LINES*PIXEL_RES/1000:.1f} km x {SAMPLES*PIXEL_RES/1000:.1f} km")
logger.info(f"Sun elevation: {SUN_ELEVATION}° (extremely grazing — polar conditions)")
logger.info(f"Output: {OUTPUT_DIR}")

# ============================================================
# STEP 1: Load the raw OHRC image
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 1: Loading OHRC .IMG file")
logger.info("=" * 70)

# PDS4 Array_2D_Image: UnsignedByte, offset=0, shape=(LINES, SAMPLES)
ohrc_raw = np.fromfile(IMG_PATH, dtype=np.uint8).reshape(LINES, SAMPLES)

logger.info(f"Loaded: shape={ohrc_raw.shape}, dtype={ohrc_raw.dtype}")
logger.info(f"Value range: [{ohrc_raw.min()}, {ohrc_raw.max()}]")
logger.info(f"Mean: {ohrc_raw.mean():.2f}, Std: {ohrc_raw.std():.2f}")
logger.info(f"Zero pixels: {(ohrc_raw == 0).sum():,} ({(ohrc_raw == 0).mean()*100:.1f}%)")
logger.info(f"Memory: {ohrc_raw.nbytes / 1024**2:.1f} MB")

# ============================================================
# STEP 2: Work on a manageable crop for the demo
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 2: Extracting analysis crop")
logger.info("=" * 70)

# The full image is 60K x 12K — too large for real-time demo.
# Use the high-contrast crop region found via statistics: Y=48000, X=3000
CROP_SIZE = 4000
y0, y1 = 48000, 48000 + CROP_SIZE
x0, x1 = 3000, 3000 + CROP_SIZE

crop = ohrc_raw[y0:y1, x0:x1].copy()
logger.info(f"Crop: [{y0}:{y1}, {x0}:{x1}] -> shape {crop.shape}")
logger.info(f"Crop value range: [{crop.min()}, {crop.max()}]")
logger.info(f"Crop ground size: {CROP_SIZE * PIXEL_RES:.0f}m x {CROP_SIZE * PIXEL_RES:.0f}m "
            f"({CROP_SIZE * PIXEL_RES / 1000:.2f} km x {CROP_SIZE * PIXEL_RES / 1000:.2f} km)")

from lunarfm_pipeline.ohrc_analytics import (
    compute_shadow_mask, compute_roughness, detect_boulders,
    compute_hazard_map, visualize_analytics, destripe_image
)

# Apply column destriping
crop = destripe_image(crop)

# Normalize to [0, 1] for analytics
crop_float = crop.astype(np.float32) / 255.0

# ============================================================
# STEP 3: Classical OHRC Analytics
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 3: Classical Analytics (Shadow, Roughness, Boulders)")
logger.info("=" * 70)

from lunarfm_pipeline.ohrc_analytics import (
    compute_shadow_mask, compute_roughness, detect_boulders,
    compute_hazard_map, visualize_analytics
)

# Shadow mask — use a low threshold since sun elevation is only 3.3°
shadow = compute_shadow_mask(crop_float, fixed_threshold=0.05)

# Surface roughness (20-pixel window = ~6m at 0.29m/pixel)
roughness = compute_roughness(crop_float, window_size=20)

# Boulder detection (downsample 4x for speed, lower threshold for real features)
boulders = detect_boulders(crop_float, downsample_factor=4, min_sigma=2, max_sigma=8, threshold=0.02)

# Hazard map
hazard = compute_hazard_map(shadow, roughness, boulders, weights=(0.4, 0.35, 0.25))

# Visualize
visualize_analytics(
    crop_float, shadow, roughness, boulders, hazard,
    save_path=os.path.join(OUTPUT_DIR, 'ohrc_classical_analytics.png')
)

logger.info("Classical analytics complete!")

# ============================================================
# STEP 4: Full Scene Overview (downsampled)
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 4: Full scene overview (downsampled)")
logger.info("=" * 70)

# Create a downsampled overview of the entire scene
DOWNSAMPLE = 16
overview = ohrc_raw[::DOWNSAMPLE, ::DOWNSAMPLE]

fig, ax = plt.subplots(figsize=(8, 16))
ax.imshow(overview, cmap='gray', vmin=0, vmax=np.percentile(overview[overview > 0], 99))
ax.set_title(f'OHRC Full Scene Overview\n{SCENE_NAME}\n'
             f'{LINES*PIXEL_RES/1000:.1f}km x {SAMPLES*PIXEL_RES/1000:.1f}km | '
             f'Sun elev: {SUN_ELEVATION}° | Pixel: {PIXEL_RES}m',
             fontsize=11)
ax.set_xlabel(f'Samples (x{DOWNSAMPLE})')
ax.set_ylabel(f'Lines (x{DOWNSAMPLE})')

# Mark the crop region
rect = plt.Rectangle(
    (x0/DOWNSAMPLE, y0/DOWNSAMPLE),
    CROP_SIZE/DOWNSAMPLE, CROP_SIZE/DOWNSAMPLE,
    linewidth=2, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)
ax.text(x0/DOWNSAMPLE, y0/DOWNSAMPLE - 20, 'Analysis Region',
        color='red', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ohrc_full_scene_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
logger.info("Full scene overview saved")

# ============================================================
# STEP 5: Contrast-Enhanced PSR Detail View
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 5: Contrast-enhanced detail views")
logger.info("=" * 70)

from skimage.exposure import equalize_adapthist

# CLAHE on the crop for PSR visibility
crop_clahe = equalize_adapthist(crop, clip_limit=0.03)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(crop_float, cmap='gray')
axes[0].set_title(f'OHRC Crop (Original)\n{CROP_SIZE*PIXEL_RES:.0f}m x {CROP_SIZE*PIXEL_RES:.0f}m', fontsize=12)

axes[1].imshow(crop_clahe, cmap='gray')
axes[1].set_title('CLAHE Enhanced (PSR detail)', fontsize=12)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ohrc_clahe_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
logger.info("CLAHE comparison saved")

# ============================================================
# STEP 6: LunarFM Embedding Extraction
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 6: LunarFM Embedding Extraction")
logger.info("=" * 70)

import torch

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")

# Load model
from lunarfm_pipeline.model_loader import load_lunarfm_model

LUNARFM_DATA = r'C:\Users\MRaza\Documents\lunarlab-public'
model = load_lunarfm_model(
    checkpoint_path=os.path.join(LUNARFM_DATA, 'model', 'last.ckpt'),
    config_path=os.path.join(LUNARFM_DATA, 'model', 'config.yaml'),
    device=device,
    eval_mode=True,
)

# Tile the crop into 112x112 patches for LunarFM
from lunarfm_pipeline.preprocessing import tile_image, normalize_for_lunarfm

# OHRC is uint8 [0, 255] — same range as Clementine UVVIS chips
crop_for_fm = crop_float[np.newaxis, :, :]  # Shape: (1, H, W)

# Scale to match Clementine value range (0-255 integer range)
crop_for_fm_scaled = crop.astype(np.float32)[np.newaxis, :, :]  # Keep in 0-255 range

tiled = tile_image(
    crop_for_fm_scaled,
    patch_size=112,
    overlap=0,
    min_valid_fraction=0.1,
)

# Compute scene-specific statistics for correct OHRC normalization
ohrc_mean = float(crop_for_fm_scaled.mean())
ohrc_std = float(crop_for_fm_scaled.std())
logger.info(f"Using scene-specific OHRC stats -> Mean: {ohrc_mean:.2f}, Std: {ohrc_std:.2f}")
custom_stats = {'mean': ohrc_mean, 'std': ohrc_std}

tiled = normalize_for_lunarfm(tiled, modality='ClementineUVVISMosaic', custom_stats=custom_stats)

logger.info(f"Tiled into {len(tiled.patches)} patches, grid: {tiled.grid_shape}")

# Extract embeddings
from lunarfm_pipeline.embeddings import extract_embeddings

emb_result = extract_embeddings(
    model=model,
    tiled_scene=tiled,
    batch_size=16,
    device=device,
    return_spatial=False,
    pooling='global_token',
)

logger.info(f"Embedding matrix: {emb_result.global_embeddings.shape}")

# Save embeddings
np.save(os.path.join(OUTPUT_DIR, 'ohrc_embeddings.npy'), emb_result.global_embeddings)

# ============================================================
# STEP 7: Embedding Visualizations
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 7: Embedding Visualizations")
logger.info("=" * 70)

from lunarfm_pipeline.visualization import (
    pca_false_color_map, cluster_and_map, similarity_heatmap
)

# PCA false-color map
pca_false_color_map(
    emb_result,
    title=f"LunarFM Embedding PCA — OHRC South Pole ({CROP_SIZE*PIXEL_RES:.0f}m crop)",
    save_path=os.path.join(OUTPUT_DIR, 'ohrc_pca_false_color.png'),
)

# K-means clustering (unsupervised terrain segmentation)
labels, kmeans = cluster_and_map(
    emb_result,
    n_clusters=6,
    title=f"Unsupervised Terrain Segmentation — OHRC South Pole",
    save_path=os.path.join(OUTPUT_DIR, 'ohrc_terrain_clusters.png'),
)

# Similarity heatmap from center patch
center_idx = len(emb_result.global_embeddings) // 2
similarity_heatmap(
    emb_result,
    query_idx=center_idx,
    title="Cosine Similarity to Center Patch",
    save_path=os.path.join(OUTPUT_DIR, 'ohrc_similarity_heatmap.png'),
)

# ============================================================
# STEP 8: Summary Statistics
# ============================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 8: Summary")
logger.info("=" * 70)

logger.info(f"Scene: {SCENE_NAME}")
logger.info(f"Location: Lunar South Pole (~89.2°S to ~89.7°S)")
logger.info(f"Sun elevation: {SUN_ELEVATION}° (extreme grazing angle)")
logger.info(f"Resolution: {PIXEL_RES} m/pixel")
logger.info(f"Analysis crop: {CROP_SIZE}x{CROP_SIZE} px = {CROP_SIZE*PIXEL_RES:.0f}m x {CROP_SIZE*PIXEL_RES:.0f}m")
logger.info(f"Shadow fraction in crop: {shadow.mean()*100:.1f}%")
logger.info(f"Boulders detected: {len(boulders)}")
logger.info(f"LunarFM patches: {len(emb_result.global_embeddings)} (grid: {emb_result.grid_shape})")
logger.info(f"Embedding dim: {emb_result.global_embeddings.shape[1]}")
logger.info(f"Terrain clusters: {len(np.unique(labels))}")
logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")

logger.info("\n" + "=" * 70)
logger.info("DEMO COMPLETE!")
logger.info("=" * 70)
