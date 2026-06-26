"""
OHRC Analytics Module
=====================

Extracts classical CV and photogrammetry metrics from Chandrayaan-2 OHRC imagery.
These metrics (shadows, boulders, roughness) are combined into a unified
Terrain Hazard Score Map for landing site and rover traverse planning.

Provides:
- PDS4 (.IMG/.XML) data loading
- Shadow Masking (proxy for Permanently Shadowed Regions)
- Surface Roughness (local standard deviation)
- Boulder Detection (LoG blob detection)
- Hazard Score Map generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import generic_filter
from skimage.feature import blob_log
from skimage.transform import resize
from typing import Tuple, Dict, Any, List
from pathlib import Path
from loguru import logger


def load_ohrc_pds4(xml_path: str) -> np.ndarray:
    """
    Load OHRC data from a PDS4 .XML label and corresponding .IMG file.
    
    Args:
        xml_path: Path to the PDS4 .XML label file.
        
    Returns:
        numpy array of the image data
    """
    try:
        import pds4_tools
    except ImportError:
        logger.error("pds4_tools not installed. Run: pip install pds4_tools")
        raise
        
    logger.info(f"Loading PDS4 product: {xml_path}")
    struct = pds4_tools.read(xml_path, quiet=True)
    
    # OHRC data is typically in the first array structure
    ohrc_data = struct[0].data
    
    logger.info(f"Loaded OHRC data: shape={ohrc_data.shape}, dtype={ohrc_data.dtype}")
    return ohrc_data


def compute_shadow_mask(image: np.ndarray, threshold_percentile: float = 5.0, fixed_threshold: float = None) -> np.ndarray:
    """
    Compute a shadow mask (proxy for PSRs) from the OHRC image.
    
    Args:
        image: Normalized image array (0-1)
        threshold_percentile: The lower percentile of brightness to consider as shadow.
        fixed_threshold: If provided, use this fixed value instead of percentile (e.g. 0.05).
        
    Returns:
        Boolean array where True indicates shadow.
    """
    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold = np.percentile(image, threshold_percentile)
        
    logger.info(f"Computing shadow mask with threshold: {threshold:.4f}")
    mask = image < threshold
    
    shadow_fraction = mask.sum() / mask.size
    logger.info(f"Shadow fraction: {shadow_fraction:.2%}")
    return mask


def destripe_image(image: np.ndarray) -> np.ndarray:
    """
    Remove vertical column striping artifacts from TDI CCD readout.
    Subtracts the column mean and adds back the global mean.
    
    Args:
        image: Original image array
        
    Returns:
        Destriped image array
    """
    logger.info("Applying column destriping to remove TDI artifacts...")
    col_means = image.mean(axis=0, keepdims=True)
    scene_mean = image.mean()
    destriped = image - col_means + scene_mean
    
    # Clip to original dtype bounds if uint8, or just return float
    if image.dtype == np.uint8:
        destriped = np.clip(destriped, 0, 255).astype(np.uint8)
    return destriped


def compute_roughness(image: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Compute surface roughness using a sliding window standard deviation.
    At 0.25m/pixel, a 20x20 window corresponds to 5m x 5m.
    Uses fast uniform filters for local variance: E[X^2] - E[X]^2
    
    Args:
        image: Image array
        window_size: Size of the sliding window in pixels
        
    Returns:
        Roughness array (same shape as input)
    """
    from scipy.ndimage import uniform_filter
    logger.info(f"Computing surface roughness (window_size={window_size})")
    
    img_float = image.astype(np.float64)
    # Fast local variance calculation
    c1 = uniform_filter(img_float, size=window_size)
    c2 = uniform_filter(img_float**2, size=window_size)
    
    variance = c2 - c1**2
    variance[variance < 0] = 0  # Handle floating point inaccuracies
    
    roughness = np.sqrt(variance)
    return roughness.astype(np.float32)


def detect_boulders(
    image: np.ndarray,
    downsample_factor: int = 4,
    min_sigma: float = 2.0,
    max_sigma: float = 6.0,
    threshold: float = 0.02
) -> np.ndarray:
    """
    Detect boulders using Laplacian of Gaussian (LoG) blob detection.
    
    Args:
        image: Normalized image array (0-1)
        downsample_factor: Downsample before detection for speed
        min_sigma: Min blob size
        max_sigma: Max blob size
        threshold: Detection threshold
        
    Returns:
        Array of shape [N_boulders, 3] where columns are (y, x, radius) in original coordinates
    """
    logger.info("Detecting boulders via LoG blob detection...")
    
    # Downsample for speed
    h, w = image.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    img_small = resize(image, (new_h, new_w), anti_aliasing=True)
    
    blobs = blob_log(img_small, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    
    # Scale coordinates and radii back to original image size
    # blob_log returns (y, x, sigma). Radius is approx sqrt(2) * sigma.
    boulders = np.zeros_like(blobs)
    if len(blobs) > 0:
        boulders[:, 0] = blobs[:, 0] * downsample_factor  # y
        boulders[:, 1] = blobs[:, 1] * downsample_factor  # x
        boulders[:, 2] = blobs[:, 2] * downsample_factor * np.sqrt(2)  # radius
        
    logger.info(f"Detected {len(boulders)} boulders")
    return boulders


def compute_hazard_map(
    shadow_mask: np.ndarray,
    roughness: np.ndarray,
    boulders: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
) -> np.ndarray:
    """
    Combine shadow, roughness, and boulders into a unified Hazard Score Map (0-1).
    
    Args:
        shadow_mask: Boolean shadow array
        roughness: Roughness array
        boulders: Array of [y, x, radius] for boulders
        weights: Tuple of (shadow_weight, rough_weight, boulder_weight). Must sum to 1.
        
    Returns:
        Hazard map array (0 = safe, 1 = dangerous)
    """
    logger.info("Computing unified Terrain Hazard Score Map")
    
    # 1. Normalize shadow score (0 or 1)
    shadow_score = shadow_mask.astype(float)
    
    # 2. Normalize roughness to [0, 1]
    r_min, r_max = roughness.min(), roughness.max()
    if r_max > r_min:
        rough_score = (roughness - r_min) / (r_max - r_min)
    else:
        rough_score = np.zeros_like(roughness)
        
    # 3. Create boulder density/presence score
    boulder_score = np.zeros_like(shadow_mask, dtype=float)
    h, w = shadow_mask.shape
    
    # Mark boulder locations with a footprint (radius * 2 for safety margin)
    for b in boulders:
        y, x, r = b
        safety_radius = int(r * 2)
        y_min = max(0, int(y) - safety_radius)
        y_max = min(h, int(y) + safety_radius)
        x_min = max(0, int(x) - safety_radius)
        x_max = min(w, int(x) + safety_radius)
        boulder_score[y_min:y_max, x_min:x_max] = 1.0
        
    # 4. Weighted combination
    w_shadow, w_rough, w_boulder = weights
    hazard = (w_shadow * shadow_score) + (w_rough * rough_score) + (w_boulder * boulder_score)
    
    # Ensure bounds
    hazard = np.clip(hazard, 0, 1)
    
    return hazard


def visualize_analytics(
    image: np.ndarray,
    shadow_mask: np.ndarray,
    roughness: np.ndarray,
    boulders: np.ndarray,
    hazard_map: np.ndarray,
    save_path: str = None
):
    """
    Generate a comprehensive plot of all OHRC analytics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Top-Left: Original + Boulders
    ax = axes[0, 0]
    ax.imshow(image, cmap='gray')
    for b in boulders[:500]:  # Plot up to 500 boulders to avoid clutter
        y, x, r = b
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.0)
        ax.add_patch(circle)
    ax.set_title(f'OHRC Image & Detected Boulders ({len(boulders)} total)', fontsize=14)
    ax.axis('off')
    
    # Top-Right: Shadow Mask
    ax = axes[0, 1]
    ax.imshow(image, cmap='gray')
    ax.imshow(shadow_mask, cmap='Blues', alpha=0.5)
    ax.set_title('Shadow Mask (PSR Proxy)', fontsize=14)
    ax.axis('off')
    
    # Bottom-Left: Roughness
    ax = axes[1, 0]
    im_r = ax.imshow(roughness, cmap='hot')
    ax.set_title('Surface Roughness (Local StdDev)', fontsize=14)
    plt.colorbar(im_r, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # Bottom-Right: Hazard Map
    ax = axes[1, 1]
    im_h = ax.imshow(hazard_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title('Terrain Hazard Map (Safe=Green, Danger=Red)', fontsize=14)
    plt.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04, label='Hazard Score')
    ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved analytics visualization to {save_path}")
    plt.show()
