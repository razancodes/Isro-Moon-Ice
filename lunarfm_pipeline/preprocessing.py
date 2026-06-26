"""
Preprocessing Module
=====================

Handles conversion of external imagery (OHRC, or any single-channel GeoTIFF)
into the format expected by LunarFM's MultiMAE encoder.

Key responsibilities:
- Load GeoTIFF/image data
- Tile large scenes into 112x112 patches (matching LunarFM's training resolution)
- Apply per-modality z-score normalization using training statistics
- Generate NaN masks for missing data regions
- Track spatial coordinates for each patch for later reconstruction
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass, field

import torch
from torchvision import transforms as T
from loguru import logger

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available — GeoTIFF loading disabled")


@dataclass
class PatchInfo:
    """Metadata for a single patch extracted from a larger scene."""
    patch_idx: int
    row_idx: int          # Row position in the grid of patches
    col_idx: int          # Column position in the grid of patches
    y_start: int          # Pixel y-start in original image
    x_start: int          # Pixel x-start in original image
    y_end: int            # Pixel y-end in original image
    x_end: int            # Pixel x-end in original image
    nan_fraction: float   # Fraction of NaN pixels in this patch
    
    # Optional geospatial info (populated if GeoTIFF has transform)
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None


@dataclass
class TiledScene:
    """
    Container for a tiled scene ready for LunarFM inference.
    
    Attributes:
        patches: Tensor of shape [N, C, H, W] — normalized patches
        nan_masks: Tensor of shape [N, C, H, W] — True where NaN
        patch_infos: List of PatchInfo metadata for each patch
        original_shape: (H, W) of the original image before tiling
        grid_shape: (n_rows, n_cols) of the patch grid
        modality_name: Which LunarFM data_group this maps to
    """
    patches: torch.Tensor
    nan_masks: torch.Tensor
    patch_infos: List[PatchInfo]
    original_shape: Tuple[int, int]
    grid_shape: Tuple[int, int]
    modality_name: str


def load_geotiff(filepath: str) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Load a GeoTIFF file and return the image data and geospatial metadata.
    
    Args:
        filepath: Path to the GeoTIFF file
        
    Returns:
        Tuple of:
        - image: numpy array of shape (C, H, W) or (H, W)
        - geo_meta: dict with 'transform', 'crs', 'bounds', or None
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF loading")
    
    with rasterio.open(filepath) as src:
        image = src.read()  # Shape: (bands, H, W)
        geo_meta = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
        }
    
    logger.info(f"Loaded GeoTIFF: {filepath}")
    logger.info(f"  Shape: {image.shape}, dtype: {image.dtype}")
    logger.info(f"  Bounds: {geo_meta['bounds']}")
    logger.info(f"  NoData value: {geo_meta['nodata']}")
    
    return image, geo_meta


def load_image_generic(filepath: str) -> np.ndarray:
    """
    Load an image from common formats (TIFF, PNG, NPY, etc.)
    Returns array of shape (C, H, W) as float32.
    """
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    
    if ext in ('.tif', '.tiff'):
        image, _ = load_geotiff(filepath)
    elif ext == '.npy':
        image = np.load(filepath)
    elif ext == '.npz':
        data = np.load(filepath)
        image = data[list(data.keys())[0]]
    else:
        # Try PIL/imageio as fallback
        from PIL import Image
        img = Image.open(filepath)
        image = np.array(img)
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # Add channel dim
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)  # HWC -> CHW
    
    image = image.astype(np.float32)
    
    # Ensure 3D: (C, H, W)
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    
    return image


def tile_image(
    image: np.ndarray,
    patch_size: int = 112,
    overlap: int = 0,
    min_valid_fraction: float = 0.1,
    geo_meta: Optional[dict] = None,
) -> TiledScene:
    """
    Tile a large image into non-overlapping (or overlapping) patches.
    
    Args:
        image: Array of shape (C, H, W), float32. May contain NaN.
        patch_size: Size of each square patch (default 112, matching LunarFM training)
        overlap: Number of overlapping pixels between adjacent patches
        min_valid_fraction: Minimum fraction of non-NaN pixels to keep a patch
        geo_meta: Optional geospatial metadata from load_geotiff
        
    Returns:
        TiledScene with patches, NaN masks, and metadata
    """
    C, H, W = image.shape
    stride = patch_size - overlap
    
    # Compute grid dimensions
    n_rows = max(1, (H - patch_size) // stride + 1)
    n_cols = max(1, (W - patch_size) // stride + 1)
    
    # If image is smaller than patch_size, pad it
    if H < patch_size or W < patch_size:
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 
                       mode='constant', constant_values=np.nan)
        logger.info(f"Padded image from ({C},{H},{W}) to {image.shape}")
        C, H, W = image.shape
        n_rows, n_cols = 1, 1
    
    patches = []
    nan_masks = []
    patch_infos = []
    patch_idx = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Clamp to image bounds
            y_end = min(y_end, H)
            x_end = min(x_end, W)
            y_start = y_end - patch_size  # Adjust start to maintain patch_size
            x_start = x_end - patch_size
            
            patch = image[:, y_start:y_end, x_start:x_end].copy()
            
            # Create NaN mask (True = NaN = invalid)
            nan_mask = np.isnan(patch)
            nan_fraction = nan_mask.sum() / nan_mask.size
            
            # Skip patches that are mostly NaN
            if nan_fraction > (1.0 - min_valid_fraction):
                continue
            
            # Compute center coordinates if geo_meta available
            center_lon, center_lat = None, None
            if geo_meta and geo_meta.get('transform'):
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                center_lon, center_lat = geo_meta['transform'] * (center_x, center_y)
            
            info = PatchInfo(
                patch_idx=patch_idx,
                row_idx=row,
                col_idx=col,
                y_start=y_start, x_start=x_start,
                y_end=y_end, x_end=x_end,
                nan_fraction=nan_fraction,
                center_lon=center_lon,
                center_lat=center_lat,
            )
            
            patches.append(patch)
            nan_masks.append(nan_mask)
            patch_infos.append(info)
            patch_idx += 1
    
    if not patches:
        raise ValueError(f"No valid patches extracted! Image shape={image.shape}, "
                         f"patch_size={patch_size}, min_valid_fraction={min_valid_fraction}")
    
    patches_tensor = torch.from_numpy(np.stack(patches)).float()
    nan_masks_tensor = torch.from_numpy(np.stack(nan_masks)).float()
    
    logger.info(f"Tiled image ({C},{H},{W}) into {len(patches)} patches "
                f"of size {patch_size}x{patch_size} (grid: {n_rows}x{n_cols})")
    
    return TiledScene(
        patches=patches_tensor,
        nan_masks=nan_masks_tensor,
        patch_infos=patch_infos,
        original_shape=(H, W),
        grid_shape=(n_rows, n_cols),
        modality_name='',  # Set by normalize_for_lunarfm
    )


def normalize_for_lunarfm(
    tiled_scene: TiledScene,
    modality: str = 'ClementineUVVISMosaic',
    custom_stats: Optional[Dict[str, float]] = None,
) -> TiledScene:
    """
    Apply z-score normalization matching LunarFM's training normalization.
    
    The model was trained with `normalization: meanstd`, meaning each modality's
    data is normalized as: x_norm = (x - mean) / std using the global statistics
    hardcoded in each dataset class (lrodatasets.py).
    
    NaN values are replaced with 0 AFTER normalization (matching the training
    pipeline's nan_value=0.0 default in AlignedDatasets).
    
    Args:
        tiled_scene: TiledScene from tile_image()
        modality: LunarFM data_group name to use for normalization stats.
                  For OHRC, use 'ClementineUVVISMosaic' (single panchromatic channel).
        custom_stats: Optional dict with 'mean' and 'std' to override defaults.
                      Use this if your OHRC data has a very different value range
                      than Clementine UVVIS and you want to rescale first.
    
    Returns:
        TiledScene with normalized patches and updated modality_name
    """
    from .model_loader import get_normalization_stats
    
    all_stats = get_normalization_stats()
    
    if modality not in all_stats:
        raise ValueError(f"Unknown modality '{modality}'. Available: {list(all_stats.keys())}")
    
    stats = all_stats[modality]
    mean = torch.tensor(stats['mean'], dtype=torch.float32)
    std = torch.tensor(stats['std'], dtype=torch.float32)
    
    if custom_stats:
        mean = torch.tensor([custom_stats['mean']], dtype=torch.float32)
        std = torch.tensor([custom_stats['std']], dtype=torch.float32)
        logger.info(f"Using custom normalization: mean={mean.item():.4f}, std={std.item():.4f}")
    else:
        logger.info(f"Using {modality} normalization: mean={mean.tolist()}, std={std.tolist()}")
    
    # Reshape for broadcasting: [1, C, 1, 1]
    n_channels = len(mean)
    mean = mean.view(1, n_channels, 1, 1)
    std = std.view(1, n_channels, 1, 1)
    
    # Verify channel count matches
    if tiled_scene.patches.shape[1] != n_channels:
        raise ValueError(
            f"Patch has {tiled_scene.patches.shape[1]} channels but "
            f"modality '{modality}' expects {n_channels} channels. "
            f"Channels: {stats['channels']}"
        )
    
    # Apply z-score normalization
    normalized = (tiled_scene.patches - mean) / std
    
    # Replace NaN with 0 (matching training pipeline: nan_value=0.0)
    normalized = torch.nan_to_num(normalized, nan=0.0)
    
    # Update tiled scene
    tiled_scene.patches = normalized
    tiled_scene.modality_name = modality
    
    logger.info(f"Normalized patches: range [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return tiled_scene


def prepare_ohrc_for_lunarfm(
    filepath: str,
    patch_size: int = 112,
    overlap: int = 0,
    min_valid_fraction: float = 0.1,
    modality: str = 'ClementineUVVISMosaic',
    value_scale_factor: Optional[float] = None,
) -> TiledScene:
    """
    End-to-end preprocessing: load OHRC data -> tile -> normalize for LunarFM.
    
    This is the main entry point for preparing OHRC imagery.
    
    Args:
        filepath: Path to OHRC GeoTIFF or image file
        patch_size: Patch size (default 112)
        overlap: Overlap between patches
        min_valid_fraction: Min valid pixel fraction per patch
        modality: LunarFM modality to map to (default: ClementineUVVISMosaic)
        value_scale_factor: Optional factor to scale OHRC pixel values to match
                           the target modality's value range before normalization.
                           E.g., if OHRC is 0-255 uint8 and Clementine is 0-255,
                           set to 1.0 (or None). If OHRC is 0-1 float, set to 255.0.
    
    Returns:
        TiledScene ready for LunarFM inference
    """
    logger.info(f"Preparing OHRC data from: {filepath}")
    
    # Load image
    if filepath.lower().endswith(('.tif', '.tiff')):
        image, geo_meta = load_geotiff(filepath)
    else:
        image = load_image_generic(filepath)
        geo_meta = None
    
    # Handle nodata values
    if geo_meta and geo_meta.get('nodata') is not None:
        nodata = geo_meta['nodata']
        image[image == nodata] = np.nan
        logger.info(f"Replaced nodata value ({nodata}) with NaN")
    
    # Optional value scaling
    if value_scale_factor is not None:
        image = image * value_scale_factor
        logger.info(f"Applied value scale factor: {value_scale_factor}")
    
    logger.info(f"Image value range (before norm): [{np.nanmin(image):.4f}, {np.nanmax(image):.4f}]")
    
    # Tile
    tiled = tile_image(
        image, 
        patch_size=patch_size, 
        overlap=overlap,
        min_valid_fraction=min_valid_fraction,
        geo_meta=geo_meta,
    )
    
    # Normalize for LunarFM
    tiled = normalize_for_lunarfm(tiled, modality=modality)
    
    return tiled


def prepare_sample_chip(
    chip_path: str,
    modality: str = 'ClementineUVVISMosaic',
    resize: int = 112,
) -> TiledScene:
    """
    Load a single LunarFM sample chip (from the data release) and prepare it.
    
    This is useful for sanity-checking the pipeline on known-good data
    before running on OHRC.
    
    Args:
        chip_path: Path to a single chip .tif file
        modality: The data_group this chip belongs to
        resize: Resize to this size (default 112, matching training)
    
    Returns:
        TiledScene with a single 112x112 patch
    """
    logger.info(f"Loading sample chip: {chip_path}")
    
    image, geo_meta = load_geotiff(chip_path)
    C, H, W = image.shape
    
    # Resize to 112x112 (matching training)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, C, H, W]
    if H != resize or W != resize:
        resizer = T.Resize(size=(resize, resize))
        image_tensor = resizer(image_tensor)
    
    # Create NaN mask  
    nan_mask = torch.isnan(image_tensor).float()
    
    # Replace NaN with 0 in the image
    image_tensor = torch.nan_to_num(image_tensor, nan=0.0)
    
    info = PatchInfo(
        patch_idx=0, row_idx=0, col_idx=0,
        y_start=0, x_start=0, y_end=resize, x_end=resize,
        nan_fraction=nan_mask.sum().item() / nan_mask.numel(),
    )
    
    tiled = TiledScene(
        patches=image_tensor,
        nan_masks=nan_mask,
        patch_infos=[info],
        original_shape=(H, W),
        grid_shape=(1, 1),
        modality_name='',
    )
    
    # Apply normalization
    tiled = normalize_for_lunarfm(tiled, modality=modality)
    
    return tiled
