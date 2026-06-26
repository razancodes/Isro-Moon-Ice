"""
Embedding Extraction Module
=============================

Extract spatial embeddings from the frozen LunarFM encoder.

The MultiMAE encoder processes tokens from all provided modalities through a 
shared transformer. For embedding extraction, we:

1. Pass data through forward_encoder with mask_inputs=False (no random masking)
2. The encoder produces tokens for each input patch plus 1 global CLS token
3. The global token (last in the sequence) serves as the chip-level embedding
4. Alternatively, mean-pool all patch tokens for a spatial average embedding
5. For spatial (patch-level) embeddings, we keep per-patch tokens

The key method is MultiMAE.get_embeddings_and_reconstruction() with encode_only=True,
or we can call forward_encoder() directly for more control.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm

from .preprocessing import TiledScene


@dataclass
class EmbeddingResult:
    """
    Container for extracted embeddings.
    
    Attributes:
        global_embeddings: [N_patches, 768] — one embedding per input patch
                          (from the CLS/global token)
        spatial_embeddings: [N_patches, n_patch_tokens, 768] — per-patch-token
                           embeddings (optional, for spatial maps within each patch)
        patch_infos: List of PatchInfo from the TiledScene
        grid_shape: (n_rows, n_cols) from the TiledScene
        original_shape: (H, W) from the TiledScene
    """
    global_embeddings: np.ndarray
    spatial_embeddings: Optional[np.ndarray]
    patch_infos: list
    grid_shape: tuple
    original_shape: tuple


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    tiled_scene: TiledScene,
    batch_size: int = 16,
    device: str = 'cpu',
    return_spatial: bool = False,
    pooling: str = 'global_token',
) -> EmbeddingResult:
    """
    Extract embeddings from LunarFM for all patches in a TiledScene.
    
    Args:
        model: Loaded MultiMAE model (in eval mode)
        tiled_scene: Preprocessed and normalized TiledScene
        batch_size: Batch size for inference
        device: Device for inference ('cpu' or 'cuda')
        return_spatial: If True, also return per-patch-token spatial embeddings
        pooling: How to compute the global embedding:
                 - 'global_token': Use the CLS/global token (last token)
                 - 'mean_pool': Mean-pool all patch tokens (excluding global token)
                 - 'both': Return concatenation of global_token and mean_pool (1536-dim)
    
    Returns:
        EmbeddingResult with extracted embeddings
    """
    model.eval()
    modality = tiled_scene.modality_name
    
    if not modality:
        raise ValueError("TiledScene has no modality_name set. "
                         "Run normalize_for_lunarfm() first.")
    
    logger.info(f"Extracting embeddings for {len(tiled_scene.patches)} patches "
                f"using modality '{modality}', pooling='{pooling}'")
    
    all_global = []
    all_spatial = [] if return_spatial else None
    
    n_patches = len(tiled_scene.patches)
    
    for start_idx in tqdm(range(0, n_patches, batch_size), 
                          desc="Extracting embeddings", 
                          disable=n_patches <= batch_size):
        end_idx = min(start_idx + batch_size, n_patches)
        
        # Get batch of patches and NaN masks
        batch_patches = tiled_scene.patches[start_idx:end_idx].to(device)
        batch_nan_masks = tiled_scene.nan_masks[start_idx:end_idx].to(device)
        
        # Build input dict: {modality_name: tensor}
        # This is how MultiMAE.forward_encoder expects its input
        x = {modality: batch_patches}
        nan_masks = {modality: batch_nan_masks}
        
        # Run encoder (no masking — we want all tokens)
        encoder_tokens, task_masks, input_info, ids_keep, ids_restore = \
            model.forward_encoder(
                x=x,
                nan_masks=nan_masks,
                mask_inputs=False,  # No random masking — keep all tokens
            )
        
        # encoder_tokens shape: [B, num_tokens + num_global_tokens, dim_tokens]
        # The last `num_global_tokens` positions are the global/CLS tokens
        # For this model: num_global_tokens = 1
        
        num_global_tokens = model.num_global_tokens  # Typically 1
        
        if pooling == 'global_token':
            # Last token(s) are global tokens
            global_emb = encoder_tokens[:, -num_global_tokens:, :].mean(dim=1)  # [B, 768]
        elif pooling == 'mean_pool':
            # Mean pool patch tokens (exclude global tokens)
            patch_tokens = encoder_tokens[:, :-num_global_tokens, :]  # [B, n_patches, 768]
            global_emb = patch_tokens.mean(dim=1)  # [B, 768]
        elif pooling == 'both':
            global_token = encoder_tokens[:, -num_global_tokens:, :].mean(dim=1)
            mean_pool = encoder_tokens[:, :-num_global_tokens, :].mean(dim=1)
            global_emb = torch.cat([global_token, mean_pool], dim=-1)  # [B, 1536]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        all_global.append(global_emb.cpu().numpy())
        
        if return_spatial:
            # Spatial tokens (excluding global token)
            spatial = encoder_tokens[:, :-num_global_tokens, :].cpu().numpy()
            all_spatial.append(spatial)
    
    global_embeddings = np.concatenate(all_global, axis=0)  # [N, 768] or [N, 1536]
    
    spatial_embeddings = None
    if return_spatial and all_spatial:
        spatial_embeddings = np.concatenate(all_spatial, axis=0)
    
    logger.info(f"Extracted {global_embeddings.shape[0]} global embeddings "
                f"of dim {global_embeddings.shape[1]}")
    
    if spatial_embeddings is not None:
        logger.info(f"Also extracted spatial embeddings: {spatial_embeddings.shape}")
    
    return EmbeddingResult(
        global_embeddings=global_embeddings,
        spatial_embeddings=spatial_embeddings,
        patch_infos=tiled_scene.patch_infos,
        grid_shape=tiled_scene.grid_shape,
        original_shape=tiled_scene.original_shape,
    )


@torch.no_grad()
def extract_embeddings_from_chips(
    model: torch.nn.Module,
    chip_paths: List[str],
    modality: str,
    device: str = 'cpu',
    resize: int = 112,
    pooling: str = 'global_token',
) -> np.ndarray:
    """
    Extract embeddings from a list of individual chip files.
    
    Convenience function for processing multiple single-chip files
    (e.g., from the LunarFM data release sample chips).
    
    Args:
        model: Loaded MultiMAE model
        chip_paths: List of paths to individual chip .tif files
        modality: Data group name for normalization
        device: Device for inference
        resize: Resize chips to this size
        pooling: Pooling method for global embedding
        
    Returns:
        Numpy array of shape [N_chips, embed_dim]
    """
    from .preprocessing import prepare_sample_chip
    
    model.eval()
    all_embeddings = []
    
    for chip_path in tqdm(chip_paths, desc="Processing chips"):
        tiled = prepare_sample_chip(chip_path, modality=modality, resize=resize)
        
        result = extract_embeddings(
            model=model,
            tiled_scene=tiled,
            batch_size=1,
            device=device,
            return_spatial=False,
            pooling=pooling,
        )
        
        all_embeddings.append(result.global_embeddings)
    
    return np.concatenate(all_embeddings, axis=0)
