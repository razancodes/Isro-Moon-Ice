"""
LunarFM Demo: End-to-End Inference Pipeline
=============================================

This script demonstrates the complete LunarFM inference pipeline:
1. Load the pretrained MultiMAE model from checkpoint
2. Process sample Clementine chips (sanity check on known-good data)
3. Extract embeddings from the frozen encoder
4. Run similarity search against the 200K precomputed embedding database
5. Visualize embeddings with PCA false-color maps and clustering
6. (When OHRC data available) Process OHRC and run full analysis

Usage:
    python demo_lunarfm.py --data-dir C:/Users/MRaza/Documents/lunarlab-public

Set OHRC_PATH environment variable to also process OHRC data:
    set OHRC_PATH=path/to/ohrc.tif
    python demo_lunarfm.py --data-dir C:/Users/MRaza/Documents/lunarlab-public
"""

import sys
import os
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch

# Add project roots to path
PROJECT_ROOT = Path(__file__).parent
LUNARFM_SRC = PROJECT_ROOT / 'LunarFM-Science-Release' / 'src'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LUNARFM_SRC))

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='LunarFM Demo Pipeline')
    parser.add_argument(
        '--data-dir', type=str,
        default=r'C:\Users\MRaza\Documents\lunarlab-public',
        help='Path to the lunarlab-public data release directory'
    )
    parser.add_argument(
        '--ohrc-path', type=str, default=None,
        help='Path to OHRC GeoTIFF (optional)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: cpu, cuda, or auto'
    )
    parser.add_argument(
        '--n-sample-chips', type=int, default=50,
        help='Number of sample Clementine chips to process for sanity check'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save outputs (default: <project>/outputs/)'
    )
    parser.add_argument(
        '--skip-similarity', action='store_true',
        help='Skip similarity search (saves time if just testing embeddings)'
    )
    return parser.parse_args()


def detect_device(preference: str = 'auto') -> str:
    """Detect the best available device."""
    if preference == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("No CUDA GPU detected, using CPU")
    else:
        device = preference
    return device


def step1_load_model(data_dir: str, device: str):
    """Step 1: Load the pretrained LunarFM model."""
    from lunarfm_pipeline.model_loader import load_lunarfm_model
    
    checkpoint_path = os.path.join(data_dir, 'model', 'last.ckpt')
    config_path = os.path.join(data_dir, 'model', 'config.yaml')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    logger.info("=" * 60)
    logger.info("STEP 1: Loading pretrained LunarFM model")
    logger.info("=" * 60)
    
    model = load_lunarfm_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        eval_mode=True,
    )
    
    return model


def step2_process_sample_chips(data_dir: str, n_chips: int = 50):
    """Step 2: Load and preprocess sample Clementine chips."""
    from lunarfm_pipeline.preprocessing import prepare_sample_chip
    
    logger.info("=" * 60)
    logger.info("STEP 2: Processing sample Clementine UVVIS chips")
    logger.info("=" * 60)
    
    chips_dir = os.path.join(
        data_dir, 'chips', 'lunar_grid_halfdegree',
        'Lunar_Clementine_UVVIS_750nm_Global_Mosaic_118m_v2'
    )
    
    chip_files = sorted(glob(os.path.join(chips_dir, '*.tif')))
    # Filter out macOS ._ metadata files
    chip_files = [f for f in chip_files if not os.path.basename(f).startswith('._')]
    
    if not chip_files:
        raise FileNotFoundError(f"No chip files found in {chips_dir}")
    
    logger.info(f"Found {len(chip_files)} chip files. Processing first {n_chips}...")
    
    selected_files = chip_files[:n_chips]
    
    # Process each chip
    tiled_scenes = []
    chip_ids = []
    for chip_file in selected_files:
        try:
            tiled = prepare_sample_chip(
                chip_file, 
                modality='ClementineUVVISMosaic',
                resize=112,
            )
            tiled_scenes.append(tiled)
            chip_id = Path(chip_file).stem
            chip_ids.append(chip_id)
        except Exception as e:
            logger.warning(f"Failed to process {chip_file}: {e}")
    
    logger.info(f"Successfully processed {len(tiled_scenes)} chips")
    
    return tiled_scenes, chip_ids


def step3_extract_embeddings(model, tiled_scenes, chip_ids, device: str):
    """Step 3: Extract embeddings from all processed chips."""
    from lunarfm_pipeline.embeddings import extract_embeddings
    
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting embeddings from LunarFM encoder")
    logger.info("=" * 60)
    
    all_embeddings = []
    
    for i, (tiled, chip_id) in enumerate(zip(tiled_scenes, chip_ids)):
        result = extract_embeddings(
            model=model,
            tiled_scene=tiled,
            batch_size=1,
            device=device,
            return_spatial=False,
            pooling='global_token',
        )
        all_embeddings.append(result.global_embeddings[0])
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(tiled_scenes)} chips")
    
    embeddings_matrix = np.stack(all_embeddings)  # [N_chips, 768]
    
    logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
    logger.info(f"  Mean norm: {np.linalg.norm(embeddings_matrix, axis=1).mean():.4f}")
    logger.info(f"  Std norm: {np.linalg.norm(embeddings_matrix, axis=1).std():.4f}")
    
    return embeddings_matrix


def step4_similarity_search(
    embeddings: np.ndarray,
    chip_ids: list,
    data_dir: str,
    top_k: int = 5,
):
    """Step 4: Search the 200K precomputed embeddings for similar chips."""
    from lunarfm_pipeline.similarity import load_precomputed_embeddings, search_similar
    
    logger.info("=" * 60)
    logger.info("STEP 4: Similarity search against 200K lunar embeddings")
    logger.info("=" * 60)
    
    parquet_path = os.path.join(
        data_dir, 'embeddings',
        'lunar_grid_halfdegree_standardized_chiplevel_embeddings.parquet'
    )
    
    db_embeddings, db_metadata = load_precomputed_embeddings(parquet_path)
    
    # Search for top-K similar chips for each of our embeddings
    results = search_similar(
        query_embeddings=embeddings,
        database_embeddings=db_embeddings,
        database_metadata=db_metadata,
        top_k=top_k,
    )
    
    # Print some results
    for i in range(min(3, len(results))):
        r = results[i]
        logger.info(f"\nQuery chip '{chip_ids[r.query_idx]}':")
        for j in range(min(top_k, len(r.top_k_scores))):
            logger.info(f"  Top-{j+1}: score={r.top_k_scores[j]:.4f}, "
                       f"chip_id={r.top_k_chip_ids[j]}")
    
    return results, db_embeddings, db_metadata


def step5_visualize(
    embeddings: np.ndarray, 
    chip_ids: list,
    output_dir: str,
):
    """Step 5: Visualize embeddings with PCA and clustering."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    logger.info("=" * 60)
    logger.info("STEP 5: Visualizing embedding space")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For chip-level analysis, we create a simple EmbeddingResult
    from lunarfm_pipeline.embeddings import EmbeddingResult
    from lunarfm_pipeline.preprocessing import PatchInfo
    
    # Create a grid layout (arrange chips in a square grid for visualization)
    n = len(embeddings)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    
    patch_infos = [
        PatchInfo(
            patch_idx=i, row_idx=i // n_cols, col_idx=i % n_cols,
            y_start=0, x_start=0, y_end=112, x_end=112,
            nan_fraction=0.0,
        )
        for i in range(n)
    ]
    
    emb_result = EmbeddingResult(
        global_embeddings=embeddings,
        spatial_embeddings=None,
        patch_infos=patch_infos,
        grid_shape=(n_rows, n_cols),
        original_shape=(n_rows * 112, n_cols * 112),
    )
    
    # PCA visualization
    from lunarfm_pipeline.visualization import pca_false_color_map, cluster_and_map
    
    logger.info("Creating PCA false-color map...")
    pca_path = os.path.join(output_dir, 'pca_false_color_map.png')
    pca_false_color_map(
        emb_result,
        title="LunarFM Embeddings — PCA False Color (Sample Chips)",
        save_path=pca_path,
    )
    
    # Clustering
    logger.info("Creating cluster map...")
    cluster_path = os.path.join(output_dir, 'cluster_map.png')
    labels, kmeans = cluster_and_map(
        emb_result,
        n_clusters=6,
        title="Unsupervised Clustering of LunarFM Embeddings",
        save_path=cluster_path,
    )
    
    logger.info(f"Outputs saved to {output_dir}")
    
    return emb_result


def step6_process_ohrc(model, ohrc_path: str, output_dir: str, device: str):
    """Step 6: Process OHRC data (when available)."""
    from lunarfm_pipeline.preprocessing import prepare_ohrc_for_lunarfm, load_image_generic
    from lunarfm_pipeline.embeddings import extract_embeddings
    from lunarfm_pipeline.visualization import (
        pca_false_color_map, umap_embedding_plot, 
        cluster_and_map, similarity_heatmap
    )
    from lunarfm_pipeline.ohrc_analytics import (
        compute_shadow_mask, compute_roughness, detect_boulders,
        compute_hazard_map, visualize_analytics
    )
    
    logger.info("=" * 60)
    logger.info("STEP 6: Processing OHRC data")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Classical CV & Photogrammetry Analytics
    logger.info("Running classical OHRC analytics (Shadows, Roughness, Boulders)")
    
    # Load raw image for analytics
    try:
        if ohrc_path.endswith('.xml'):
            from lunarfm_pipeline.ohrc_analytics import load_ohrc_pds4
            raw_image = load_ohrc_pds4(ohrc_path)
        else:
            raw_image = load_image_generic(ohrc_path)[0]  # Take first channel
            
        # Normalize to 0-1 for analytics
        img_min, img_max = np.nanmin(raw_image), np.nanmax(raw_image)
        norm_image = (raw_image - img_min) / (img_max - img_min)
        norm_image = np.nan_to_num(norm_image, nan=0.0)
        
        shadow_mask = compute_shadow_mask(norm_image, threshold_percentile=5.0)
        roughness = compute_roughness(raw_image, window_size=20)
        boulders = detect_boulders(norm_image, downsample_factor=4)
        hazard = compute_hazard_map(shadow_mask, roughness, boulders)
        
        visualize_analytics(
            norm_image, shadow_mask, roughness, boulders, hazard,
            save_path=os.path.join(output_dir, 'ohrc_hazard_analytics.png')
        )
        logger.info("Classical analytics complete!")
    except Exception as e:
        logger.error(f"Failed to run classical analytics: {e}")
    
    # 2. LunarFM Embedding Extraction
    logger.info("Running LunarFM Embedding Pipeline on OHRC")
    
    # Prepare OHRC data
    tiled = prepare_ohrc_for_lunarfm(
        filepath=ohrc_path,
        patch_size=112,
        overlap=0,
        modality='ClementineUVVISMosaic',
    )
    
    logger.info(f"OHRC tiled into {len(tiled.patches)} patches "
                f"(grid: {tiled.grid_shape})")
    
    # Extract embeddings
    result = extract_embeddings(
        model=model,
        tiled_scene=tiled,
        batch_size=16,
        device=device,
        return_spatial=True,
        pooling='global_token',
    )
    
    # Visualize
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA false color
    pca_false_color_map(
        result,
        title="OHRC Crater — LunarFM Embedding PCA",
        save_path=os.path.join(output_dir, 'ohrc_pca.png'),
    )
    
    # Clustering
    cluster_and_map(
        result,
        n_clusters=6,
        title="OHRC Crater — Unsupervised Terrain Segmentation",
        save_path=os.path.join(output_dir, 'ohrc_clusters.png'),
    )
    
    # UMAP
    try:
        umap_embedding_plot(
            result,
            color_by='cluster',
            title="OHRC Crater — UMAP Embedding Space",
            save_path=os.path.join(output_dir, 'ohrc_umap.png'),
        )
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP visualization")
    
    # Similarity heatmap from center patch
    center_idx = len(result.global_embeddings) // 2
    similarity_heatmap(
        result,
        query_idx=center_idx,
        title="OHRC — Similarity to Center Patch",
        save_path=os.path.join(output_dir, 'ohrc_similarity.png'),
    )
    
    return result


def main():
    args = parse_args()
    
    device = detect_device(args.device)
    
    output_dir = args.output_dir or os.path.join(str(PROJECT_ROOT), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("LunarFM Inference Pipeline — ISRO BAH Hackathon")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load model
    model = step1_load_model(args.data_dir, device)
    
    # Step 2: Process sample chips
    tiled_scenes, chip_ids = step2_process_sample_chips(
        args.data_dir, n_chips=args.n_sample_chips
    )
    
    # Step 3: Extract embeddings
    embeddings = step3_extract_embeddings(model, tiled_scenes, chip_ids, device)
    
    # Step 4: Similarity search (optional)
    if not args.skip_similarity:
        try:
            sim_results, db_embeddings, db_metadata = step4_similarity_search(
                embeddings, chip_ids, args.data_dir
            )
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            logger.info("Continuing without similarity search...")
    
    # Step 5: Visualize
    emb_result = step5_visualize(embeddings, chip_ids, output_dir)
    
    # Step 6: Process OHRC (if path provided)
    ohrc_path = args.ohrc_path or os.environ.get('OHRC_PATH')
    if ohrc_path and os.path.exists(ohrc_path):
        ohrc_output = os.path.join(output_dir, 'ohrc')
        ohrc_result = step6_process_ohrc(model, ohrc_path, ohrc_output, device)
        
        # Save OHRC embeddings for downstream use
        np.save(
            os.path.join(ohrc_output, 'ohrc_embeddings.npy'),
            ohrc_result.global_embeddings,
        )
        logger.info(f"Saved OHRC embeddings to {ohrc_output}/ohrc_embeddings.npy")
    else:
        logger.info("\nNo OHRC data path provided. Skipping OHRC processing.")
        logger.info("To process OHRC, run with: --ohrc-path path/to/ohrc.tif")
    
    # Save sample embeddings
    np.save(os.path.join(output_dir, 'sample_embeddings.npy'), embeddings)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {output_dir}")
    
    return model, embeddings, emb_result


if __name__ == '__main__':
    model, embeddings, emb_result = main()
