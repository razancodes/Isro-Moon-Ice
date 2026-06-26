"""
Visualization Module
=====================

Visualize LunarFM embeddings in interpretable ways:

1. PCA false-color maps: Project 768-dim embeddings to 3 principal components,
   map to RGB channels, and reconstruct the spatial layout of the scene.
   
2. UMAP 2D/3D: Nonlinear dimensionality reduction for scatter plots and 
   interactive exploration of the embedding manifold.
   
3. K-means clustering: Unsupervised terrain segmentation using embedding space.
   Reconstruct cluster assignments to a spatial map.

4. Similarity heatmaps: Visualize which patches are most similar to a selected
   query patch.
"""

import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from loguru import logger

from .embeddings import EmbeddingResult


def pca_false_color_map(
    embedding_result: EmbeddingResult,
    n_components: int = 3,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "LunarFM Embedding PCA — Terrain Structure",
    save_path: Optional[str] = None,
    return_components: bool = False,
) -> Optional[np.ndarray]:
    """
    Create a false-color spatial map by projecting embeddings to 3 PCA components
    and mapping them to RGB channels.
    
    Each patch position in the grid gets colored according to its first 3 principal
    components. Regions with similar terrain/morphology will appear in similar colors.
    
    Args:
        embedding_result: EmbeddingResult from extract_embeddings
        n_components: Number of PCA components (3 for RGB)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the figure
        return_components: If True, return the PCA-projected components
        
    Returns:
        If return_components: PCA-projected array [N_patches, n_components]
        Otherwise: None
    """
    embeddings = embedding_result.global_embeddings
    grid_shape = embedding_result.grid_shape
    patch_infos = embedding_result.patch_infos
    
    logger.info(f"Computing PCA ({n_components} components) on {len(embeddings)} embeddings")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    logger.info(f"  Total explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Normalize each component to [0, 1] for RGB mapping
    scaler = MinMaxScaler()
    components_norm = scaler.fit_transform(components)
    
    # Reconstruct spatial grid
    n_rows, n_cols = grid_shape
    rgb_map = np.full((n_rows, n_cols, 3), 0.5)  # Gray for missing patches
    
    for info, color in zip(patch_infos, components_norm[:, :3]):
        if info.row_idx < n_rows and info.col_idx < n_cols:
            rgb_map[info.row_idx, info.col_idx, :] = color[:3]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, 
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # Main PCA map
    ax = axes[0]
    ax.imshow(rgb_map, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    
    # Add variance explained annotation
    var_text = '\n'.join([
        f'PC{i+1}: {v:.1%}' 
        for i, v in enumerate(pca.explained_variance_ratio_[:3])
    ])
    ax.text(0.02, 0.98, f'Explained Variance:\n{var_text}', 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Individual component maps
    ax = axes[1]
    component_maps = []
    for i in range(min(3, n_components)):
        comp_map = np.full((n_rows, n_cols), np.nan)
        for info, val in zip(patch_infos, components[:, i]):
            if info.row_idx < n_rows and info.col_idx < n_cols:
                comp_map[info.row_idx, info.col_idx] = val
        component_maps.append(comp_map)
    
    # Stack component visualizations vertically
    ax.set_visible(False)
    
    # Create sub-axes for individual components
    gs = fig.add_gridspec(3, 4)
    for i, comp_map in enumerate(component_maps):
        sub_ax = fig.add_subplot(gs[i, 3])
        im = sub_ax.imshow(comp_map, cmap='viridis', interpolation='nearest')
        sub_ax.set_title(f'PC{i+1}', fontsize=10)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        plt.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PCA map to {save_path}")
    
    plt.show()
    
    if return_components:
        return components
    return None


def umap_embedding_plot(
    embedding_result: EmbeddingResult,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    color_by: str = 'position',
    figsize: Tuple[int, int] = (10, 8),
    title: str = "LunarFM Embedding Space (UMAP)",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a UMAP scatter plot of embeddings.
    
    Args:
        embedding_result: EmbeddingResult from extract_embeddings
        n_components: UMAP dimensions (2 or 3)
        n_neighbors: UMAP neighborhood size
        min_dist: UMAP minimum distance
        color_by: How to color points:
                  - 'position': Color by spatial position (row, col)
                  - 'cluster': Color by K-means cluster
                  - 'nan_fraction': Color by NaN fraction
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        UMAP-projected coordinates [N_patches, n_components]
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        raise
    
    embeddings = embedding_result.global_embeddings
    patch_infos = embedding_result.patch_infos
    
    logger.info(f"Computing UMAP ({n_components}D) on {len(embeddings)} embeddings")
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    
    umap_coords = reducer.fit_transform(embeddings)
    
    # Determine coloring
    if color_by == 'position':
        # Color by spatial position (combine row and col into hue)
        rows = np.array([info.row_idx for info in patch_infos])
        cols = np.array([info.col_idx for info in patch_infos])
        colors = np.stack([rows / max(rows.max(), 1), 
                          cols / max(cols.max(), 1), 
                          np.zeros_like(rows)], axis=1)
        cmap_label = 'Spatial Position'
    elif color_by == 'cluster':
        n_clusters = min(8, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        colors = labels
        cmap_label = 'Cluster'
    elif color_by == 'nan_fraction':
        colors = np.array([info.nan_fraction for info in patch_infos])
        cmap_label = 'NaN Fraction'
    else:
        colors = np.arange(len(embeddings))
        cmap_label = 'Index'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if n_components == 2:
        scatter = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=colors if isinstance(colors, np.ndarray) and colors.ndim == 1 else None,
            cmap='tab10' if color_by == 'cluster' else 'viridis',
            s=15, alpha=0.7,
        )
        if isinstance(colors, np.ndarray) and colors.ndim == 1:
            plt.colorbar(scatter, ax=ax, label=cmap_label)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {save_path}")
    
    plt.show()
    
    return umap_coords


def cluster_and_map(
    embedding_result: EmbeddingResult,
    n_clusters: int = 6,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Unsupervised Terrain Segmentation (K-Means on LunarFM Embeddings)",
    save_path: Optional[str] = None,
    cmap: str = 'Set2',
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering on embeddings and visualize as a spatial map.
    
    This produces an unsupervised terrain segmentation of the scene.
    Different clusters should correspond to different geomorphological units
    (e.g., crater floor, rim, ejecta, smooth terrain, rough terrain).
    
    Args:
        embedding_result: EmbeddingResult from extract_embeddings
        n_clusters: Number of clusters
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        cmap: Colormap for clusters
        
    Returns:
        Tuple of (cluster_labels, fitted_kmeans_model)
    """
    embeddings = embedding_result.global_embeddings
    grid_shape = embedding_result.grid_shape
    patch_infos = embedding_result.patch_infos
    
    logger.info(f"K-Means clustering with {n_clusters} clusters on {len(embeddings)} embeddings")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Reconstruct spatial cluster map
    n_rows, n_cols = grid_shape
    cluster_map = np.full((n_rows, n_cols), -1, dtype=int)
    
    for info, label in zip(patch_infos, labels):
        if info.row_idx < n_rows and info.col_idx < n_cols:
            cluster_map[info.row_idx, info.col_idx] = label
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, 
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # Cluster map
    ax = axes[0]
    masked_map = np.ma.masked_where(cluster_map == -1, cluster_map)
    im = ax.imshow(masked_map, cmap=cmap, interpolation='nearest', 
                   vmin=0, vmax=n_clusters-1)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    plt.colorbar(im, ax=ax, label='Cluster', ticks=range(n_clusters))
    
    # Cluster statistics
    ax = axes[1]
    ax.set_visible(False)
    
    # Add text with cluster stats
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    stats_text = "Cluster Statistics:\n" + "-" * 25 + "\n"
    for i in range(n_clusters):
        stats_text += f"Cluster {i}: {cluster_counts[i]} patches\n"
    
    fig.text(0.78, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster map to {save_path}")
    
    plt.show()
    
    return labels, kmeans


def similarity_heatmap(
    embedding_result: EmbeddingResult,
    query_idx: int,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Show which patches are most similar to a selected query patch.
    
    Creates a heatmap of cosine similarity from the query patch to all others.
    
    Args:
        embedding_result: EmbeddingResult from extract_embeddings
        query_idx: Index of the query patch
        figsize: Figure size  
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Similarity scores for all patches
    """
    from sklearn.preprocessing import normalize
    
    embeddings = embedding_result.global_embeddings
    grid_shape = embedding_result.grid_shape
    patch_infos = embedding_result.patch_infos
    
    # Compute cosine similarity from query to all
    emb_norm = normalize(embeddings, norm='l2')
    query_emb = emb_norm[query_idx:query_idx+1]
    similarities = (emb_norm @ query_emb.T).flatten()
    
    # Reconstruct spatial heatmap
    n_rows, n_cols = grid_shape
    sim_map = np.full((n_rows, n_cols), np.nan)
    
    for info, sim in zip(patch_infos, similarities):
        if info.row_idx < n_rows and info.col_idx < n_cols:
            sim_map[info.row_idx, info.col_idx] = sim
    
    # Mark query position
    query_info = patch_infos[query_idx]
    
    if title is None:
        title = f"Cosine Similarity to Patch ({query_info.row_idx}, {query_info.col_idx})"
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    
    # Mark query position
    ax.plot(query_info.col_idx, query_info.row_idx, 'c*', markersize=15, 
            markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved similarity heatmap to {save_path}")
    
    plt.show()
    
    return similarities
