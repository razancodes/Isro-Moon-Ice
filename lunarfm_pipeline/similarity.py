"""
Similarity Search Module
=========================

Search for the most similar lunar regions to our target scene by comparing
embeddings against LunarFM's precomputed 200K chip-level embedding database.

The precomputed embeddings are stored in a parquet file:
    lunar_grid_halfdegree_standardized_chiplevel_embeddings.parquet

This file contains ~200K rows, each with a chip identifier and a 768-dim
embedding vector computed from all modalities.

We can use these to:
1. Find global analogs: Which regions on the Moon look most similar to our crater?
2. Check overlap with known ice candidates: Do our most-similar chips correspond
   to regions with anomalous Mini-RF CPR values?
3. Internal similarity: Which parts of our own scene are most similar to each other?
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .embeddings import EmbeddingResult


@dataclass  
class SimilarityResult:
    """Result of a similarity search query."""
    query_idx: int                   # Index of the query patch
    top_k_indices: np.ndarray        # Indices into the database
    top_k_scores: np.ndarray         # Cosine similarity scores
    top_k_chip_ids: List[str]        # Chip IDs from the database
    top_k_metadata: Optional[pd.DataFrame]  # Full metadata rows


def load_precomputed_embeddings(
    parquet_path: str,
    grid_parquet_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load the precomputed 200K lunar chip embeddings.
    
    The parquet file contains columns including the embedding vector
    and chip identifiers. The exact column format needs to be discovered
    from the actual file.
    
    Args:
        parquet_path: Path to the embeddings parquet file
        grid_parquet_path: Optional path to the lunar grid parquet with geometries
        
    Returns:
        Tuple of:
        - embeddings: numpy array of shape [N, 768]
        - metadata: DataFrame with chip IDs, splits, and optionally geometries
    """
    logger.info(f"Loading precomputed embeddings from {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"Loaded DataFrame with {len(df)} rows, columns: {list(df.columns)}")
    
    # The embeddings may be stored as a list/array in an 'embedding' column
    # or as individual columns. Let's handle both cases.
    if 'embedding' in df.columns:
        embeddings = np.stack(df['embedding'].values)
    else:
        # Try to find embedding columns (they might be numbered)
        embed_cols = [c for c in df.columns if c.startswith('emb_') or c.startswith('dim_')]
        if embed_cols:
            embeddings = df[embed_cols].values.astype(np.float32)
        else:
            # Assume all numeric columns except known metadata columns are embedding dims
            metadata_cols = {'chip_id', 'identifier', 'split', 'geometry', 'centroid_lonlat',
                            'lat', 'lon', 'longitude', 'latitude'}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            embed_cols = [c for c in numeric_cols if c not in metadata_cols]
            
            if len(embed_cols) >= 100:  # Likely embedding dimensions
                embeddings = df[embed_cols].values.astype(np.float32)
                logger.info(f"Inferred {len(embed_cols)} embedding dimensions from numeric columns")
            else:
                raise ValueError(
                    f"Cannot identify embedding columns. Columns: {list(df.columns)}")
    
    logger.info(f"Embedding matrix shape: {embeddings.shape}")
    
    # Build metadata DataFrame (keep non-embedding columns)
    if 'embedding' in df.columns:
        metadata = df.drop(columns=['embedding'])
    else:
        metadata = df[list(set(df.columns) - set(embed_cols) if 'embed_cols' in dir() else [])]
        if metadata.empty:
            metadata = pd.DataFrame({'index': range(len(df))})
    
    # Load grid geometries if available
    if grid_parquet_path and Path(grid_parquet_path).exists():
        try:
            import geopandas as gpd
            grid = gpd.read_parquet(grid_parquet_path)
            logger.info(f"Loaded grid geometries: {len(grid)} entries")
        except Exception as e:
            logger.warning(f"Could not load grid geometries: {e}")
    
    return embeddings, df


def search_similar(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    database_metadata: pd.DataFrame,
    top_k: int = 10,
    metric: str = 'cosine',
) -> List[SimilarityResult]:
    """
    Find the top-K most similar chips in the database for each query embedding.
    
    Args:
        query_embeddings: [N_query, embed_dim] array
        database_embeddings: [N_database, embed_dim] array  
        database_metadata: DataFrame with metadata for database entries
        top_k: Number of most similar results to return
        metric: Similarity metric ('cosine' or 'euclidean')
        
    Returns:
        List of SimilarityResult, one per query
    """
    logger.info(f"Searching {len(query_embeddings)} queries against "
                f"{len(database_embeddings)} database entries (top-{top_k})")
    
    if metric == 'cosine':
        # Normalize for cosine similarity
        query_norm = normalize(query_embeddings, norm='l2')
        db_norm = normalize(database_embeddings, norm='l2')
        
        # Compute all pairwise similarities
        sim_matrix = query_norm @ db_norm.T  # [N_query, N_database]
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(query_embeddings, database_embeddings)
        sim_matrix = -dist_matrix  # Negate so higher = more similar
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    results = []
    
    # Get chip ID column name
    id_col = None
    for candidate in ['chip_id', 'identifier', 'index']:
        if candidate in database_metadata.columns:
            id_col = candidate
            break
    
    for i in range(len(query_embeddings)):
        # Get top-K indices (highest similarity)
        top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
        top_scores = sim_matrix[i][top_indices]
        
        # Get chip IDs
        if id_col and id_col in database_metadata.columns:
            top_chip_ids = database_metadata.iloc[top_indices][id_col].tolist()
        else:
            top_chip_ids = [str(idx) for idx in top_indices]
        
        # Get full metadata rows
        top_metadata = database_metadata.iloc[top_indices].copy()
        
        results.append(SimilarityResult(
            query_idx=i,
            top_k_indices=top_indices,
            top_k_scores=top_scores,
            top_k_chip_ids=top_chip_ids,
            top_k_metadata=top_metadata,
        ))
    
    logger.info(f"Similarity search complete. "
                f"Score range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    
    return results


def internal_similarity(
    embedding_result: EmbeddingResult,
    threshold: float = 0.9,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all patches in a scene.
    
    Useful for finding which parts of the OHRC scene have similar terrain
    characteristics according to LunarFM's learned representation.
    
    Args:
        embedding_result: EmbeddingResult from extract_embeddings
        threshold: Similarity threshold for highlighting similar regions
        
    Returns:
        Similarity matrix of shape [N_patches, N_patches]
    """
    emb_norm = normalize(embedding_result.global_embeddings, norm='l2')
    sim_matrix = emb_norm @ emb_norm.T
    
    n_patches = len(sim_matrix)
    n_above_threshold = (sim_matrix > threshold).sum() - n_patches  # Exclude diagonal
    
    logger.info(f"Internal similarity matrix: {sim_matrix.shape}")
    logger.info(f"  {n_above_threshold} pairs above threshold {threshold}")
    logger.info(f"  Mean similarity: {sim_matrix.mean():.4f}")
    
    return sim_matrix
