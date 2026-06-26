"""
LunarFM Pipeline for ISRO BAH Hackathon
========================================

Inference pipeline for running LunarFM (Multi-modal Masked Autoencoder) 
on Chandrayaan-2 OHRC data to extract spatial embeddings for ice detection
in lunar south polar permanently shadowed regions.

Modules:
    model_loader    - Load pretrained MultiMAE from checkpoint
    preprocessing   - Tile and normalize OHRC imagery for LunarFM input
    embeddings      - Extract embeddings from the frozen encoder
    similarity      - Similarity search against precomputed lunar embeddings
    visualization   - PCA/UMAP/clustering visualization of embedding space
    classifier      - Few-shot classification head on frozen embeddings
"""

__version__ = "0.1.0"
