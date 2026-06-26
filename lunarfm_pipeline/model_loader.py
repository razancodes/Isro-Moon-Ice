"""
Model Loader
=============

Load the pretrained LunarFM (MultiMAE) model from a PyTorch Lightning checkpoint
and prepare it for inference with a subset of modalities.

The key challenge: MultiMAE's __init__ requires a `num_channels` dict that defines
which input/output adapters to create. The checkpoint was trained with all 7 data
groups (18 channels total). We need to instantiate the model with the SAME architecture
as training, load weights, then run inference with only the modalities we have.

The encoder's forward_encoder() naturally handles partial modality input — it only
processes modalities present in the input dict.
"""

import sys
import os
from pathlib import Path
from collections import Counter
from typing import Optional, Union

import torch
import yaml
from loguru import logger


def get_num_channels_from_config(config_path: str) -> dict:
    """
    Parse the training config.yaml to reconstruct the num_channels dict
    that MultiMAE.__init__ expects.
    
    The config uses group_collate=True, meaning modalities with the same 
    `data_group` are concatenated along the channel dim. The num_channels 
    dict maps data_group_name -> count_of_modalities_in_that_group.
    
    Returns:
        dict mapping data_group name -> number of channels
        e.g. {'DGHRM_RockAbsortion_SAM': 1, 'DGHRM_Temperature': 2, ...}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets_spec = config['dataloader']['datasets_spec']
    
    # Count how many modalities share each data_group
    groups = []
    for name, spec in datasets_spec.items():
        group = spec.get('data_group', name)
        groups.append(group)
    
    num_channels = dict(Counter(groups))
    
    logger.info(f"Reconstructed num_channels from config: {num_channels}")
    return num_channels


def get_model_hparams_from_config(config_path: str) -> dict:
    """
    Extract model hyperparameters from the training config.
    
    Returns:
        dict with model constructor kwargs
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    
    # Remove _target_ key (Hydra convention, not a constructor arg)
    hparams = {k: v for k, v in model_cfg.items() if k != '_target_'}
    
    return hparams


def load_lunarfm_model(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cpu',
    eval_mode: bool = True
) -> torch.nn.Module:
    """
    Load the pretrained LunarFM MultiMAE model from a Lightning checkpoint.
    
    This function:
    1. Parses the training config to reconstruct the num_channels dict
    2. Extracts model hyperparameters from the config
    3. Instantiates MultiMAE with the correct architecture
    4. Loads pretrained weights from the checkpoint
    5. Moves model to the specified device
    
    Args:
        checkpoint_path: Path to last.ckpt 
        config_path: Path to config.yaml (the one shipped with the model)
        device: 'cpu' or 'cuda' or 'cuda:0' etc.
        eval_mode: If True, set model to eval mode (no dropout, etc.)
    
    Returns:
        Loaded MultiMAE model ready for inference
    """
    # Add the LunarFM source directory to sys.path so we can import lunarlab
    lunarfm_src = str(Path(__file__).parent.parent / 'LunarFM-Science-Release' / 'src')
    if lunarfm_src not in sys.path:
        sys.path.insert(0, lunarfm_src)
    
    from lunarlab.models.multi_mae.multimae import MultiMAE
    
    # Step 1: Get num_channels from config
    num_channels = get_num_channels_from_config(config_path)
    
    # Step 2: Get model hyperparameters from config  
    hparams = get_model_hparams_from_config(config_path)
    
    # Step 3: Override num_channels (config has it as None, we computed it)
    hparams['num_channels'] = num_channels
    
    # Step 4: Determine nan_input_handling from experiment name
    # The experiment name contains 'nan_token_zero' indicating this setting
    nan_input_handling = 'token_zero'
    hparams['nan_input_handling'] = nan_input_handling
    
    logger.info(f"Instantiating MultiMAE with hparams: { {k:v for k,v in hparams.items() if k != 'num_channels'} }")
    logger.info(f"  num_channels: {num_channels}")
    
    # Step 5: Instantiate model
    model = MultiMAE(**hparams)
    
    # Step 6: Load checkpoint weights
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Lightning checkpoints store weights under 'state_dict' key
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Load weights (strict=False allows partial loading if needed)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)} keys")
        for k in missing[:10]:
            logger.warning(f"  Missing: {k}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing)-10} more")
    
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
        for k in unexpected[:10]:
            logger.warning(f"  Unexpected: {k}")
    
    if not missing and not unexpected:
        logger.info("All checkpoint weights loaded successfully (exact match)")
    
    # Step 7: Move to device and set mode
    model = model.to(device)
    
    if eval_mode:
        model.eval()
        logger.info("Model set to eval mode")
    
    logger.info(f"LunarFM model loaded on {device}")
    logger.info(f"  Encoder depth: {model.get_num_layers()} layers")
    logger.info(f"  Token dim: {model.dim_tokens}")
    logger.info(f"  Patch size: {model.patch_size}")
    logger.info(f"  Input adapters: {list(model.input_adapters.keys())}")
    
    return model


def get_available_modalities(model: torch.nn.Module) -> list:
    """
    Get the list of modality names (data groups) that the model has input adapters for.
    
    Returns:
        List of modality/data_group names
    """
    return list(model.input_adapters.keys())


def get_normalization_stats() -> dict:
    """
    Return the hardcoded per-modality normalization statistics used during training.
    These are the mean/std values from each dataset class in lrodatasets.py.
    
    For our OHRC use case, we primarily need the ClementineUVVISMosaic stats
    since OHRC (panchromatic, ~450-750nm) maps most closely to that modality.
    
    Returns:
        Dict mapping data_group_name -> {mean, std} 
        For multi-channel groups, returns per-channel stats in order.
    """
    return {
        # Single-channel modalities
        'DGHRM_RockAbsortion_SAM': {
            'channels': ['rock_abundance'],
            'mean': [0.0031], 'std': [0.0052]
        },
        'LRO_LOLA_Global_LDEM_118m_Mar2014': {
            'channels': ['dem'],
            'mean': [-725.30262], 'std': [4607.3882]
        },
        'ClementineUVVISMosaic': {
            'channels': ['uvvis_750nm'],
            'mean': [39.15767], 'std': [13.04669]
        },
        
        # Two-channel groups (concatenated in order from config)
        'DGHRM_Temperature': {
            'channels': ['bolometric_temp', 'regolith_temp'],
            'mean': [96.5667, 94.3388], 'std': [6.0609, 6.8827]
        },
        'MiniRF_Global_Mosaic': {
            'channels': ['cpr', 's1'],
            'mean': [41.3813, 48.6072], 'std': [50.3208, 50.2141]
        },
        
        # Four-channel group (GRAIL: anomaly, bouguer, disturbance, stdev — order from config)
        'GRAIL_Global_Mosaic': {
            'channels': ['anomaly', 'bouguer', 'disturbance', 'stdev'],
            'mean': [-1.4164, 3.9903, -5.3486, 2927.2222],
            'std': [132.1000, 241.4268, 147.5498, 28.1879]
        },
        
        # Seven-channel group (WAC Hapke: 321, 360, 415, 566, 604, 643, 689nm — order from config)
        'LRO_WAC_hapke_Eq_Mosaic': {
            'channels': ['hapke_321nm', 'hapke_360nm', 'hapke_415nm', 'hapke_566nm',
                         'hapke_604nm', 'hapke_643nm', 'hapke_689nm'],
            'mean': [0.0176, 0.0209, 0.0256, 0.0369, 0.0399, 0.0423, 0.0456],
            'std': [0.0047, 0.0057, 0.0071, 0.0098, 0.0104, 0.0109, 0.0117]
        },
    }
