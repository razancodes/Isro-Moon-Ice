# ISRO Bharatiya Antariiksh Hackathon (BAH) - Problem Statement 8

**Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning**

This repository contains an end-to-end processing pipeline integrating high-resolution optical imagery (OHRC) and dual-frequency synthetic aperture radar (DFSAR) from Chandrayaan-2. It satisfies both the Physics-based radar track and the AI-driven representation track of Problem Statement 8.

## Core Idea behind the Problem statement and how we have approached it: 
Finding ice on the Moon is incredibly difficult because rough rocks and deep craters can easily confuse satellite sensors. We tackled this problem by building two completely pipelines that analyze different types of satellite data, ultimately working together to find the safest path to the ice:

1. **The Radar Pipeline (DFSAR)**: We used Chandrayaan-2's dual-frequency radar to peer up to 5 meters *beneath* the lunar dust. By mathematically comparing the L-band (deep penetration) and S-band (shallow penetration) radar signals, we successfully filtered out false-positive rocks and identified **456 square kilometers of deeply buried water-ice**.
2. **The Optical AI Pipeline (OHRC)**: Using Chandrayaan-2's ultra-high-resolution optical camera (capable of seeing objects as small as 25cm), we mapped the exact physical hazards on the surface. We used classical computer vision to detect boulders and steep slopes, and an advanced AI Foundation Model (LunarFM) to automatically group the terrain into safe vs. dangerous zones.

### What Is Left To Do?
- **Rover Traverse Path Planning**. Now that we have mapped the underground ice (the destination) and the surface hazards (the obstacles), This algorithm will simulate a rover landing in a safe, flat zone and autonomously driving to the ice without falling into steep craters or hitting boulders.
- **[LunarFM-IceNet Hackathon Plan](hackathon_plan.md)**: We have formulated a complete, tiered research plan to fuse these pipelines and build an end-to-end physics-informed foundation model. Read the [full implementation roadmap here](hackathon_plan.md).

---

## 📸 Pipeline Visualizations

### Optical Analytics (OHRC 0.25m/pixel)
These visualizations are generated automatically by `demo_lunarfm.py`. They operate on ultra-high-resolution optical data to map surface hazards that a rover must avoid.

![Classical Analytics](assets/ohrc_classical_analytics.png)
**Terrain Hazard Map:** This image shows the results of our classical computer vision algorithms. It highlights Permanently Shadowed Regions (PSRs) in dark blue (areas that never see sunlight and are dangerously cold), extremely rough slopes in yellow/red, and individual boulders (red dots) detected using a Laplacian of Gaussian blob detector.

![LunarFM PCA Embeddings](assets/ohrc_pca_false_color.png)
**LunarFM AI Embeddings (PCA False Color):** This image represents the "brain" of the AI. We passed the lunar surface through ISRO's pre-trained Foundation Model (MultiMAE) to extract 768-dimensional features. We compressed these features into RGB colors using Principal Component Analysis (PCA). Similar colors mean the AI thinks the terrain has a similar physical and geological structure.

![Terrain Clusters](assets/ohrc_terrain_clusters.png)
**K-Means Terrain Segmentation:** Here, we asked the AI to group the terrain into 6 distinct categories without any human supervision. The AI successfully separated flat crater floors (safe for driving), steep illuminated rims, and dark shadows entirely on its own.

### Radar Analytics (DFSAR L-band / S-band)
These visualizations are generated automatically by `dfsar_processing.py`. They combine physics-based radar metrics to look *beneath* the surface.

![DFSAR Ice Detection Map](assets/dfsar_ice_detection.png)
**Subsurface Ice Anomaly Map:** This is a massive 100km x 100km crop of the Lunar South Pole. 
- **Left (CPR):** Shows the raw L-band Circular Polarization Ratio. Bright areas indicate strong radar scattering (which could be ice OR rough rocks).
- **Center (Ice Map):** The culmination of our physics pipeline. We overlay the CPR with the S-band Degree of Polarization. **Bright lime green** pixels represent confirmed subsurface water-ice (where the radar signature strengthens with depth). Orange pixels are rejected false-positive rocky areas.
- **Right (RGB):** The Yamaguchi S-band decomposition. Red shows surface scattering, green shows double-bounce (rocks/craters), and blue shows volumetric scattering (deep ice/dust).

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup
The pipeline requires Python 3.10+ and heavy remote sensing / deep learning libraries. It is highly recommended to use a Conda environment.

```powershell
# Create and activate environment
conda create -n isro_bah python=3.10 -y
conda activate isro_bah

# Install core scientific and GIS dependencies
pip install numpy rasterio loguru scikit-learn matplotlib psutil

# Install PyTorch (required for LunarFM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data & Weights Acquisition
Since radar data, optical imagery, and neural network weights total tens of gigabytes, they are **excluded from this repository via `.gitignore`**. You must download them manually and place them in the correct directory structure.

**A. LunarFM Pre-trained Weights**
The AI pipeline requires the pre-trained LunarFM (MultiMAE) weights. Note that these weights are **not** available on the public ISRO data portal. 
1. The weights are provided separately under a specific licensing agreement by the LunarFM development team. Please follow their official access request process to obtain the files.
2. Once access is granted, download the `last.ckpt` weights file.
3. Place it exactly at: `LunarFM-Science-Release/weights/last.ckpt`

**B. Chandrayaan-2 PRADAN Data**
1. Navigate to the ISRO PRADAN/ISSDC data portal.
2. **For Track 1 (Radar)**: Download the DFSAR L4-MOSAIC bundles for the 2025-06-30 South Pole region (both L-band and S-band).
3. **For Track 2 (Optical)**: Download the OHRC calibrated bundle for 2025-01-25.

### 3. Data Directory Structure
Extract your downloaded files so your project root matches this exact structure:

```text
Isro-BAH-RS/
│
├── ohrc_data/
│   └── data/
│       └── calibrated/
│           └── 20250125/
│               ├── ch2_ohr_ncp_20250125T0328498909_d_img_d18.tif  # High-res optical
│               └── ch2_ohr_ncp_20250125T0328498909_d_img_d18.xml
│
├── dfsar_data/
│   ├── L_band/
│   │   └── data/derived/20250630/
│   │       ├── ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif
│   │       ├── ch2_sar_ndxl_20250630mpcpspwest_d_srd_xx_fp_xx_xxx.tif
│   │       └── ch2_sar_ndxl_20250630mpcpspwest_d_trt_xx_fp_xx_xxx.tif
│   └── S_band/
│       └── data/derived/20250630/
│           ├── ch2_sar_ndxl_20250630my4rspwest_d_evn_xx_fp_xx_xxx.tif
│           ├── ch2_sar_ndxl_20250630my4rspwest_d_odd_xx_fp_xx_xxx.tif
│           ├── ch2_sar_ndxl_20250630my4rspwest_d_vol_xx_fp_xx_xxx.tif
│           └── ch2_sar_ndxl_20250630my4rspwest_d_hlx_xx_fp_xx_xxx.tif
│
└── LunarFM-Science-Release/  # Cloned ISRO LunarFM repository
    └── weights/
        └── last.ckpt         # Pre-trained MultiMAE weights
```

### 4. Running the Pipeline

#### Part A: OHRC Optical & AI Pipeline (Hazards & LunarFM)
This script processes the 0.25m/pixel OHRC imagery. It applies classical photogrammetry to extract terrain hazards (shadows, boulders, roughness) and uses the frozen LunarFM AI model to extract spatial embeddings and perform unsupervised terrain segmentation.

```powershell
# Set the environment variable to point to your specific OHRC TIFF
$env:OHRC_PATH = "C:\Users\MRaza\Documents\Isro-BAH-RS\ohrc_data\data\calibrated\20250125\ch2_ohr_ncp_20250125T0328498909_d_img_d18.tif"

# Run the pipeline
python demo_lunarfm.py --data-dir "C:\Users\MRaza\Documents\lunarlab-public"
```
**Outputs**: Generated in `outputs/ohrc_demo/` (Hazard Maps, Boulder Overlays, PCA Embeddings, Terrain Clusters).

#### Part B: DFSAR Radar Pipeline (Subsurface Ice Detection)
This script processes 17 GB of L-band and S-band radar data. It uses a highly memory-efficient tiled approach (row-wise processing) to prevent RAM exhaustion.

```powershell
python dfsar_processing.py
```
**Outputs**: Generated in `outputs/dfsar_results/` (High-confidence ice masks, DOP/CPR maps, Volumetric estimation logs, and a final 600km visual dashboard).

---

## 🏗️ Repository Architecture

- **`dfsar_processing.py`**: Core radar physics pipeline. Implements memory-efficient `rasterio` window reads (500 rows at a time). Calculates DOP from Yamaguchi decompositions, applies ISRO ice thresholds (`CPR > 1.0` & `DOP < 0.87`), computes depth index, and runs volumetric scaling.
- **`demo_lunarfm.py`**: Core optical AI pipeline. Orchestrates data ingestion, classical filtering, and neural network embedding extraction.
- **`lunarfm_pipeline/`**:
  - `ohrc_analytics.py`: Classical computer vision (Laplacian of Gaussian for boulders, rolling standard deviation for surface roughness). Handles OHRC TDI readout column destriping.
  - `model_loader.py`: Reconstructs PyTorch Lightning architecture dynamically from `config.yaml` to load frozen ISRO MultiMAE weights.
  - `preprocessing.py`: Handles grid tiling (112x112 patches) and scene-specific standard scalar normalization.
  - `embeddings.py` & `visualization.py`: Extracts 768-D tokens, performs PCA dimensionality reduction, and applies K-Means spatial clustering.

---

## 📝 Troubleshooting & Notes

1. **OHRC Destriping**: Raw OHRC images often contain vertical column banding due to TDI (Time Delay Integration) CCD readout noise when looking into deep shadows. `ohrc_analytics.py` applies a mandatory column-wise zero-mean destriping pass before processing.
2. **Memory Limits (DFSAR)**: The DFSAR L4-MOSAIC products total ~16.8 GB. `dfsar_processing.py` uses row-chunking. Do *not* attempt to load the entire arrays into memory `src.read(1)` unless you have 64GB+ of RAM.
3. **Model Normalization**: LunarFM weights were trained on Clementine data. Applying Clementine global statistics to OHRC data crushes the dynamic range due to vastly different lighting geometries at the South Pole. The pipeline utilizes **scene-specific Z-score normalization** before passing patches to the MultiMAE encoder.
