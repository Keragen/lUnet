The pipeline takes raw two-channel microscopy movies (for example, green and red channels) and processes them in two consecutive steps. First, it converts the raw .TIF frames into JPEG previews and runs two deep-learning U-Net models to segment each frame into nucleus and cytoplasm regions. It then cleans and separates objects (e.g., watershed for nuclei and a Voronoi-like assignment for cytoplasm), and exports labeled masks plus per-cell measurements like size and shape.

In the second step, the pipeline reads those segmentation masks together with the original .TIF data and a CSV file containing tracked spot coordinates (x, y, z, frame). For each spot and each frame, it crops a small window around the spot, detects the spot pixels mainly inside the nuclear region, and can optionally use a machine-learning classifier to remove false-positive spot objects. It then measures fluorescence signals over time: the spot intensity (red and green), the local nuclear background around the spot, and the average intensity of the full nucleus and cytoplasm belonging to that spot’s cell. Finally, it writes per-frame quantification tables, produces QC composite images, generates per-track plots, and saves spot mask examples for later quality control or machine-learning training.

# LUnet + Spot-On pipeline (microscopy movies → segmentation → spot quantification)

This repo provides a two-step R pipeline that must be run **in sequence**:

1. **`LUnet-server4.r`**  
   Preprocesses raw microscope `.TIF` files, generates RGB previews, and produces **nucleus + cytoplasm segmentation masks** using two pretrained U-Net models.

2. **`spoton5.r`**  
   Uses the LUnet masks + original `.TIF` movies + a spot tracking `.csv` to **detect spot masks**, **quantify red/green intensities**, generate QC composites, and export tables/plots.

> Important: LUnet generates masks at **1024×1024** for performance. `spoton5.r` automatically rescales masks to match the original image size (e.g. 2048×2048) when needed.

---

## Step 0 — Requirements

### R packages
You will need (at least):
- `keras`, `magick`, `EBImage`, `ggplot2`, `sp`, `stringr`
- `foreach`, `doParallel`
- LUnet also uses: `shotGroups`, `scales`, `dplyr`, `gtools`
- Spot-On also uses: `geometry`

### Deep learning models
- U-Net weights (`.h5`):
  - nucleus model (NUC)
  - cytoplasm model (CYTO)
- Spot validation model (`.h5`, optional but used by default in the script):
  - `fullmodelSpot.h5` (filters false-positive spot objects in the red channel)

A working TensorFlow/Keras setup is required to load these `.h5` models.

---

## Step 1 — Run LUnet preprocessing (`LUnet-server4.r`)

### Purpose
- Read raw multi-channel `.TIF` movies
- Generate RGB `.jpg` previews + channel-specific `.jpg`
- Segment **nuclei** and **cytoplasm** using 2 U-Nets
- Export masks + basic object tracking + per-object biometry measurements

### Expected input naming
Files are expected to follow a pattern like:
`<basename>_w<channel>_s<movie>_t<frame>.TIF`

(2 channels are assumed.)

### Key outputs (per dataset folder)
LUnet writes to `targetmovie/.../<subfolder>/`:
- `results-masks/` : color mask overlays (jpg)
- `results-nuclei/` : nuclei masks as text (`.txt`, labeled IDs)
- `results-cytoplasm/` : cytoplasm masks as text (`.txt`, labeled IDs)
- `red/` and `green/` : exported channel jpgs
- `redgreenfiles.txt` : mapping between each generated jpg frame and the original green/red `.TIF` paths  
- `*_BIOMETRY.txt` : per-object measurements (surface, intensity, length/width, elongation, orientation, infolding, …)

### Configure and run
Edit paths in the script:
- `movieroot`   : input folder containing `.TIF`
- `targetmovie` : output folder where results will be written
- `unetpathnuc`, `unetpathcyto` : paths to the `.h5` weights

Parallelism:
- `makeCluster(5)` sets the number of workers (per subfolder).

Run the script in R. At the end it prints: `all done!!`

---

## Step 2 — Run Spot-On quantification (`spoton5.r`)

### Purpose
For each tracked spot (track/frame coordinates), the script:
- Crops a small window around the spot (auto window size: ~40 px for 2048 images, ~20 px for 1024)
- Detects a **spot mask** using a threshold computed on nuclear pixels only (mean + 2×sd)
- Optionally cleans spot objects using a ML model (`fullmodelSpot.h5`) in the **red** channel
- Quantifies:
  - spot intensity (red/green)
  - local nuclear background (excluding spot)
  - nucleus-window intensity
  - full nucleus and full cytoplasm mean intensities (using the nucleus ID near the spot)
- Exports per-frame tables, per-track plots, QC composite images, and spot mask samples for later ML/QC

### Inputs
Spot-On expects:

1) **LUnet output folder** (`targetpath`), containing for each frame:
- `results-nuclei/<basename>_t<frame>.jpg.txt`
- `results-cytoplasm/<basename>_t<frame>.jpg.txt`
- `redgreenfiles.txt` (maps each jpg frame to the original `.TIF` files)

2) **Original `.TIF` movies**
The script uses `redgreenfiles.txt` to locate the original green/red `.TIF` per frame and then selects the z-slice.

3) **Tracking CSV (`.csv`)**
Tab-separated file with columns:
`row, track, frame, x, y, z`

Missing frames per track are filled by repeating first/last known coordinates so that each track has values over the full time range.

### Outputs
Written under `spotpath/<dataset>/<subfolder>/`:
- `signal_data_<...>.txt` : per-frame measurements (tab-separated)
- QC composite images per frame:
  - `<basename>_track<track>_t<frame>.jpg`
- Per-track plots:
  - background-corrected spot signals over time (red/green)
  - full nucleus/cytoplasm mean intensities over time
- Spot mask samples for ML/QC under `AIspotpath/...`:
  - spot masks (`track<id>-<frame>.jpg`)
  - corresponding nucleus masks in `NUCmask/`

### Configure and run
Edit key variables:
- `setwd(...)`         : folder with original movies
- `targetpath`         : LUnet output root (from Step 1)
- `spotpath`           : where quantification results are saved
- `AIspotpath`         : where mask samples for ML/QC are saved
- `modelspotpath`      : spot-cleaning ML model (`fullmodelSpot.h5`)
- `alldir0`, `finalMaskpath` : dataset selection / output naming

Parallelism:
- `makeCluster(30)` runs one worker per track (set this to match the maximum number of tracks in your CSVs).

Run the script in R. At the end it stops the cluster and prints: `all done!!`

---

## Notes / assumptions

- The pipeline assumes **two channels** per frame (typically green/red).
- LUnet produces masks at **1024×1024** for speed; Spot-On can upsample masks if the original frames are 2048×2048.
- Spot-On uses the **red-derived spot mask** to quantify the green spot signal as well (same pixel mask).
- Full nucleus/cytoplasm signals are computed for the nucleus label (`nucname`) found near the spot window.

---

## Quick checklist

1. ✅ Put raw `.TIF` movies in `movieroot` (LUnet step).
2. ✅ Set `targetmovie` for LUnet output and run `LUnet-server4.r`.
3. ✅ Ensure `redgreenfiles.txt` and mask folders exist in `targetpath`.
4. ✅ Provide tracking `.csv` files next to the `.TIF` movies.
5. ✅ Set `targetpath`, `spotpath`, and `modelspotpath` then run `spoton5.r`.
6. ✅ Collect tables/plots/QC images from `spotpath`.





Download the latest release to get the code together with the ML models required by this software.




