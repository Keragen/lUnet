# LUnet preprocessing pipeline (microscopy movies)

This repository contains an R script (`LUnet-server4.r`) used to preprocess multi-channel microscopy movies and generate segmentation masks (nuclei + cytoplasm) prior to signal quantification (with **Spot-On**).

---

## What the script does

### 1) Load two pretrained U-Net models (Keras)
The pipeline uses two separate U-Net models (`.h5` weights):
- **NUC** model: nucleus segmentation
- **CYTO** model: cytoplasm segmentation

### 2) Parse input `.TIF` files (two channels)
The script scans subfolders under an input root directory and expects file names like:

`<basename>_w<channel>_s<movie>_t<frame>.TIF`

It groups files by:
- `basename`
- `movie` (s)
- `frame` (t)
- `channel` (w)

### 3) Build RGB preview images
For each frame:
- reads both channels
- applies an **auto-contrast** normalization
- merges into an RGB image (`rgbImage`)
- exports:
  - an RGB `.jpg`
  - a per-channel `.jpg` into `green/` and `red/`

It also writes a `redgreenfiles.txt` mapping file linking the generated `.jpg` to the original `.TIF` files (useful for Spot-On).

### 4) Segment nuclei and cytoplasm
For each RGB image:
- predicts probability maps using both U-Nets
- thresholds predictions (`nucmaskthreshold`, `cytomaskthreshold`)
- removes small false positives (size-based filtering)
- separates nuclei using **watershed**
- assigns cytoplasm regions by **seeded propagation / Voronoi-like partitioning** around nuclei
- optionally fills small background gaps and removes nucleus pixels from cytoplasm masks

### 5) Simple object tracking across frames
To keep consistent object IDs over time:
- computes object centers (via PCA-based centroid estimation)
- stores centers in `positions.txt`
- remaps object IDs between consecutive frames using nearest-neighbor matching (distance threshold `thr`)

### 6) Export masks + measurements
Per processed folder/movie the script generates:
- `results-masks/` : color mask images
- `results-nuclei/` : nuclei masks exported as `.txt`
- `results-cytoplasm/` : cytoplasm masks exported as `.txt`
- `results-composite/`, `results-watershed/`, `results-voronoi/`, `results-annot/` (created by the script for intermediate/visual outputs)
- `*_BIOMETRY.txt` : per-object measurements including surface/area, mean intensity, length/width, elongation, orientation, and an “infolding” metric (based on convex hull vs. object area)

### 7) Parallel processing
The script parallelizes computation **per subfolder** using `foreach` + `doParallel` (cluster size set by `makeCluster(n)`).

---

## Requirements

R packages used include:
- `keras`, `magick`, `EBImage`
- `shotGroups`, `scales`, `dplyr`, `gtools`, `stringr`
- `foreach`, `doParallel`

You also need a working Keras/TensorFlow setup compatible with the `.h5` model weights.

---

## Configuration (edit in the script)

Key paths:
- `movieroot` : input root folder containing `.TIF`
- `targetmovie` : output root folder
- `unetpathnuc`, `unetpathcyto` : model weight files

Key parameters:
- `input_sizeX`, `input_sizeY` (default 1024)
- `nucmaskthreshold`, `cytomaskthreshold`
- `watersize1`, `watersize2` (watershed)
- `Voronoilambda` (propagation / Voronoi assignment)
- `dobackground`, `extractnucfromcyto`

Parallelism:
- `makeCluster(5)` sets the number of worker processes

---

## Running

1. Install required R packages (and EBImage/Keras dependencies).
2. Edit the paths and parameters at the top of the script.
3. Run the script in R.

At the end it stops the cluster and prints: `all done!!`




Download the latest release to get the code together with the ML models required by this software.


