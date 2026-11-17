# MSPE-Net

MSPE-Net extracts signal photons from multi-track ICESat-2 point clouds. This release contains the network, the final fine-tuned checkpoint, the ten-region labeled evaluation dataset, and the scores for the adjacent-track experiment.

## Run an evaluation

The reference environment uses Python 3.11, NumPy 1.26.4, and PyTorch 2.4.0 with CUDA 12.1. Install the PyTorch build for your device and the listed dependencies, then run from the repository root:

~~~sh
python -m pip install -r requirements.txt
python evaluate.py mont_2 --target 7 --count 1
python evaluate.py mont_2 --target 7 --count 5
python evaluate.py mont_2
~~~

The first two commands score the same target with one and five input tracks. The third evaluates the whole region with all tracks present. The program prints TP, FP, FN, precision, recall, and F₁. The device defaults to CUDA when available. Sampling and floating-point operations may give different results on other hardware or software versions.

## Evaluation data

The ten regions contain 2,009,597 measured photons on 102 region-local tracks. Each file in `data/*.npz` contains a numeric `points` array with six columns in this order: planar X (m), planar Y (m), elevation (m), reference class, region-local track ID, and source along-track distance. Class 1 is signal, class 0 is noise, and -1 is excluded from scoring when present. The integer `bounds` array identifies the original inference chunks: chunk i contains rows from bounds[i] up to, excluding, bounds[i+1].

`data/regions.csv` lists region IDs, terrain, counts, and file checksums. `data/source_granules.csv` lists the source ATL03 granule/beam pairs by region as reported in the study. The mountain and polar reference labels use airborne lidar elevation data. The forest, urban, and coastal reference labels use manual interpretation aided by photon confidence and spatial continuity. Source products: [ICESat-2 ATL03 v6](https://doi.org/10.5067/ATLAS/ATL03.006), [ATL03 v7](https://doi.org/10.5067/ATLAS/ATL03.007), and [USGS 3DEP](https://www.usgs.gov/3d-elevation-program).

## Checkpoint and reference-assisted evaluation

The final checkpoint is `weights/accepted_fp_reflux.pth` (SHA-256 `dda212fc04e9c914c42e991e3f2c2a00f3d23bda73ee627de13d8705e8348ac5`). Fine-tuning used simulated point clouds. Aggregate false-positive height offsets from measured mountainous data informed the simulated hard negatives, and performance on measured terrain data was used to assess checkpoint acceptance.

To reproduce the reported benchmark, each chunk uses the minimum and maximum elevation of its reference-labeled signal photons on the current input tracks. The interval is extended by 10 m at each end. Photons outside it receive a noise prediction and remain eligible for scoring. If the chunk contains no reference signal, all its selected photons enter the network. XYZ coordinates are scaled independently to [-1, 1] within the retained chunk. To run inference without using reference labels in preprocessing, add `--no-reference-gate`. That option changes the evaluation protocol.

## Figures 15–16 and Table VII

Every track serves in turn as the fixed target. Additional tracks enter in increasing absolute cross-track distance from that target. Positions are the per-track median cross-track coordinates over the common along-track interval. Ties are broken by local track ID. Only photons on the fixed target enter target-track scores. Input-track scores pool labeled photons across all included tracks.

- `benchmark/track_positions.csv` lists the positions used to order tracks.
- `benchmark/track_scores.csv` contains all 510 configurations with confusion counts and unrounded scores for both evaluation scopes.
- `benchmark/selected_examples.csv` gives the five illustrated targets. They were selected for strictly increasing F₁ under both scopes across the five configurations.
- `benchmark/mean_scores.csv` gives the Figure 16 target-track curves and the corresponding all-input comparison. First average over targets within each region, then average the regions equally within each terrain or across the ten regions.

## Code credits

The backbone, transformer, and point-set operations build on [Point-Transformers](https://github.com/qq456cvb/Point-Transformers) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Their license notices are in `third_party/`.
