# Computer Vision for Elliptic Curves

Code accompanying the paper:

> *Computer Vision for Elliptic Curves.* arXiv:2604.15155. https://arxiv.org/abs/2604.15155

This repository applies convolutional neural networks to arithmetic data attached to elliptic curves over $\mathbb{Q}$ — in particular, sequences of Frobenius traces $a_p$ and their Dirichlet-twisted variants — to study problems such as distinguishing genuine $L$-function data from synthetic Sato–Tate samples, classifying analytic rank, and probing what features the networks rely on (saliency maps, murmurations).

## Repository layout

```
src/
├── config.py                    # paths to data/ and results/
├── data_generation/             # build datasets from a_p coefficients
├── models/                      # CNN architectures + training scripts
└── analysis/                    # evaluation, saliency, plotting
data/                            # raw CSVs and generated .npy arrays
results/                         # checkpoints, metrics, saliency outputs
plots/                           # figures
```

All scripts read `DATA_DIR` and `RESULTS_DIR` from `src/config.py`.

## Pipeline

A typical end-to-end run goes:

1. **Build inputs:** `data_generation/get_chi.py` → `data_generation/generate_fake_ap.py` → `data_generation/generate_image.py`
2. **Train a model:** one of the scripts in `models/`
3. **Analyze:** scripts in `analysis/` (transfer evaluation, saliency, murmuration plots)

## Scripts

### `src/data_generation/`

- **`get_chi.py`** — Generates all primitive Dirichlet characters for moduli 1 through 500 via SageMath's `DirichletGroup` and writes their evaluations to `chifull_500.txt`. Required for building twisted images.
- **`generate_fake_ap.py`** — Produces 500 000 synthetic $a_p$ sequences (1 000 primes each) by rejection-sampling angles from the Sato–Tate distribution and mapping them to integers in Hasse's bound. Output: `fake_ap.csv`. Multiprocessed.
- **`generate_image.py`** — Converts $a_p$ sequences (from `ap_nocm.csv` for genuine curves and optionally `fake_ap.csv` for synthetic ones) into 2D twisted images using the Dirichlet characters from `get_chi.py`. Saves memory-mapped `.npy` arrays of shape `(N, SIZE, SIZE, 2)` (real/imag channels).

### `src/models/`

- **`L_func_classification.py`** — Trains a 2D CNN to discriminate genuine vs. synthetic elliptic curves from twisted images. Weighted BCE for class imbalance, F1-on-real model selection, checkpoint resume, multi-threshold reporting.
- **`L_func_classification_1d.py`** — 1D ablation of the above: trains on only the first column of the real channel (raw $a_p$ values), same loss / training scheme.
- **`rank_classification_2d.py`** — Trains a 2D CNN to classify analytic rank (0/1/2) from twisted images of curves in `ap_nocm.csv`. Also computes per-class saliency maps (gradient of each logit w.r.t. inputs) and saves enhanced heatmaps.

### `src/analysis/`

- **`evaluate_transfer.py`** — Loads a trained `L_function` checkpoint and reports binary metrics (accuracy, precision, recall, F1) on combined real + fake twisted-image arrays for a given image size.
- **`classify_and_plot_fake_ranks.py`** — Trains the 1D rank classifier on real labeled curves, applies it to synthetic curves from `fake_ap.csv`, and plots a murmuration figure grouped by predicted rank with train/val accuracy.
- **`check_murmuration_csv.py`** — Computes per-prime mean $a_p$ for each rank (0/1/2) from a CSV and produces a murmuration plot. Useful as a sanity check on a dataset.
- **`generate_saliency.py`** — For a trained genuine-vs-fake CNN: backpropagates output through inputs, averages absolute gradients, and saves saliency `.npy` plus marginal heatmaps. Optionally splits the genuine set by rank (0 vs. 1) using `ap_nocm.csv`.
- **`plot_rank_saliency_maps.py`** — Plot-only companion for the rank classifier. Loads pre-computed `rank{R}_wrt_class{C}_saliency_avg.npy` files and produces per-rank, all-ranks-combined, and pairwise-difference heatmaps with marginal projections. Cheap to re-run when iterating on plot styling.

## Citation

```bibtex
@misc{he2026computervisionconversetheorems,
      title={Computer vision and converse theorems}, 
      author={Yang-Hui He and Kyu-Hwan Lee and Thomas Oliver and Yidi Qi},
      year={2026},
      eprint={2604.15155},
      archivePrefix={arXiv},
      primaryClass={math.NT},
      url={https://arxiv.org/abs/2604.15155}, 
}
```
