# Computer Vision for Elliptic Curves

[![arXiv](https://img.shields.io/badge/arXiv-2604.15155-b31b1b.svg)](https://arxiv.org/abs/2604.15155)

This repository applies convolutional neural networks to arithmetic data attached to elliptic curves over $\mathbb{Q}$ - in particular, sequences of Frobenius traces $a_p$ and their Dirichlet-twisted variants — to study problems such as distinguishing genuine $L$-function curves from synthetic data drawn from Sato–Tate distribution, classifying analytic rank, and probing what features the networks rely on using saliency maps.

## Repository layout

```
src/
├── config.py                    # paths to data/ and results/
├── data_generation/             # build datasets
├── models/                      # CNN architectures + training scripts
└── analysis/                    # transfer learning and saliency maps
data/                            # raw CSVs and generated .npy arrays
results/                         # checkpoints, metrics, saliency outputs
plots/                           # figures
```

## Scripts

### `src/data_generation/`

- **`get_chi.py`** — Generates all primitive Dirichlet characters via SageMath
- **`generate_fake_ap.py`** — Produces synthetic $a_p$ sequences by rejection-sampling angles from the Sato–Tate distribution.
- **`generate_image.py`** — Converts $a_p$ sequences into 2D twisted images using the Dirichlet characters from `get_chi.py`. 

### `src/models/`

- **`L_func_classification.py`** — Trains a 2D CNN to discriminate genuine vs. synthetic elliptic curves from twisted images. 
- **`L_func_classification_1d.py`** — 1D version of the above: trains on only the first column of the real channel (raw $a_p$ values).
- **`rank_classification_2d.py`** — Trains a 2D CNN to classify rank from twisted images of curves.

### `src/analysis/`

- **`evaluate_transfer.py`** — Transfer learning on a different conductor range.
- **`classify_and_plot_fake_ranks.py`** — Trains the 1D rank classifier on real labeled curves, applies it to synthetic curves and plots a murmuration figure grouped by predicted rank.
- **`generate_saliency.py`** — Generate saliency maps as `.npy` files and plot heatmaps.
- **`plot_rank_saliency_maps.py`** — Plot only.

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
