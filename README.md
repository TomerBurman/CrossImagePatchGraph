# Cross-Image GLOT

A refactored research codebase for class-conditioned cross-image patch graphs over frozen DINOv2 features.

## Separation of concerns

- `src/cross_image_glot/`: reusable implementation.
- `notebooks/`: thin Colab experiment entry points.
- `configs/`: experiment configurations.
- Google Drive: datasets, feature shards, checkpoints, and results.

## Google Drive layout

```text
MyDrive/CrossImageGLOT/
├── data/
│   ├── images.zip
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── features/dinov2_vits14_224/
├── checkpoints/
└── results/
```

Raw images and feature shards are copied from Drive into `/content/CrossImageGLOT_runtime` for fast access. Newly generated feature shards and all checkpoints are persisted back to Drive.

## Notebook order

1. `00_prepare_data_and_features.ipynb`: download/copy images and build missing DINOv2 shards.
2. `01_episodes_and_frozen_baselines.ipynb`: cached episodes and frozen CLS/mean-patch baselines.
3. `02_graph_construction_diagnostics.ipynb`: graph shape, edge, batching, and leakage diagnostics.
4. `03_train_graphsage.ipynb`: plain GraphSAGE experiment.
5. `04_train_residual_graph_model.ipynb`: CLS-preserving residual graph correction.
6. `05_final_evaluation.ipynb`: final evaluation on held-out test classes only.

Every notebook is designed to run from a fresh Colab runtime. Notebooks never import other notebooks.

## First use

1. Upload this repository to GitHub.
2. Replace `REPO_URL` in each notebook with the repository URL.
3. Open the notebook from GitHub in Colab.
4. Run `00_prepare_data_and_features.ipynb` until all three feature caches are complete.

Once the cache exists, graph training notebooks do not load DINOv2 and do not extract the image archive.
