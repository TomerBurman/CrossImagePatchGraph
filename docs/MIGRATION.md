# Migration from the monolithic notebook

The original notebook mixed storage, image extraction, DINOv2 inference, episode sampling, graph diagnostics, model definitions, training, and evaluation. The refactor maps those responsibilities as follows:

- Original cells 0–13 → `00_prepare_data_and_features.ipynb`, `storage.py`, `data.py`, `dinov2_cache.py`
- Original cells 14–21 and 77–81 → `01_episodes_and_frozen_baselines.ipynb`, `data.py`, `baselines.py`
- Original cells 22–29 → `02_graph_construction_diagnostics.ipynb`, `graph_builder.py`
- Original cells 30–76 and 82–86 → `03_train_graphsage.ipynb`, `models.py`, `training.py`
- New baseline-preserving architecture → `04_train_residual_graph_model.ipynb`
- Test-only evaluation → `05_final_evaluation.ipynb`

The raw-image dataset and DINOv2 extractor are intentionally unavailable in training notebooks. They are only needed when a feature split is missing.
