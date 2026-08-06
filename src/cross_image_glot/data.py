from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import EXPECTED_CLASSES, EXPECTED_IMAGES


class MiniImageNetImageDataset(Dataset):
    """Raw images, used only to build missing frozen-feature shards."""

    def __init__(self, data_dir: str | Path, image_dir: str | Path, split: str, transform: Callable) -> None:
        if split not in EXPECTED_IMAGES:
            raise ValueError(f"Unknown split: {split}")
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform
        self.frame = pd.read_csv(self.data_dir / f"{split}.csv")[["filename", "label"]].copy()
        self.frame["filename"] = self.frame["filename"].astype(str)
        self.frame["label"] = self.frame["label"].astype(str)
        if len(self.frame) != EXPECTED_IMAGES[split]:
            raise ValueError(f"{split}: incorrect image count.")
        if self.frame["label"].nunique() != EXPECTED_CLASSES[split]:
            raise ValueError(f"{split}: incorrect class count.")
        counts = self.frame.groupby("label").size()
        if not counts.eq(600).all():
            raise ValueError(f"{split}: every class must contain 600 images.")
        self.class_ids = sorted(self.frame["label"].unique().tolist())
        self.class_to_indices = {
            class_id: self.frame.index[self.frame["label"] == class_id].tolist()
            for class_id in self.class_ids
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = self.image_dir / row["filename"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "dataset_index": index,
            "filename": row["filename"],
            "class_id": row["label"],
        }

    def indices_for_class(self, class_id: str) -> list[int]:
        return self.class_to_indices[class_id]


class MiniImageNetFeatureDataset(Dataset):
    """Random-access view of sharded frozen DINOv2 features."""

    def __init__(self, cache_dir: str | Path, split: str, max_cached_shards: int = 4) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.split_dir = self.cache_dir / split
        self.max_cached_shards = max_cached_shards
        metadata_path = self.cache_dir / "metadata.json"
        index_path = self.split_dir / "index.csv"
        if not metadata_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"Incomplete cache for split {split!r}.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.index = pd.read_csv(index_path).sort_values("dataset_index").reset_index(drop=True)
        if len(self.index) != EXPECTED_IMAGES[split]:
            raise ValueError(f"{split}: incorrect cached image count.")
        self.class_ids = sorted(self.index["label"].astype(str).unique().tolist())
        self.class_to_indices = {
            class_id: self.index.index[self.index["label"].astype(str) == class_id].tolist()
            for class_id in self.class_ids
        }
        self._shards: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def indices_for_class(self, class_id: str) -> list[int]:
        return self.class_to_indices[class_id]

    def _load_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        if shard_name in self._shards:
            self._shards.move_to_end(shard_name)
            return self._shards[shard_name]
        shard = torch.load(self.split_dir / shard_name, map_location="cpu", weights_only=True)
        self._shards[shard_name] = shard
        self._shards.move_to_end(shard_name)
        while len(self._shards) > self.max_cached_shards:
            self._shards.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> dict:
        row = self.index.iloc[index]
        shard = self._load_shard(str(row["shard_name"]))
        offset = int(row["offset"])
        cached_index = int(shard["dataset_indices"][offset])
        if cached_index != index:
            raise RuntimeError(f"Index mismatch: requested {index}, shard contains {cached_index}.")
        return {
            "cls": shard["cls"][offset],
            "patches": shard["patches"][offset],
            "class_id": str(row["label"]),
            "filename": str(row["filename"]),
            "dataset_index": index,
        }


class FewShotFeatureEpisodeDataset(Dataset):
    """Deterministic class-disjoint N-way K-shot episodes from cached features."""

    def __init__(
        self,
        base_dataset: MiniImageNetFeatureDataset,
        n_way: int,
        k_shot: int,
        queries_per_class: int,
        num_episodes: int,
        seed: int,
        vary_by_epoch: bool,
    ) -> None:
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.queries_per_class = queries_per_class
        self.num_episodes = num_episodes
        self.seed = seed
        self.vary_by_epoch = vary_by_epoch
        self.epoch = 0
        if n_way > len(base_dataset.class_ids):
            raise ValueError("n_way exceeds the number of split classes.")
        if k_shot + queries_per_class > 600:
            raise ValueError("Not enough images per class.")

    def __len__(self) -> int:
        return self.num_episodes

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def _rng(self, episode_index: int) -> np.random.Generator:
        epoch = self.epoch if self.vary_by_epoch else 0
        return np.random.default_rng(np.random.SeedSequence([self.seed, epoch, episode_index]))

    def __getitem__(self, episode_index: int) -> dict:
        rng = self._rng(episode_index)
        class_ids = rng.choice(self.base_dataset.class_ids, size=self.n_way, replace=False).tolist()
        support_items, query_items = [], []
        support_indices, query_indices = [], []
        for class_id in class_ids:
            chosen = rng.choice(
                self.base_dataset.indices_for_class(class_id),
                size=self.k_shot + self.queries_per_class,
                replace=False,
            ).tolist()
            class_support_indices = chosen[: self.k_shot]
            class_query_indices = chosen[self.k_shot :]
            support_indices.append(class_support_indices)
            query_indices.append(class_query_indices)
            support_items.append([self.base_dataset[i] for i in class_support_indices])
            query_items.append([self.base_dataset[i] for i in class_query_indices])

        def stack(items: list[list[dict]], key: str) -> torch.Tensor:
            return torch.stack([torch.stack([item[key] for item in group]) for group in items])

        support_labels = torch.arange(self.n_way)[:, None].expand(self.n_way, self.k_shot).clone()
        query_labels = torch.arange(self.n_way)[:, None].expand(self.n_way, self.queries_per_class).clone()
        return {
            "support_cls": stack(support_items, "cls"),
            "support_patches": stack(support_items, "patches"),
            "query_cls": stack(query_items, "cls"),
            "query_patches": stack(query_items, "patches"),
            "support_labels": support_labels,
            "query_labels": query_labels,
            "class_ids": class_ids,
            "support_indices": support_indices,
            "query_indices": query_indices,
            "episode_index": episode_index,
        }


def validate_class_disjointness(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    classes = {
        split: set(pd.read_csv(data_dir / f"{split}.csv")["label"].astype(str))
        for split in ("train", "val", "test")
    }
    if not classes["train"].isdisjoint(classes["val"]):
        raise ValueError("Train and validation classes overlap.")
    if not classes["train"].isdisjoint(classes["test"]):
        raise ValueError("Train and test classes overlap.")
    if not classes["val"].isdisjoint(classes["test"]):
        raise ValueError("Validation and test classes overlap.")


def validate_episode_disjointness(episode: dict) -> None:
    support = {int(x) for group in episode["support_indices"] for x in group}
    query = {int(x) for group in episode["query_indices"] for x in group}
    if not support.isdisjoint(query):
        raise ValueError("Support and query indices overlap.")
