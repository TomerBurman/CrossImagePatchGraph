from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import MiniImageNetImageDataset
from .storage import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    copy_file_if_needed,
    copy_tree_update,
    split_cache_complete,
)


def build_dinov2_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@dataclass
class DINOv2Features:
    cls: torch.Tensor
    patches: torch.Tensor
    grid_size: tuple[int, int]


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 ViT-S/14 extractor."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
        device: str | torch.device | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if model_name != "dinov2_vits14":
            raise ValueError("This initial implementation supports dinov2_vits14 only.")
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = 14
        self.embedding_dim = 384
        if image_size % self.patch_size:
            raise ValueError("image_size must be divisible by 14.")
        self.grid_size = (image_size // self.patch_size,) * 2
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_dtype = output_dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, trust_repo=True
        ).to(self.device)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, extraction_batch_size: int = 32, return_cpu: bool = True) -> DINOv2Features:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError(f"Expected [...,3,H,W], received {tuple(images.shape)}.")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} images.")
        leading_shape = images.shape[:-3]
        flat = images.reshape(-1, 3, self.image_size, self.image_size)
        destination = torch.device("cpu") if return_cpu else self.device
        cls_chunks, patch_chunks = [], []
        for start in range(0, len(flat), extraction_batch_size):
            batch = flat[start : start + extraction_batch_size].to(self.device, non_blocking=True)
            context = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
            with context:
                output = self.backbone.forward_features(batch)
            cls_chunks.append(output["x_norm_clstoken"].to(destination, self.output_dtype))
            patch_chunks.append(output["x_norm_patchtokens"].to(destination, self.output_dtype))
        cls = torch.cat(cls_chunks).reshape(*leading_shape, self.embedding_dim)
        patches = torch.cat(patch_chunks).reshape(*leading_shape, self.num_patches, self.embedding_dim)
        return DINOv2Features(cls=cls, patches=patches, grid_size=self.grid_size)


class DINOv2FeatureCacheBuilder:
    """Build local shards and immediately mirror completed files to Drive."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        extractor: DINOv2FeatureExtractor,
        local_cache_dir: str | Path,
        persistent_cache_dir: str | Path,
        images_per_shard: int = 256,
        extraction_batch_size: int = 32,
        num_workers: int = 2,
    ) -> None:
        self.extractor = extractor
        self.local_cache_dir = Path(local_cache_dir)
        self.persistent_cache_dir = Path(persistent_cache_dir)
        self.images_per_shard = images_per_shard
        self.extraction_batch_size = extraction_batch_size
        self.num_workers = num_workers
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "format_version": self.FORMAT_VERSION,
            "model_name": extractor.model_name,
            "image_size": extractor.image_size,
            "patch_size": extractor.patch_size,
            "grid_size": list(extractor.grid_size),
            "num_patches": extractor.num_patches,
            "embedding_dim": extractor.embedding_dim,
            "dtype": str(extractor.output_dtype).replace("torch.", ""),
            "images_per_shard": images_per_shard,
            "preprocessing_id": "bicubic_resize_224_imagenet_normalization_v1",
        }
        metadata_path = self.local_cache_dir / "metadata.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != self.metadata:
                raise ValueError("Existing cache metadata is incompatible with this extractor.")
        else:
            atomic_json_save(self.metadata, metadata_path)
        self._persist_file(metadata_path)

    def _persist_file(self, local_path: Path) -> None:
        persistent = self.persistent_cache_dir / local_path.relative_to(self.local_cache_dir)
        copy_file_if_needed(local_path, persistent)

    def build(self, dataset: MiniImageNetImageDataset) -> pd.DataFrame:
        split_dir = self.local_cache_dir / dataset.split
        split_dir.mkdir(parents=True, exist_ok=True)
        index_path = split_dir / "index.csv"
        if split_cache_complete(self.local_cache_dir, dataset.split, len(dataset)):
            print(f"{dataset.split}: local cache complete; DINOv2 skipped.")
            copy_tree_update(split_dir, self.persistent_cache_dir / dataset.split)
            self._persist_file(self.local_cache_dir / "metadata.json")
            return pd.read_csv(index_path)

        loader = DataLoader(
            dataset,
            batch_size=self.images_per_shard,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.extractor.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )
        records: list[dict] = []
        for shard_id, batch in enumerate(loader):
            shard_name = f"shard_{shard_id:05d}.pt"
            shard_path = split_dir / shard_name
            indices = batch["dataset_index"].long()
            expected = torch.arange(
                shard_id * self.images_per_shard,
                min((shard_id + 1) * self.images_per_shard, len(dataset)),
            )
            if not torch.equal(indices, expected):
                raise RuntimeError("Raw dataset order changed unexpectedly.")
            if shard_path.exists():
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                valid = (
                    torch.equal(shard["dataset_indices"].long(), expected)
                    and tuple(shard["cls"].shape) == (len(expected), self.extractor.embedding_dim)
                    and tuple(shard["patches"].shape)
                    == (len(expected), self.extractor.num_patches, self.extractor.embedding_dim)
                )
                if not valid:
                    raise RuntimeError(f"Incompatible partial shard: {shard_path}")
                status = "reused"
            else:
                features = self.extractor(
                    batch["image"], extraction_batch_size=self.extraction_batch_size, return_cpu=True
                )
                atomic_torch_save(
                    {
                        "dataset_indices": indices.cpu().contiguous(),
                        "cls": features.cls.cpu().contiguous(),
                        "patches": features.patches.cpu().contiguous(),
                    },
                    shard_path,
                )
                status = "created"
            self._persist_file(shard_path)
            for offset, dataset_index in enumerate(indices.tolist()):
                records.append({
                    "dataset_index": dataset_index,
                    "split": dataset.split,
                    "filename": batch["filename"][offset],
                    "label": batch["class_id"][offset],
                    "shard_name": shard_name,
                    "offset": offset,
                })
            print(f"[{dataset.split}] {shard_id + 1:03d}/{len(loader):03d} {status}; persisted")

        index = pd.DataFrame(records).sort_values("dataset_index").reset_index(drop=True)
        if len(index) != len(dataset):
            raise RuntimeError("Cache index is incomplete.")
        atomic_csv_save(index, index_path)
        summary_path = split_dir / "summary.json"
        atomic_json_save({"split": dataset.split, "num_images": len(dataset), "num_shards": len(loader)}, summary_path)
        for path in (index_path, summary_path, self.local_cache_dir / "metadata.json"):
            self._persist_file(path)
        if not split_cache_complete(self.persistent_cache_dir, dataset.split, len(dataset)):
            raise RuntimeError(f"Persistent cache for {dataset.split!r} is incomplete.")
        return index
