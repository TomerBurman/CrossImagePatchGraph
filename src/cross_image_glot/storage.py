from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch

from .config import EXPECTED_IMAGES

DATASET_FILES = ("train.csv", "val.csv", "test.csv", "images.zip")


def copy_file_if_needed(source: Path, destination: Path) -> bool:
    """Atomically copy a file when the destination is absent or size-mismatched."""
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return False
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return True


def copy_tree_update(source_root: Path, destination_root: Path) -> int:
    """Copy missing or size-mismatched files while preserving the tree."""
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    if not source_root.exists():
        return 0
    copied = 0
    for source in source_root.rglob("*"):
        if not source.is_file() or source.name.endswith(".tmp"):
            continue
        destination = destination_root / source.relative_to(source_root)
        copied += int(copy_file_if_needed(source, destination))
    return copied


def atomic_torch_save(value: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv_save(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def split_cache_complete(cache_dir: str | Path, split: str, expected_images: int | None = None) -> bool:
    cache_dir = Path(cache_dir)
    expected_images = EXPECTED_IMAGES[split] if expected_images is None else expected_images
    split_dir = cache_dir / split
    index_path = split_dir / "index.csv"
    summary_path = split_dir / "summary.json"
    if not (cache_dir / "metadata.json").exists() or not index_path.exists() or not summary_path.exists():
        return False
    try:
        index = pd.read_csv(index_path)
    except Exception:
        return False
    required = {"dataset_index", "split", "shard_name", "offset", "label", "filename"}
    if not required.issubset(index.columns):
        return False
    if len(index) != expected_images or index["dataset_index"].nunique() != expected_images:
        return False
    if set(index["split"].astype(str)) != {split}:
        return False
    return all((split_dir / name).exists() for name in index["shard_name"].unique())


def restore_feature_splits(
    splits: list[str] | tuple[str, ...],
    persistent_root: Path,
    local_root: Path,
) -> int:
    """Restore only the feature splits required by the current notebook."""
    persistent_root = Path(persistent_root)
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    metadata = persistent_root / "metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"Persistent feature metadata is missing: {metadata}")
    copied += int(copy_file_if_needed(metadata, local_root / "metadata.json"))
    for split in splits:
        if split not in EXPECTED_IMAGES:
            raise ValueError(f"Unknown split: {split}")
        source = persistent_root / split
        if not source.exists():
            raise FileNotFoundError(f"Persistent split is missing: {source}")
        copied += copy_tree_update(source, local_root / split)
        if not split_cache_complete(local_root, split):
            raise RuntimeError(f"Local cache restore for {split!r} is incomplete.")
    return copied


def download_shared_drive_folder_files(
    shared_folder_id: str,
    destination_dir: Path,
    filenames: tuple[str, ...] = DATASET_FILES,
) -> None:
    """Download missing files from a shared Google Drive folder into MyDrive.

    This function imports Colab/Google API packages lazily and is intended only
    for notebook 00.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    missing = {name for name in filenames if not (destination_dir / name).exists()}
    if not missing:
        print("Persistent dataset archive already exists in Drive.")
        return

    from google.colab import auth  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore

    auth.authenticate_user()
    service = build("drive", "v3")
    result = service.files().list(
        q=f"'{shared_folder_id}' in parents and trashed = false",
        fields="files(id, name)",
    ).execute()
    available = {item["name"]: item["id"] for item in result.get("files", [])}
    unavailable = missing - set(available)
    if unavailable:
        raise FileNotFoundError(f"Files missing from shared folder: {sorted(unavailable)}")

    for name in sorted(missing):
        destination = destination_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        request = service.files().get_media(fileId=available[name])
        with temporary.open("wb") as output:
            downloader = MediaIoBaseDownload(output, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        os.replace(temporary, destination)
        print(f"Persisted {name} to {destination}")


def restore_and_extract_images(
    persistent_data_dir: Path,
    local_data_dir: Path,
    expected_image_count: int = 60_000,
) -> Path:
    """Copy the persistent archive/manifests locally and extract JPEGs locally."""
    persistent_data_dir = Path(persistent_data_dir)
    local_data_dir = Path(local_data_dir)
    local_data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        copy_file_if_needed(persistent_data_dir / name, local_data_dir / name)

    image_dir = local_data_dir / "images" / "images"
    marker = local_data_dir / ".miniimagenet_extracted"
    count = sum(1 for _ in image_dir.glob("*.jpg")) if image_dir.exists() else 0
    if not marker.exists() or count != expected_image_count:
        extraction_root = local_data_dir / "images"
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
        extraction_root.mkdir(parents=True, exist_ok=True)
        with ZipFile(local_data_dir / "images.zip", "r") as archive:
            archive.extractall(extraction_root)
        count = sum(1 for _ in image_dir.glob("*.jpg"))
        if count != expected_image_count:
            raise RuntimeError(f"Expected {expected_image_count:,} images, found {count:,}.")
        marker.write_text(f"{count}\n", encoding="utf-8")
    return image_dir
