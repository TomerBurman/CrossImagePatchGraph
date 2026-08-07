from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED_IMAGES = {"train": 38_400, "val": 9_600, "test": 12_000}
EXPECTED_CLASSES = {"train": 64, "val": 16, "test": 20}


@dataclass(frozen=True)
class ProjectPaths:
    """Persistent Drive paths and fast local Colab paths."""

    drive_root: Path = Path("/content/drive/MyDrive/CrossImagePatchGraph")
    local_root: Path = Path("/content/CrossImagePatchGraph_runtime")
    feature_cache_name: str = "dinov2_vits14_224"

    @property
    def drive_data_dir(self) -> Path:
        return self.drive_root / "data"

    @property
    def drive_feature_dir(self) -> Path:
        return self.drive_root / "features" / self.feature_cache_name

    @property
    def drive_checkpoint_dir(self) -> Path:
        return self.drive_root / "checkpoints"

    @property
    def drive_results_dir(self) -> Path:
        return self.drive_root / "results"

    @property
    def local_data_dir(self) -> Path:
        return self.local_root / "data"

    @property
    def local_image_dir(self) -> Path:
        return self.local_data_dir / "images" / "images"

    @property
    def local_feature_dir(self) -> Path:
        return self.local_root / "features" / self.feature_cache_name

    def ensure_directories(self) -> None:
        for directory in (
            self.drive_data_dir,
            self.drive_feature_dir,
            self.drive_checkpoint_dir,
            self.drive_results_dir,
            self.local_data_dir,
            self.local_feature_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = ProjectPaths()
