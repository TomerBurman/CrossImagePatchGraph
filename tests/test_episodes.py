import torch

from cross_image_glot.data import FewShotFeatureEpisodeDataset, validate_episode_disjointness


class FakeFeatureDataset:
    def __init__(self):
        self.class_ids = [f"c{i}" for i in range(6)]
        self.class_to_indices = {class_id: list(range(i * 20, (i + 1) * 20)) for i, class_id in enumerate(self.class_ids)}

    def indices_for_class(self, class_id):
        return self.class_to_indices[class_id]

    def __getitem__(self, index):
        return {
            "cls": torch.full((4,), float(index)),
            "patches": torch.full((9, 4), float(index)),
            "dataset_index": index,
            "class_id": "unused",
            "filename": str(index),
        }


def test_episode_shapes_and_disjointness():
    dataset = FewShotFeatureEpisodeDataset(FakeFeatureDataset(), 5, 5, 3, 10, 42, False)
    episode = dataset[0]
    assert episode["support_cls"].shape == (5, 5, 4)
    assert episode["query_patches"].shape == (5, 3, 9, 4)
    validate_episode_disjointness(episode)
