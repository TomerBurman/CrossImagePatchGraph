from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class BaselineMetrics:
    loss: float
    accuracy: float
    correct: int
    num_queries: int
    num_episodes: int
    episode_accuracy_mean: float
    episode_accuracy_ci95: float

    def to_dict(self) -> dict:
        return asdict(self)


@torch.inference_mode()
def frozen_baseline_episode(
    episode: dict,
    representation: str,
    device: torch.device,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if representation == "cls":
        support = episode["support_cls"].to(device=device, dtype=torch.float32)
        query = episode["query_cls"].to(device=device, dtype=torch.float32)
    elif representation == "mean_patch":
        support = episode["support_patches"].to(device=device, dtype=torch.float32).mean(dim=-2)
        query = episode["query_patches"].to(device=device, dtype=torch.float32).mean(dim=-2)
    else:
        raise ValueError("representation must be 'cls' or 'mean_patch'.")
    prototypes = F.normalize(support.mean(dim=1), p=2, dim=-1)
    queries = F.normalize(query.reshape(-1, query.shape[-1]), p=2, dim=-1)
    logits = (queries @ prototypes.T) / temperature
    targets = torch.as_tensor(episode["query_labels"], dtype=torch.long, device=device).reshape(-1)
    return logits, targets


@torch.inference_mode()
def cls_scores_for_query(
    episode: dict,
    query_class_position: int,
    query_position: int,
    device: torch.device,
    temperature: float = 0.1,
) -> torch.Tensor:
    support = episode["support_cls"].to(device=device, dtype=torch.float32)
    query = episode["query_cls"][query_class_position, query_position].to(device=device, dtype=torch.float32)
    prototypes = F.normalize(support.mean(dim=1), p=2, dim=-1)
    query = F.normalize(query, p=2, dim=-1)
    return (query @ prototypes.T) / temperature


@torch.inference_mode()
def evaluate_frozen_baseline(
    episode_dataset,
    representation: str,
    device: torch.device,
    num_episodes: int,
    temperature: float = 0.1,
) -> BaselineMetrics:
    num_episodes = min(num_episodes, len(episode_dataset))
    total_loss = total_correct = total_queries = 0
    episode_accuracies: list[float] = []
    for episode_index in range(num_episodes):
        logits, targets = frozen_baseline_episode(
            episode_dataset[episode_index], representation, device, temperature
        )
        loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=-1)
        correct = int((predictions == targets).sum().item())
        query_count = targets.numel()
        total_loss += loss.item() * query_count
        total_correct += correct
        total_queries += query_count
        episode_accuracies.append(correct / query_count)
    values = torch.tensor(episode_accuracies, dtype=torch.float32)
    mean = float(values.mean().item())
    ci95 = 1.96 * float(values.std(unbiased=True).item() / math.sqrt(num_episodes)) if num_episodes > 1 else float("nan")
    return BaselineMetrics(
        loss=total_loss / total_queries,
        accuracy=total_correct / total_queries,
        correct=total_correct,
        num_queries=total_queries,
        num_episodes=num_episodes,
        episode_accuracy_mean=mean,
        episode_accuracy_ci95=ci95,
    )
