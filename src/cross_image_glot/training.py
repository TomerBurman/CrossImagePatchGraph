from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch, Data

from .baselines import cls_scores_for_query
from .graph_builder import ClassConditionedPatchGraphBuilder
from .storage import atomic_json_save, atomic_torch_save


@dataclass
class QueryCandidateGraphs:
    graphs: list[Data]
    target: int
    query_class_position: int
    query_position: int


@dataclass
class EpisodeRunResult:
    loss: float
    accuracy: float
    correct: int
    num_queries: int
    logits: torch.Tensor
    targets: torch.Tensor
    gradient_norm: float | None = None


@dataclass
class SplitMetrics:
    loss: float
    accuracy: float
    correct: int
    num_queries: int
    num_episodes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _get_2d_value(values, row: int, column: int):
    return values[row, column] if isinstance(values, torch.Tensor) else values[row][column]


def _to_python_int(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _to_python_int_list(values) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.reshape(-1).tolist()]
    return [int(value) for value in values]


def build_query_candidate_graphs(
    graph_builder: ClassConditionedPatchGraphBuilder,
    episode: dict,
    query_class_position: int,
    query_position: int,
) -> QueryCandidateGraphs:
    support_patches = episode["support_patches"]
    query_patches = episode["query_patches"]
    query_labels = episode["query_labels"]
    n_way = support_patches.shape[0]
    queries_per_class = query_patches.shape[1]
    if not 0 <= query_class_position < n_way or not 0 <= query_position < queries_per_class:
        raise IndexError("Query coordinates are outside the episode.")
    query_data_id = None
    if "query_indices" in episode:
        query_data_id = _to_python_int(_get_2d_value(episode["query_indices"], query_class_position, query_position))
    graphs = []
    for candidate_id in range(n_way):
        support_ids = None
        if "support_indices" in episode:
            support_ids = _to_python_int_list(episode["support_indices"][candidate_id])
        graphs.append(graph_builder.build_graph(
            query_patches=query_patches[query_class_position, query_position],
            support_patches=support_patches[candidate_id],
            candidate_id=candidate_id,
            query_data_id=query_data_id,
            support_data_ids=support_ids,
        ))
    target = _to_python_int(_get_2d_value(query_labels, query_class_position, query_position))
    return QueryCandidateGraphs(graphs, target, query_class_position, query_position)


def score_graph_list(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    graph_microbatch_size: int = 2,
) -> torch.Tensor:
    if graph_microbatch_size <= 0 or not graphs:
        raise ValueError("A positive microbatch size and at least one graph are required.")
    chunks = []
    for start in range(0, len(graphs), graph_microbatch_size):
        graph_chunk = [graph.clone().cpu() for graph in graphs[start : start + graph_microbatch_size]]
        batch = Batch.from_data_list(graph_chunk).to(device)
        scores = model(batch)
        if scores.shape != (len(graph_chunk),):
            raise RuntimeError(f"Expected {(len(graph_chunk),)}, received {tuple(scores.shape)}.")
        chunks.append(scores)
    return torch.cat(chunks)


def train_feature_episode(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_builder: ClassConditionedPatchGraphBuilder,
    episode: dict,
    device: torch.device,
    graph_microbatch_size: int = 2,
    max_gradient_norm: float | None = 1.0,
) -> EpisodeRunResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n_way = episode["support_patches"].shape[0]
    queries_per_class = episode["query_patches"].shape[1]
    num_queries = n_way * queries_per_class
    logits_rows, targets = [], []
    total_loss = 0.0
    correct = 0
    for class_pos in range(n_way):
        for query_pos in range(queries_per_class):
            query_graphs = build_query_candidate_graphs(graph_builder, episode, class_pos, query_pos)
            scores = score_graph_list(model, query_graphs.graphs, device, graph_microbatch_size)
            target = torch.tensor([query_graphs.target], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
            (loss / num_queries).backward()
            correct += int(int(scores.argmax()) == query_graphs.target)
            total_loss += float(loss.detach())
            logits_rows.append(scores.detach().cpu())
            targets.append(query_graphs.target)
    gradient_norm_value = None
    if max_gradient_norm is not None:
        gradient_norm_value = float(clip_grad_norm_(model.parameters(), max_gradient_norm).item())
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {name!r}.")
    optimizer.step()
    return EpisodeRunResult(
        total_loss / num_queries,
        correct / num_queries,
        correct,
        num_queries,
        torch.stack(logits_rows),
        torch.tensor(targets, dtype=torch.long),
        gradient_norm_value,
    )


@torch.inference_mode()
def evaluate_feature_episode(
    model: torch.nn.Module,
    graph_builder: ClassConditionedPatchGraphBuilder,
    episode: dict,
    device: torch.device,
    graph_microbatch_size: int = 2,
) -> EpisodeRunResult:
    model.eval()
    n_way = episode["support_patches"].shape[0]
    queries_per_class = episode["query_patches"].shape[1]
    num_queries = n_way * queries_per_class
    logits_rows, targets = [], []
    total_loss = 0.0
    correct = 0
    for class_pos in range(n_way):
        for query_pos in range(queries_per_class):
            query_graphs = build_query_candidate_graphs(graph_builder, episode, class_pos, query_pos)
            scores = score_graph_list(model, query_graphs.graphs, device, graph_microbatch_size)
            target = torch.tensor([query_graphs.target], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
            correct += int(int(scores.argmax()) == query_graphs.target)
            total_loss += float(loss)
            logits_rows.append(scores.cpu())
            targets.append(query_graphs.target)
    return EpisodeRunResult(
        total_loss / num_queries,
        correct / num_queries,
        correct,
        num_queries,
        torch.stack(logits_rows),
        torch.tensor(targets, dtype=torch.long),
    )


def train_epoch(
    model,
    optimizer,
    graph_builder,
    episode_dataset,
    device,
    epoch: int,
    num_episodes: int,
    graph_microbatch_size: int = 2,
    log_interval: int = 10,
) -> SplitMetrics:
    if hasattr(episode_dataset, "set_epoch"):
        episode_dataset.set_epoch(epoch)
    num_episodes = min(num_episodes, len(episode_dataset))
    total_loss = total_correct = total_queries = 0
    for episode_index in range(num_episodes):
        result = train_feature_episode(
            model, optimizer, graph_builder, episode_dataset[episode_index], device, graph_microbatch_size
        )
        total_loss += result.loss * result.num_queries
        total_correct += result.correct
        total_queries += result.num_queries
        if log_interval > 0 and (episode_index + 1) % log_interval == 0:
            print(f"  train episode {episode_index + 1:4d}/{num_episodes}: loss={total_loss / total_queries:.4f}, accuracy={total_correct / total_queries:.4f}")
    return SplitMetrics(total_loss / total_queries, total_correct / total_queries, total_correct, total_queries, num_episodes)


@torch.inference_mode()
def evaluate_episode_dataset(
    model,
    graph_builder,
    episode_dataset,
    device,
    num_episodes: int,
    graph_microbatch_size: int = 2,
    log_interval: int = 10,
    split_name: str = "validation",
) -> SplitMetrics:
    model.eval()
    num_episodes = min(num_episodes, len(episode_dataset))
    total_loss = total_correct = total_queries = 0
    for episode_index in range(num_episodes):
        result = evaluate_feature_episode(
            model, graph_builder, episode_dataset[episode_index], device, graph_microbatch_size
        )
        total_loss += result.loss * result.num_queries
        total_correct += result.correct
        total_queries += result.num_queries
        if log_interval > 0 and (episode_index + 1) % log_interval == 0:
            print(f"  {split_name} episode {episode_index + 1:4d}/{num_episodes}: loss={total_loss / total_queries:.4f}, accuracy={total_correct / total_queries:.4f}")
    return SplitMetrics(total_loss / total_queries, total_correct / total_queries, total_correct, total_queries, num_episodes)


# Residual CLS + graph training ------------------------------------------------
def train_residual_feature_episode(
    model,
    optimizer,
    graph_builder,
    episode,
    device,
    graph_microbatch_size: int = 2,
    cls_temperature: float = 0.1,
    max_gradient_norm: float | None = 1.0,
) -> EpisodeRunResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n_way = episode["support_patches"].shape[0]
    queries_per_class = episode["query_patches"].shape[1]
    num_queries = n_way * queries_per_class
    logits_rows, targets = [], []
    total_loss = 0.0
    correct = 0
    for class_pos in range(n_way):
        for query_pos in range(queries_per_class):
            query_graphs = build_query_candidate_graphs(graph_builder, episode, class_pos, query_pos)
            graph_scores = score_graph_list(model.graph_matcher, query_graphs.graphs, device, graph_microbatch_size)
            cls_scores = cls_scores_for_query(episode, class_pos, query_pos, device, cls_temperature)
            scores = model.combine(cls_scores, graph_scores)
            target = torch.tensor([query_graphs.target], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
            (loss / num_queries).backward()
            correct += int(int(scores.argmax()) == query_graphs.target)
            total_loss += float(loss.detach())
            logits_rows.append(scores.detach().cpu())
            targets.append(query_graphs.target)
    gradient_norm = None
    if max_gradient_norm is not None:
        gradient_norm = float(clip_grad_norm_(model.parameters(), max_gradient_norm).item())
    optimizer.step()
    return EpisodeRunResult(total_loss / num_queries, correct / num_queries, correct, num_queries, torch.stack(logits_rows), torch.tensor(targets), gradient_norm)


@torch.inference_mode()
def evaluate_residual_feature_episode(
    model,
    graph_builder,
    episode,
    device,
    graph_microbatch_size: int = 2,
    cls_temperature: float = 0.1,
) -> EpisodeRunResult:
    model.eval()
    n_way = episode["support_patches"].shape[0]
    queries_per_class = episode["query_patches"].shape[1]
    num_queries = n_way * queries_per_class
    logits_rows, targets = [], []
    total_loss = 0.0
    correct = 0
    for class_pos in range(n_way):
        for query_pos in range(queries_per_class):
            query_graphs = build_query_candidate_graphs(graph_builder, episode, class_pos, query_pos)
            graph_scores = score_graph_list(model.graph_matcher, query_graphs.graphs, device, graph_microbatch_size)
            cls_scores = cls_scores_for_query(episode, class_pos, query_pos, device, cls_temperature)
            scores = model.combine(cls_scores, graph_scores)
            target = torch.tensor([query_graphs.target], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
            correct += int(int(scores.argmax()) == query_graphs.target)
            total_loss += float(loss)
            logits_rows.append(scores.cpu())
            targets.append(query_graphs.target)
    return EpisodeRunResult(total_loss / num_queries, correct / num_queries, correct, num_queries, torch.stack(logits_rows), torch.tensor(targets))


def train_residual_epoch(model, optimizer, graph_builder, episode_dataset, device, epoch, num_episodes, graph_microbatch_size=2, cls_temperature=0.1, log_interval=10) -> SplitMetrics:
    if hasattr(episode_dataset, "set_epoch"):
        episode_dataset.set_epoch(epoch)
    num_episodes = min(num_episodes, len(episode_dataset))
    total_loss = total_correct = total_queries = 0
    for i in range(num_episodes):
        result = train_residual_feature_episode(model, optimizer, graph_builder, episode_dataset[i], device, graph_microbatch_size, cls_temperature)
        total_loss += result.loss * result.num_queries
        total_correct += result.correct
        total_queries += result.num_queries
        if log_interval > 0 and (i + 1) % log_interval == 0:
            print(f"  train episode {i + 1:4d}/{num_episodes}: loss={total_loss / total_queries:.4f}, accuracy={total_correct / total_queries:.4f}, residual_scale={model.residual_scale.item():.5f}")
    return SplitMetrics(total_loss / total_queries, total_correct / total_queries, total_correct, total_queries, num_episodes)


@torch.inference_mode()
def evaluate_residual_dataset(model, graph_builder, episode_dataset, device, num_episodes, graph_microbatch_size=2, cls_temperature=0.1, log_interval=10, split_name="validation") -> SplitMetrics:
    num_episodes = min(num_episodes, len(episode_dataset))
    total_loss = total_correct = total_queries = 0
    for i in range(num_episodes):
        result = evaluate_residual_feature_episode(model, graph_builder, episode_dataset[i], device, graph_microbatch_size, cls_temperature)
        total_loss += result.loss * result.num_queries
        total_correct += result.correct
        total_queries += result.num_queries
        if log_interval > 0 and (i + 1) % log_interval == 0:
            print(f"  {split_name} episode {i + 1:4d}/{num_episodes}: loss={total_loss / total_queries:.4f}, accuracy={total_correct / total_queries:.4f}")
    return SplitMetrics(total_loss / total_queries, total_correct / total_queries, total_correct, total_queries, num_episodes)


# Checkpointing ----------------------------------------------------------------
def save_checkpoint_atomic(checkpoint: dict, output_path: Path) -> None:
    atomic_torch_save(checkpoint, output_path)


def make_checkpoint(model, optimizer, epoch, best_validation_accuracy, epochs_without_improvement, history, configuration) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_accuracy": best_validation_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "configuration": configuration,
    }


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_training_checkpoint(checkpoint_path: Path, model, optimizer, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    return checkpoint


def save_history(history: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
