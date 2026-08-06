from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool


@dataclass
class CandidateReadoutOutput:
    query_embeddings: torch.Tensor
    support_image_embeddings: torch.Tensor
    prototypes: torch.Tensor
    cosine_similarities: torch.Tensor
    scores: torch.Tensor


class ResidualGraphSAGEBlock(nn.Module):
    """
    One residual GraphSAGE message-passing block.

    Input:
        h:          [num_nodes, hidden_dim]
        edge_index: [2, num_edges]

    Output:
        h:          [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.conv = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr="mean",
            normalize=False,
            root_weight=True,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Residual update:

            message = GraphSAGE(h, edge_index)
            output  = LayerNorm(h + Dropout(GELU(message)))
        """
        message = self.conv(
            h,
            edge_index,
        )

        message = self.activation(message)
        message = self.dropout(message)

        return self.norm(h + message)


class PatchGraphSAGEEncoder(nn.Module):
    """
    Refine patch-node representations with GraphSAGE.

    Default architecture:

        [num_nodes, 387]
            ↓ Linear
        [num_nodes, 256]
            ↓ LayerNorm + GELU + Dropout
        [num_nodes, 256]
            ↓ GraphSAGE block 1
        [num_nodes, 256]
            ↓ GraphSAGE block 2
        [num_nodes, 256]

    The current model uses edge_index but intentionally ignores
    edge_attr. Edge-aware message passing will be implemented as
    a separate model later.
    """

    def __init__(
        self,
        input_dim: int = 387,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                ResidualGraphSAGEBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        graph,
        return_all_layers: bool = False,
    ):
        """
        Parameters
        ----------
        graph:
            PyG Data or Batch object containing:

                graph.x:          [num_nodes, input_dim]
                graph.edge_index: [2, num_edges]

        return_all_layers:
            When True, also return the representation after the
            input projection and after each GraphSAGE layer.

        Returns
        -------
        refined_nodes:
            [num_nodes, hidden_dim]

        layer_outputs, optional:
            List containing num_layers + 1 tensors.
        """
        x = graph.x
        edge_index = graph.edge_index

        if x.ndim != 2:
            raise ValueError(
                "graph.x must have shape [num_nodes, input_dim], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected node feature dimension {self.input_dim}, "
                f"received {x.shape[-1]}."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "graph.edge_index must have shape [2, num_edges], "
                f"received {tuple(edge_index.shape)}."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "graph.edge_index must have dtype torch.long."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "graph.x contains NaN or infinite values."
            )

        # Current graph construction already produces float32.
        # This also protects against accidentally passing cached float16
        # features directly into a float32 model.
        x = x.to(
            dtype=self.input_projection.weight.dtype
        )

        h = self.input_projection(x)
        h = self.input_norm(h)
        h = self.input_activation(h)
        h = self.input_dropout(h)

        layer_outputs = [h]

        for layer in self.layers:
            h = layer(
                h,
                edge_index,
            )

            layer_outputs.append(h)

        if return_all_layers:
            return h, layer_outputs

        return h


class MeanPrototypeCosineReadout(nn.Module):
    """
    Mean-pool refined patch nodes into image embeddings,
    average support-image embeddings into a class prototype,
    and calculate one cosine score per graph.

    Node metadata expected in the PyG Batch:
        graph_batch.batch:    [num_nodes]
        graph_batch.image_id: [num_nodes]

    Local image IDs:
        0       = query image
        1 ... K = support images
    """

    def __init__(
        self,
        temperature: float = 0.1,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        initial_logit_scale = torch.log(
            torch.tensor(
                1.0 / temperature,
                dtype=torch.float32,
            )
        )

        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                initial_logit_scale
            )
        else:
            self.register_buffer(
                "logit_scale",
                initial_logit_scale,
            )

        self.learnable_temperature = (
            learnable_temperature
        )

    @property
    def temperature(self) -> torch.Tensor:
        """
        Return T = 1 / scale.
        """
        return 1.0 / self.logit_scale.exp()

    def forward(
        self,
        refined_nodes: torch.Tensor,
        graph_batch,
    ) -> CandidateReadoutOutput:
        """
        Parameters
        ----------
        refined_nodes:
            Refined node representations:
            [total_nodes, hidden_dim]

        graph_batch:
            PyG Batch containing G graphs.

        Returns
        -------
        CandidateReadoutOutput
        """
        if refined_nodes.ndim != 2:
            raise ValueError(
                "refined_nodes must have shape "
                "[total_nodes, hidden_dim]."
            )

        if refined_nodes.shape[0] != graph_batch.num_nodes:
            raise ValueError(
                "Number of refined nodes does not match "
                "graph_batch.num_nodes."
            )

        if not hasattr(graph_batch, "batch"):
            raise ValueError(
                "graph_batch must contain a node-to-graph "
                "assignment vector named 'batch'."
            )

        if not hasattr(graph_batch, "image_id"):
            raise ValueError(
                "graph_batch must contain image_id metadata."
            )

        if not hasattr(
            graph_batch,
            "num_support_images",
        ):
            raise ValueError(
                "graph_batch must contain "
                "num_support_images metadata."
            )

        if not torch.isfinite(refined_nodes).all():
            raise ValueError(
                "refined_nodes contains NaN or infinity."
            )

        graph_ids = graph_batch.batch.long()
        local_image_ids = graph_batch.image_id.long()

        num_graphs = graph_batch.num_graphs

        support_counts = (
            graph_batch.num_support_images
            .reshape(-1)
            .long()
        )

        if support_counts.numel() != num_graphs:
            raise ValueError(
                "Expected one num_support_images value "
                "for every graph."
            )

        if not torch.all(
            support_counts == support_counts[0]
        ):
            raise ValueError(
                "All graphs in one batch must currently "
                "have the same number of support images."
            )

        num_support_images = int(
            support_counts[0].item()
        )

        num_images_per_graph = (
            num_support_images + 1
        )

        if local_image_ids.min() < 0:
            raise ValueError(
                "image_id values cannot be negative."
            )

        if (
            local_image_ids.max()
            >= num_images_per_graph
        ):
            raise ValueError(
                "image_id contains a value outside the "
                "expected query/support image range."
            )

        # Convert local image IDs into batch-global image IDs.
        #
        # Graph 0:
        #   query=0, supports=1...K
        #
        # Graph 1:
        #   query=K+1, supports=K+2...2(K+1)-1
        #
        # etc.
        global_image_ids = (
            graph_ids * num_images_per_graph
            + local_image_ids
        )

        total_images = (
            num_graphs * num_images_per_graph
        )

        # [G * (K+1), hidden_dim]
        pooled_images = global_mean_pool(
            x=refined_nodes,
            batch=global_image_ids,
            size=total_images,
        )

        # [G, K+1, hidden_dim]
        pooled_images = pooled_images.reshape(
            num_graphs,
            num_images_per_graph,
            refined_nodes.shape[-1],
        )

        # image_id 0 is always the query.
        query_embeddings = pooled_images[:, 0]

        # image_id 1...K are support images.
        support_image_embeddings = (
            pooled_images[:, 1:]
        )

        # [G, hidden_dim]
        prototypes = (
            support_image_embeddings.mean(dim=1)
        )

        normalized_queries = F.normalize(
            query_embeddings,
            p=2,
            dim=-1,
        )

        normalized_prototypes = F.normalize(
            prototypes,
            p=2,
            dim=-1,
        )

        cosine_similarities = (
            normalized_queries
            * normalized_prototypes
        ).sum(dim=-1)

        # Prevent uncontrolled scale growth if temperature
        # becomes learnable later.
        scale = self.logit_scale.exp().clamp(
            max=100.0
        )

        scores = cosine_similarities * scale

        return CandidateReadoutOutput(
            query_embeddings=query_embeddings,
            support_image_embeddings=(
                support_image_embeddings
            ),
            prototypes=prototypes,
            cosine_similarities=(
                cosine_similarities
            ),
            scores=scores,
        )


class CrossImageGraphMatcher(nn.Module):
    """
    Complete trainable graph matching module:

        graph
        → GraphSAGE
        → image mean pooling
        → support prototype
        → cosine score
    """

    def __init__(
        self,
        input_dim: int = 387,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = PatchGraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.readout = MeanPrototypeCosineReadout(
            temperature=temperature,
            learnable_temperature=(
                learnable_temperature
            ),
        )

    def forward(
        self,
        graph_batch,
        return_embeddings: bool = False,
    ):
        refined_nodes = self.encoder(
            graph_batch
        )

        output = self.readout(
            refined_nodes=refined_nodes,
            graph_batch=graph_batch,
        )

        if return_embeddings:
            return output

        return output.scores


class BaselinePreservingResidualMatcher(nn.Module):
    """Combine frozen CLS logits with a learnable graph correction.

    At initialization `residual_scale == 0`, so final logits equal the frozen
    CLS baseline exactly. The scale receives gradients immediately; graph
    parameters begin receiving gradients once the scale moves away from zero.
    """

    def __init__(self, graph_matcher: CrossImageGraphMatcher, initial_scale: float = 0.0) -> None:
        super().__init__()
        self.graph_matcher = graph_matcher
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def graph_scores(self, graph_batch) -> torch.Tensor:
        return self.graph_matcher(graph_batch)

    def combine(self, cls_scores: torch.Tensor, graph_scores: torch.Tensor) -> torch.Tensor:
        if cls_scores.shape != graph_scores.shape:
            raise ValueError("CLS and graph score shapes must match.")
        return cls_scores + self.residual_scale * graph_scores

    def forward(self, graph_batch, cls_scores: torch.Tensor) -> torch.Tensor:
        return self.combine(cls_scores, self.graph_scores(graph_batch))
