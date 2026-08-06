from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class EpisodePatchGraphs:
    graphs: list[Data]
    targets: torch.Tensor
    num_queries: int
    num_candidates: int


class ClassConditionedPatchGraphBuilder:
    """
    Construct one graph from:

        one query image
        one candidate class containing K support images

    Node order
    ----------
    Query patches:
        nodes [0, P)

    Support image 0:
        nodes [P, 2P)

    Support image 1:
        nodes [2P, 3P)

    ...

    Support image K-1:
        nodes [KP, (K+1)P)

    Initial node feature
    --------------------
    x_i = [DINO_patch_i, normalized_row_i,
          normalized_column_i, is_query_i]

    With DINOv2 ViT-S/14:
        D = 384
        P = 256
        x dimension = 384 + 2 + 1 = 387

    Edge attributes
    ---------------
    edge_attr[:, 0] = cosine similarity
    edge_attr[:, 1] = is spatial edge
    edge_attr[:, 2] = is semantic cross-image edge
    edge_attr[:, 3] = relative grid-row displacement
    edge_attr[:, 4] = relative grid-column displacement
    """

    SPATIAL_EDGE = 0
    SEMANTIC_EDGE = 1

    def __init__(
        self,
        grid_size: tuple[int, int] = (16, 16),
        top_k: int = 10,
        min_similarity: float | None = None,
        graph_dtype: torch.dtype = torch.float32,
        similarity_device: str | torch.device | None = None,
    ) -> None:
        grid_height, grid_width = grid_size

        if grid_height <= 0 or grid_width <= 0:
            raise ValueError(
                f"Invalid grid size: {grid_size}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if (
            min_similarity is not None
            and not -1.0 <= min_similarity <= 1.0
        ):
            raise ValueError(
                "min_similarity must be between -1 and 1."
            )

        self.grid_size = grid_size
        self.num_patches = grid_height * grid_width
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.graph_dtype = graph_dtype

        if similarity_device is None:
            similarity_device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.similarity_device = torch.device(
            similarity_device
        )

        self.coordinates = self._build_coordinates()

        (
            self.single_image_spatial_edges,
            self.single_image_displacements,
        ) = self._build_spatial_template()

    def _build_coordinates(self) -> torch.Tensor:
        """
        Create normalized grid coordinates.

        Returns
        -------
        Tensor[P, 2]

        Coordinates are ordered in the same row-major order
        as the DINOv2 patch tokens.
        """
        grid_height, grid_width = self.grid_size

        row_denominator = max(grid_height - 1, 1)
        column_denominator = max(grid_width - 1, 1)

        rows = (
            torch.arange(
                grid_height,
                dtype=torch.float32,
            )
            / row_denominator
        )

        columns = (
            torch.arange(
                grid_width,
                dtype=torch.float32,
            )
            / column_denominator
        )

        row_grid, column_grid = torch.meshgrid(
            rows,
            columns,
            indexing="ij",
        )

        coordinates = torch.stack(
            [
                row_grid.reshape(-1),
                column_grid.reshape(-1),
            ],
            dim=-1,
        )

        return coordinates

    def _build_spatial_template(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct bidirectional four-neighbour edges for one image.

        Every undirected relationship is represented by two entries:

            u -> v
            v -> u
        """
        grid_height, grid_width = self.grid_size

        sources: list[int] = []
        targets: list[int] = []
        displacements: list[list[float]] = []

        for row in range(grid_height):
            for column in range(grid_width):
                current = row * grid_width + column

                # Horizontal relationship:
                # current <-> patch to the right
                if column + 1 < grid_width:
                    right = current + 1

                    sources.extend([current, right])
                    targets.extend([right, current])

                    displacements.extend(
                        [
                            [0.0, 1.0],
                            [0.0, -1.0],
                        ]
                    )

                # Vertical relationship:
                # current <-> patch below
                if row + 1 < grid_height:
                    below = (
                        (row + 1) * grid_width
                        + column
                    )

                    sources.extend([current, below])
                    targets.extend([below, current])

                    displacements.extend(
                        [
                            [1.0, 0.0],
                            [-1.0, 0.0],
                        ]
                    )

        edge_index = torch.tensor(
            [sources, targets],
            dtype=torch.long,
        )

        displacement = torch.tensor(
            displacements,
            dtype=torch.float32,
        )

        return edge_index, displacement

    def _validate_inputs(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> None:
        if query_patches.ndim != 2:
            raise ValueError(
                "query_patches must have shape [P, D], "
                f"received {tuple(query_patches.shape)}."
            )

        if support_patches.ndim != 3:
            raise ValueError(
                "support_patches must have shape [K, P, D], "
                f"received {tuple(support_patches.shape)}."
            )

        if query_patches.shape[0] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} query patches, "
                f"received {query_patches.shape[0]}."
            )

        if support_patches.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} support patches "
                f"per image, received "
                f"{support_patches.shape[1]}."
            )

        if (
            query_patches.shape[-1]
            != support_patches.shape[-1]
        ):
            raise ValueError(
                "Query and support embedding dimensions differ: "
                f"{query_patches.shape[-1]} versus "
                f"{support_patches.shape[-1]}."
            )

        if support_patches.shape[0] <= 0:
            raise ValueError(
                "At least one support image is required."
            )

        if not torch.isfinite(query_patches).all():
            raise ValueError(
                "Query patches contain NaN or infinity."
            )

        if not torch.isfinite(support_patches).all():
            raise ValueError(
                "Support patches contain NaN or infinity."
            )

    def _build_spatial_edges(
        self,
        patch_features: torch.Tensor,
        num_images: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Add four-neighbour edges independently to every image.
        """
        normalized_features = F.normalize(
            patch_features,
            p=2,
            dim=-1,
        )

        edge_indices = []
        edge_attributes = []
        edge_types = []

        base_edges = self.single_image_spatial_edges
        displacement = self.single_image_displacements

        for image_id in range(num_images):
            offset = image_id * self.num_patches

            image_edges = base_edges + offset
            source, target = image_edges

            cosine_similarity = (
                normalized_features[source]
                * normalized_features[target]
            ).sum(dim=-1)

            spatial_indicator = torch.ones_like(
                cosine_similarity
            )

            semantic_indicator = torch.zeros_like(
                cosine_similarity
            )

            image_edge_attributes = torch.stack(
                [
                    cosine_similarity,
                    spatial_indicator,
                    semantic_indicator,
                    displacement[:, 0],
                    displacement[:, 1],
                ],
                dim=-1,
            )

            edge_indices.append(image_edges)
            edge_attributes.append(
                image_edge_attributes
            )

            edge_types.append(
                torch.full(
                    size=(image_edges.shape[1],),
                    fill_value=self.SPATIAL_EDGE,
                    dtype=torch.long,
                )
            )

        return (
            torch.cat(edge_indices, dim=1),
            torch.cat(edge_attributes, dim=0),
            torch.cat(edge_types, dim=0),
        )

    def _build_semantic_edges(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Connect each query patch to its top-k support patches.

        The search is performed across all K support images of
        the candidate class:

            support: [K, P, D] -> [K*P, D]

        Similarity matrix:

            [P, D] @ [D, K*P] -> [P, K*P]
        """
        query = query_patches.to(
            device=self.similarity_device,
            dtype=torch.float32,
        )

        support = support_patches.reshape(
            -1,
            support_patches.shape[-1],
        ).to(
            device=self.similarity_device,
            dtype=torch.float32,
        )

        normalized_query = F.normalize(
            query,
            p=2,
            dim=-1,
        )

        normalized_support = F.normalize(
            support,
            p=2,
            dim=-1,
        )

        similarity_matrix = (
            normalized_query
            @ normalized_support.T
        )

        effective_top_k = min(
            self.top_k,
            normalized_support.shape[0],
        )

        selected_similarities, selected_support = (
            torch.topk(
                similarity_matrix,
                k=effective_top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
        )

        query_nodes = torch.arange(
            self.num_patches,
            device=self.similarity_device,
        ).unsqueeze(1).expand_as(selected_support)

        if self.min_similarity is not None:
            keep = (
                selected_similarities
                >= self.min_similarity
            )

            query_nodes = query_nodes[keep]
            selected_support = selected_support[keep]
            selected_similarities = (
                selected_similarities[keep]
            )

        else:
            query_nodes = query_nodes.reshape(-1)
            selected_support = (
                selected_support.reshape(-1)
            )
            selected_similarities = (
                selected_similarities.reshape(-1)
            )

        if selected_similarities.numel() == 0:
            raise RuntimeError(
                "No semantic edges survived graph construction. "
                "Reduce min_similarity or disable the threshold."
            )

        # Query occupies nodes [0, P).
        # Flattened support occupies nodes [P, P + K*P).
        global_support_nodes = (
            selected_support + self.num_patches
        )

        query_to_support = torch.stack(
            [
                query_nodes,
                global_support_nodes,
            ],
            dim=0,
        )

        support_to_query = torch.stack(
            [
                global_support_nodes,
                query_nodes,
            ],
            dim=0,
        )

        semantic_edge_index = torch.cat(
            [
                query_to_support,
                support_to_query,
            ],
            dim=1,
        ).cpu()

        # Both directions use the same visual similarity.
        bidirectional_similarities = torch.cat(
            [
                selected_similarities,
                selected_similarities,
            ],
            dim=0,
        ).cpu()

        zeros = torch.zeros_like(
            bidirectional_similarities
        )

        semantic_edge_attributes = torch.stack(
            [
                bidirectional_similarities,
                zeros,
                torch.ones_like(
                    bidirectional_similarities
                ),
                zeros,
                zeros,
            ],
            dim=-1,
        )

        semantic_edge_types = torch.full(
            size=(semantic_edge_index.shape[1],),
            fill_value=self.SEMANTIC_EDGE,
            dtype=torch.long,
        )

        return (
            semantic_edge_index,
            semantic_edge_attributes,
            semantic_edge_types,
            selected_similarities.detach().cpu(),
        )

    def build_graph(
        self,
        query_patches: torch.Tensor,
        support_patches: torch.Tensor,
        candidate_id: int | None = None,
        query_data_id: int | None = None,
        support_data_ids: Sequence[int] | None = None,
    ) -> Data:
        """
        Construct one query-candidate-class graph.

        Parameters
        ----------
        query_patches:
            [P, D]

        support_patches:
            [K, P, D]

        candidate_id:
            Candidate position in the current episode.
            This is metadata only and is never concatenated
            into node features.

        query_data_id, support_data_ids:
            Dataset-row identifiers used only for debugging and
            leakage verification.
        """
        self._validate_inputs(
            query_patches,
            support_patches,
        )

        # Cached embeddings are float16. Graph/GNN calculations
        # initially use float32.
        query_patches = (
            query_patches
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

        support_patches = (
            support_patches
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

        num_support_images = support_patches.shape[0]
        embedding_dim = query_patches.shape[-1]
        num_images = num_support_images + 1

        flattened_support = support_patches.reshape(
            num_support_images * self.num_patches,
            embedding_dim,
        )

        # [P + K*P, D]
        patch_features = torch.cat(
            [
                query_patches,
                flattened_support,
            ],
            dim=0,
        )

        num_nodes = patch_features.shape[0]

        # Repeat the same 16x16 coordinate system for each image.
        positions = self.coordinates.repeat(
            num_images,
            1,
        )

        node_is_query = torch.zeros(
            num_nodes,
            dtype=torch.bool,
        )

        node_is_query[: self.num_patches] = True

        query_indicator = (
            node_is_query
            .to(torch.float32)
            .unsqueeze(-1)
        )

        # 0 = query
        # 1...K = support images
        image_id = torch.arange(
            num_images,
            dtype=torch.long,
        ).repeat_interleave(self.num_patches)

        patch_id = torch.arange(
            self.num_patches,
            dtype=torch.long,
        ).repeat(num_images)

        # [num_nodes, D + 2 + 1]
        x = torch.cat(
            [
                patch_features,
                positions,
                query_indicator,
            ],
            dim=-1,
        ).to(self.graph_dtype)

        (
            spatial_edge_index,
            spatial_edge_attr,
            spatial_edge_type,
        ) = self._build_spatial_edges(
            patch_features=patch_features,
            num_images=num_images,
        )

        (
            semantic_edge_index,
            semantic_edge_attr,
            semantic_edge_type,
            selected_similarities,
        ) = self._build_semantic_edges(
            query_patches=query_patches,
            support_patches=support_patches,
        )

        edge_index = torch.cat(
            [
                spatial_edge_index,
                semantic_edge_index,
            ],
            dim=1,
        )

        edge_attr = torch.cat(
            [
                spatial_edge_attr,
                semantic_edge_attr,
            ],
            dim=0,
        ).to(self.graph_dtype)

        edge_type = torch.cat(
            [
                spatial_edge_type,
                semantic_edge_type,
            ],
            dim=0,
        )

        graph_arguments = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_type": edge_type,
            "pos": positions.to(self.graph_dtype),
            "node_is_query": node_is_query,
            "image_id": image_id,
            "patch_id": patch_id,

            # Graph-level metadata:
            "num_support_images": torch.tensor(
                [num_support_images],
                dtype=torch.long,
            ),
            "semantic_similarity_mean": torch.tensor(
                [selected_similarities.mean().item()],
                dtype=torch.float32,
            ),
            "semantic_similarity_min": torch.tensor(
                [selected_similarities.min().item()],
                dtype=torch.float32,
            ),
            "semantic_similarity_max": torch.tensor(
                [selected_similarities.max().item()],
                dtype=torch.float32,
            ),
        }

        if candidate_id is not None:
            graph_arguments["candidate_id"] = torch.tensor(
                [candidate_id],
                dtype=torch.long,
            )

        if query_data_id is not None:
            graph_arguments["query_data_id"] = torch.tensor(
                [query_data_id],
                dtype=torch.long,
            )

        if support_data_ids is not None:
            if (
                len(support_data_ids)
                != num_support_images
            ):
                raise ValueError(
                    "support_data_ids length must equal "
                    "the number of support images."
                )

            graph_arguments["support_data_ids"] = (
                torch.tensor(
                    list(support_data_ids),
                    dtype=torch.long,
                )
            )

        return Data(**graph_arguments)

    def build_episode_graphs(
        self,
        episode: dict,
    ) -> EpisodePatchGraphs:
        """
        Construct all query-candidate graphs in one episode.

        Input shapes
        ------------
        support_patches:
            [N, K, P, D]

        query_patches:
            [N, Q, P, D]

        Output graph count
        ------------------
            N * Q * N

        For 5-way and one query per class:
            5 * 1 * 5 = 25 graphs
        """
        support_patches = episode["support_patches"]
        query_patches = episode["query_patches"]
        query_labels = episode["query_labels"]

        if support_patches.ndim != 4:
            raise ValueError(
                "support_patches must have shape [N,K,P,D]."
            )

        if query_patches.ndim != 4:
            raise ValueError(
                "query_patches must have shape [N,Q,P,D]."
            )

        (
            n_way,
            k_shot,
            support_patch_count,
            embedding_dim,
        ) = support_patches.shape

        (
            query_class_count,
            queries_per_class,
            query_patch_count,
            query_embedding_dim,
        ) = query_patches.shape

        if query_class_count != n_way:
            raise ValueError(
                "Support and query class counts differ."
            )

        if support_patch_count != query_patch_count:
            raise ValueError(
                "Support and query patch counts differ."
            )

        if embedding_dim != query_embedding_dim:
            raise ValueError(
                "Support and query embedding dimensions differ."
            )

        graphs: list[Data] = []

        # The labels are read only for the eventual loss.
        # They are not passed into build_graph().
        targets = query_labels.reshape(-1).long().cpu()

        for query_class_position in range(n_way):
            for query_position in range(
                queries_per_class
            ):
                query_data_id = None

                if "query_indices" in episode:
                    query_data_id = int(
                        episode["query_indices"]
                        [query_class_position]
                        [query_position]
                    )

                for candidate_id in range(n_way):
                    support_data_ids = None

                    if "support_indices" in episode:
                        support_data_ids = (
                            episode["support_indices"]
                            [candidate_id]
                        )

                    graph = self.build_graph(
                        query_patches=(
                            query_patches[
                                query_class_position,
                                query_position,
                            ]
                        ),
                        support_patches=(
                            support_patches[candidate_id]
                        ),
                        candidate_id=candidate_id,
                        query_data_id=query_data_id,
                        support_data_ids=(
                            support_data_ids
                        ),
                    )

                    graphs.append(graph)

        num_queries = n_way * queries_per_class

        expected_graph_count = (
            num_queries * n_way
        )

        if len(graphs) != expected_graph_count:
            raise RuntimeError(
                f"Created {len(graphs)} graphs; "
                f"expected {expected_graph_count}."
            )

        return EpisodePatchGraphs(
            graphs=graphs,
            targets=targets,
            num_queries=num_queries,
            num_candidates=n_way,
        )


def validate_patch_graph(
    graph: Data,
    builder: ClassConditionedPatchGraphBuilder,
) -> None:
    num_support_images = int(
        graph.num_support_images.item()
    )

    num_images = num_support_images + 1
    num_patches = builder.num_patches

    expected_nodes = num_images * num_patches

    grid_height, grid_width = builder.grid_size

    # Directed four-neighbour entries for one image:
    spatial_edges_per_image = 2 * (
        grid_height * (grid_width - 1)
        + grid_width * (grid_height - 1)
    )

    expected_spatial_edges = (
        num_images * spatial_edges_per_image
    )

    expected_semantic_edges = 2 * (
        num_patches
        * min(
            builder.top_k,
            num_support_images * num_patches,
        )
    )

    assert graph.num_nodes == expected_nodes

    assert graph.x.shape == (
        expected_nodes,
        384 + 2 + 1,
    )

    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 5

    assert graph.edge_index.dtype == torch.long
    assert graph.edge_type.dtype == torch.long

    assert graph.edge_index.min() >= 0
    assert graph.edge_index.max() < graph.num_nodes

    assert torch.isfinite(graph.x).all()
    assert torch.isfinite(graph.edge_attr).all()

    # No self-loops were explicitly added.
    source, target = graph.edge_index

    assert not torch.any(source == target)

    spatial_mask = (
        graph.edge_type
        == builder.SPATIAL_EDGE
    )

    semantic_mask = (
        graph.edge_type
        == builder.SEMANTIC_EDGE
    )

    actual_spatial_edges = int(
        spatial_mask.sum()
    )

    actual_semantic_edges = int(
        semantic_mask.sum()
    )

    assert (
        actual_spatial_edges
        == expected_spatial_edges
    )

    if builder.min_similarity is None:
        assert (
            actual_semantic_edges
            == expected_semantic_edges
        )

    # Spatial edges must stay inside one image.
    spatial_source = source[spatial_mask]
    spatial_target = target[spatial_mask]

    assert torch.equal(
        graph.image_id[spatial_source],
        graph.image_id[spatial_target],
    )

    # Semantic edges must connect exactly one query node
    # and one support node.
    semantic_source = source[semantic_mask]
    semantic_target = target[semantic_mask]

    semantic_source_is_query = (
        graph.node_is_query[semantic_source]
    )

    semantic_target_is_query = (
        graph.node_is_query[semantic_target]
    )

    assert torch.all(
        semantic_source_is_query
        ^ semantic_target_is_query
    )

    # Each image contributes exactly P nodes.
    nodes_per_image = torch.bincount(
        graph.image_id,
        minlength=num_images,
    )

    assert torch.all(
        nodes_per_image == num_patches
    )

    # Query indicator is consistent in both metadata
    # and the final x feature.
    assert int(graph.node_is_query.sum()) == num_patches

    assert torch.equal(
        graph.x[:, -1].bool(),
        graph.node_is_query,
    )

    # Labels and original class names are not node features.
    assert getattr(graph, "y", None) is None
    assert getattr(graph, "class_ids", None) is None

    print("Graph validation passed.")
    print("Nodes:", graph.num_nodes)
    print("Node feature dimension:", graph.x.shape[1])
    print("Spatial edges:", actual_spatial_edges)
    print("Semantic edges:", actual_semantic_edges)
    print("Total edges:", graph.edge_index.shape[1])
    print(
        "Selected semantic similarity:",
        {
            "min": round(
                graph.semantic_similarity_min.item(),
                4,
            ),
            "mean": round(
                graph.semantic_similarity_mean.item(),
                4,
            ),
            "max": round(
                graph.semantic_similarity_max.item(),
                4,
            ),
        },
    )

