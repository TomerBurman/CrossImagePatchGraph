import torch
from torch_geometric.data import Batch

from cross_image_glot.graph_builder import ClassConditionedPatchGraphBuilder
from cross_image_glot.models import CrossImageGraphMatcher


def test_matcher_returns_one_score_per_graph():
    builder = ClassConditionedPatchGraphBuilder(grid_size=(3, 3), top_k=2, similarity_device="cpu")
    graphs = [builder.build_graph(torch.randn(9, 8), torch.randn(2, 9, 8)) for _ in range(3)]
    batch = Batch.from_data_list(graphs)
    model = CrossImageGraphMatcher(input_dim=11, hidden_dim=16, num_layers=1, dropout=0.0)
    assert model(batch).shape == (3,)
