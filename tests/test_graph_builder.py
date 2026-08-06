import torch

from cross_image_glot.graph_builder import ClassConditionedPatchGraphBuilder


def test_small_graph_shapes():
    builder = ClassConditionedPatchGraphBuilder(grid_size=(3, 3), top_k=2, similarity_device="cpu")
    query = torch.randn(9, 8)
    support = torch.randn(2, 9, 8)
    graph = builder.build_graph(query, support)
    assert graph.x.shape == (27, 11)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 5
