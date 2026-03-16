import numpy as np
import torch


UNREACHABLE_NODE_DISTANCE = 510


def floyd_warshall(adjacency_matrix):
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.int64)
    nrows, ncols = adjacency_matrix.shape
    if nrows != ncols:
        raise ValueError("adjacency_matrix must be square.")

    dist = adjacency_matrix.copy(order="C")
    path = np.zeros((nrows, ncols), dtype=np.int64)

    dist[dist == 0] = UNREACHABLE_NODE_DISTANCE
    np.fill_diagonal(dist, 0)

    for k in range(nrows):
        cost_through_k = dist[:, [k]] + dist[[k], :]
        mask = cost_through_k < dist
        path[mask] = k
        dist = np.minimum(dist, cost_through_k)

    unreachable_mask = dist >= UNREACHABLE_NODE_DISTANCE
    path[unreachable_mask] = UNREACHABLE_NODE_DISTANCE
    dist[unreachable_mask] = UNREACHABLE_NODE_DISTANCE
    return dist, path


def collect_path_edges(path, i, j):
    k = int(path[i][j])
    if k == 0:
        return []
    return collect_path_edges(path, i, k) + [k] + collect_path_edges(path, k, j)


def build_edge_input(max_dist, path, edge_feat):
    path = np.asarray(path, dtype=np.int64)
    edge_feat = np.asarray(edge_feat, dtype=np.int64)
    nrows, ncols = path.shape
    if nrows != ncols:
        raise ValueError("path must be square.")

    edge_fea_all = -1 * np.ones((nrows, ncols, int(max_dist), edge_feat.shape[-1]), dtype=np.int64)

    for i in range(nrows):
        for j in range(ncols):
            if i == j or path[i][j] == UNREACHABLE_NODE_DISTANCE:
                continue
            shortest_path = [i] + collect_path_edges(path, i, j) + [j]
            for k in range(len(shortest_path) - 1):
                edge_fea_all[i, j, k, :] = edge_feat[shortest_path[k], shortest_path[k + 1], :]

    return edge_fea_all


def pad_1d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen, padlen], float("-inf"), dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x: torch.Tensor, padlen1: int, padlen2: int, padlen3: int) -> torch.Tensor:
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class GraphBatch:
    __slots__ = (
        "idx",
        "attn_bias",
        "attn_edge_type",
        "spatial_pos",
        "in_degree",
        "out_degree",
        "x",
        "edge_input",
        "y",
    )

    def __init__(
        self,
        idx: torch.Tensor,
        attn_bias: torch.Tensor,
        attn_edge_type: torch.Tensor,
        spatial_pos: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
        x: torch.Tensor,
        edge_input: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self.idx = idx
        self.attn_bias = attn_bias
        self.attn_edge_type = attn_edge_type
        self.spatial_pos = spatial_pos
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.x = x
        self.edge_input = edge_input
        self.y = y

    def to(self, device):
        self.idx = self.idx.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.attn_edge_type = self.attn_edge_type.to(device)
        self.spatial_pos = self.spatial_pos.to(device)
        self.in_degree = self.in_degree.to(device)
        self.out_degree = self.out_degree.to(device)
        self.x = self.x.to(device)
        self.edge_input = self.edge_input.to(device)
        self.y = self.y.to(device)
        return self

    def __len__(self) -> int:
        return int(self.in_degree.size(0))


def collate_graph_batch(items, max_node: int = 512, multi_hop_max_dist: int = 20, spatial_pos_max: int = 20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    if not items:
        raise ValueError("The collator received an empty batch after max_node filtering.")

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
        )
        for item in items
    ]

    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(*items)

    for idx, attn_bias in enumerate(attn_biases):
        attn_bias[1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    return GraphBatch(
        idx=torch.as_tensor(idxs, dtype=torch.long),
        attn_bias=torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]),
        attn_edge_type=torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]),
        spatial_pos=torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]),
        in_degree=torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]),
        out_degree=torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]),
        x=torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]),
        edge_input=torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        ),
        y=torch.cat(ys),
    )
