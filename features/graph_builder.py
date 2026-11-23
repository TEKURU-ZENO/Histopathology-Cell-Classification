import torch
import numpy as np
from scipy.spatial import KDTree

def build_spatial_knn_graph(centroids, k=8, eps=1e-7):
    """
    Builds a symmetrized k-NN spatial graph from nuclear centroids.
    
    Args:
        centroids: (N, 2) array or tensor of (x, y) coordinates
        k: number of nearest neighbors (default: 8)
        eps: small epsilon to avoid divide-by-zero
        
    Returns:
        edge_index: (2, E) torch.LongTensor of directed edges (undirected graph)
        edge_attr: (E, 3) torch.FloatTensor of [distance, dx, dy]
    """
    if isinstance(centroids, torch.Tensor):
        c_np = centroids.detach().cpu().numpy()
    else:
        c_np = np.asarray(centroids, dtype=np.float32)

    N = len(c_np)
    if N <= 1:
        # Trivial graph for isolated nodes
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
        return edge_index, edge_attr

    actual_k = min(k + 1, N)  # include self in query, then drop self
    tree = KDTree(c_np)
    distances, indices = tree.query(c_np, k=actual_k)

    # Collect directed edges
    edges_set = set()
    for i in range(N):
        for j in indices[i]:
            if i != j:
                # Add both (i, j) and (j, i) to symmetrize
                edges_set.add((i, j))
                edges_set.add((j, i))

    # Convert to sorted list of edges
    edges_list = sorted(list(edges_set))
    if len(edges_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
        return edge_index, edge_attr

    src = [e[0] for e in edges_list]
    dst = [e[1] for e in edges_list]

    c_src = c_np[src]
    c_dst = c_np[dst]

    delta = c_dst - c_src  # (dx, dy) = (x_j - x_i, y_j - y_i)
    dist = np.linalg.norm(delta, axis=1, keepdims=True)  # ||c_i - c_j||_2
    unit_dir = delta / (dist + eps)  # unit direction

    attr = np.hstack([dist, unit_dir])  # (E, 3)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)

    return edge_index, edge_attr
