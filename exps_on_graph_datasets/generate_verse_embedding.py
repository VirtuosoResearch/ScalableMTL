import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--num_communities", type=int, default=100)
args = parser.parse_args()

# %%
from build_community_detections import generate_network, generate_network_v3
from torch_geometric.utils import is_undirected
import torch_geometric.transforms as T

transform = T.ToUndirected()
dataset_name = args.dataset
data = generate_network(net_name=dataset_name, num_cmty=args.num_communities)
data = transform(data)

# %%
from verse.python.wrapper import VERSE

verse = VERSE(cpath="/home/ldy/verse/src")

# %%
import numpy as np
from scipy.sparse import csr_matrix

edge_index = data.edge_index.numpy()
edges = np.ones(edge_index.shape[1])

graph = csr_matrix((edges, (edge_index[0], edge_index[1])))

# %%
w = verse.verse_ppr(graph, n_hidden=128)

# %%
np.save(f"./verse/embeddings/{dataset_name}_top{args.num_communities}_default_ppr_128.npy", w)


# # %%
# import networkx as nx

# edge_list = data.edge_index.numpy()
# edge_list = [(edge_list[0, i], edge_list[1, i]) for i in range(edge_list.shape[1])]
# graph = nx.from_edgelist(edge_list)
# # %%
# from karateclub import Node2Vec

# model = Node2Vec()
# model.fit(graph)
# X = model.get_embedding()
