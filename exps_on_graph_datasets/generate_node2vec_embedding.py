import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()
# %%

from build_community_detections import generate_network
from torch_geometric.utils import is_undirected
import torch_geometric.transforms as T

transform = T.ToUndirected()
dataset_name = args.dataset
data = generate_network(net_name=dataset_name)
data = transform(data)

# %%
import networkx as nx

edge_list = data.edge_index.numpy()
edge_list = [(edge_list[0, i], edge_list[1, i]) for i in range(edge_list.shape[1])]
graph = nx.from_edgelist(edge_list)
# %%
from karateclub import Node2Vec

model = Node2Vec(workers=10)
model.fit(graph)
X = model.get_embedding()

# %%
import numpy as np
np.save(f"./verse/embeddings/{dataset_name}_node2vec_128.npy", X)