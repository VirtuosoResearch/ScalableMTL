import torch
import numpy as np
import pandas as pd
import networkx as nx
import os
from scipy import sparse
from torch_geometric.data import Data

num_of_nodes = {
    'youtube': 1134890
}

data_dirs = {
    'youtube': './data/com_youtube/com-youtube.ungraph.txt',
    'dblp': './data/com_dblp/com-dblp.ungraph.txt',
    'amazon': './data/com_amazon/com-amazon.ungraph.txt',
    'livejournal': './data/com_livejournal/com-lj.ungraph.txt',
    'orkut': './data/com_orkut/com-orkut.ungraph.txt',
}

community_dirs = {
    'youtube': './data/com_youtube/com-youtube.top5000.cmty.txt',
    'dblp': './data/com_dblp/com-dblp.top5000.cmty.txt',
    'amazon': './data/com_amazon/com-amazon.top5000.cmty.txt',
    'livejournal': './data/com_livejournal/com-lj.top5000.cmty.txt',
    'orkut': './data/com_orkut/com-orkut.top5000.cmty.txt'
}

"""
Generate features
"""

def generate_network(net_name = 'youtube', num_cmty=100, train_ratio=0.2, val_ratio=0.2, feature_dim=128, embedding_dir = None):
    rng = np.random.default_rng(1024)
    # load graph structure
    edge_row = []
    edge_col = []
    with open(data_dirs[net_name], 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            src, dst = line.split()
            edge_row.append(int(src)) 
            edge_col.append(int(dst))

    edge_row = np.array(edge_row)
    edge_index = np.array(edge_col)
    edge_index = np.stack([edge_row, edge_col])
    edge_index = torch.Tensor(edge_index).type(torch.long)
    data = Data(edge_index=edge_index)

    # load communities as labels
    labels = []; nodes_in_cmty = []
    with open(community_dirs[net_name], 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = [int(idx) for idx in line]
            labels.append(line)
            nodes_in_cmty.append(len(line))
    labels = np.array(labels)
    nodes_in_cmty = np.array(nodes_in_cmty)

    # extract the largest communities
    cmty_idxes = np.argsort(nodes_in_cmty)[-num_cmty:]
    labels = labels[cmty_idxes]
    nodes_in_cmty = nodes_in_cmty[cmty_idxes]

    # extract subgraph induced by the top community nodes
    new_node_idxes = set()
    for nodes in labels:
        new_node_idxes = new_node_idxes.union(set(nodes))
    new_node_idxes = np.sort(list(new_node_idxes))

    new_num_nodes = len(new_node_idxes)
    to_new_node_idx = np.zeros(data.num_nodes, dtype=np.long)
    to_new_node_idx[new_node_idxes] = np.arange(new_num_nodes)
    labels = [to_new_node_idx[line] for line in labels]

    community_labels = []
    for line in labels:
        tmp_labels = np.zeros(new_num_nodes)
        tmp_labels[line] = 1
        community_labels.append(tmp_labels)
    community_labels = np.array(community_labels).transpose()

    new_node_idxes = torch.Tensor(new_node_idxes).type(torch.long)
    new_data = data.subgraph(new_node_idxes)
    new_data.y = torch.Tensor(community_labels).type(torch.long)

    # generate training masks
    train_masks = []; val_masks = []; test_masks = []
    for tmp_labels in labels:
        num_seeds_train = int(len(tmp_labels)*train_ratio)
        num_seeds_val = int(len(tmp_labels)*val_ratio)
        tmp_labels = rng.permutation(tmp_labels)

        train_seed_set = tmp_labels[:num_seeds_train]
        val_seed_set   = tmp_labels[num_seeds_train:num_seeds_train+num_seeds_val]
        test_seed_set  = tmp_labels[num_seeds_train+num_seeds_val:]

        # sample negative nodes (not in the community)
        negative_samples = [i for i in range(new_num_nodes) if i not in tmp_labels]
        num_neg_train = int(len(negative_samples)*train_ratio)
        num_neg_val = int(len(negative_samples)*val_ratio)
        negative_samples = rng.permutation(negative_samples)
        train_neg_set = negative_samples[:num_neg_train]
        val_neg_set  = negative_samples[num_neg_train:num_neg_train+num_neg_val]
        test_neg_set = negative_samples[num_neg_train+num_neg_val:]
        
        train_mask, val_mask, test_mask = np.zeros(new_num_nodes), np.zeros(new_num_nodes), np.zeros(new_num_nodes)
        train_mask[train_seed_set] = 1; train_mask[train_neg_set] = 1
        val_mask[val_seed_set] = 1; val_mask[val_neg_set] = 1
        test_mask[test_seed_set] = 1; test_mask[test_neg_set] = 1

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    train_masks = np.array(train_masks).transpose()
    val_masks = np.array(val_masks).transpose()
    test_masks = np.array(test_masks).transpose()

    new_data.train_mask = torch.Tensor(train_masks).type(torch.bool)
    new_data.val_mask = torch.Tensor(val_masks).type(torch.bool)
    new_data.test_mask = torch.Tensor(test_masks).type(torch.bool)

    # generate node features
    if (embedding_dir is None) or (not os.path.exists(embedding_dir)):
        node_features = rng.normal(size=(new_data.num_nodes, feature_dim))
        new_data.x = torch.Tensor(node_features)
    else:
        node_embeddings = np.load(embedding_dir)
        new_data.x = torch.Tensor(node_embeddings)
    return new_data


def generate_network_v2(net_name = 'youtube', num_cmty=100, train_ratio=0.2, val_ratio=0.2, feature_dim=128, embedding_dir = None):
    rng = np.random.default_rng(1024)
    # load graph structure
    edge_row = []
    edge_col = []
    with open(data_dirs[net_name], 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            src, dst = line.split()
            edge_row.append(int(src)) 
            edge_col.append(int(dst))

    edge_row = np.array(edge_row)
    edge_index = np.array(edge_col)
    edge_index = np.stack([edge_row, edge_col])
    edge_index = torch.Tensor(edge_index).type(torch.long)
    data = Data(edge_index=edge_index)

    # load communities as labels
    labels = []; nodes_in_cmty = []
    with open(community_dirs[net_name], 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = [int(idx) for idx in line]
            labels.append(line)
            nodes_in_cmty.append(len(line))
    labels = np.array(labels)
    nodes_in_cmty = np.array(nodes_in_cmty)

    # extract the largest communities
    cmty_idxes = np.argsort(nodes_in_cmty)[-num_cmty:]
    labels = labels[cmty_idxes]
    nodes_in_cmty = nodes_in_cmty[cmty_idxes]

    # extract subgraph induced by the top community nodes
    new_node_idxes = set()
    for nodes in labels:
        new_node_idxes = new_node_idxes.union(set(nodes))
    new_node_idxes = np.sort(list(new_node_idxes))

    new_num_nodes = len(new_node_idxes)
    to_new_node_idx = np.zeros(data.num_nodes, dtype=np.long)
    to_new_node_idx[new_node_idxes] = np.arange(new_num_nodes)
    labels = [to_new_node_idx[line] for line in labels]

    community_labels = []
    for line in labels:
        tmp_labels = np.zeros(new_num_nodes)
        tmp_labels[line] = 1
        community_labels.append(tmp_labels)
    community_labels = np.array(community_labels).transpose()

    new_node_idxes = torch.Tensor(new_node_idxes).type(torch.long)
    new_data = data.subgraph(new_node_idxes)
    new_data.y = torch.Tensor(community_labels).type(torch.long)

    # generate training masks
    train_masks = []; val_masks = []; test_masks = []
    for tmp_labels in labels:
        tmp_train_ratio = train_ratio
        if len(tmp_labels) <= 100:
            pass
        else:
            tmp_train_ratio = train_ratio*50/len(tmp_labels)
        # elif len(tmp_labels) < 500:
        #     tmp_train_ratio = train_ratio/2
        # elif len(tmp_labels) < 1000:
        #     tmp_train_ratio = train_ratio/10
        # else:
        #     tmp_train_ratio = train_ratio/20
        num_seeds_train = int(len(tmp_labels)*tmp_train_ratio)
        num_seeds_val = int(len(tmp_labels)*val_ratio)
        tmp_labels = rng.permutation(tmp_labels)

        train_seed_set = tmp_labels[:num_seeds_train]
        val_seed_set   = tmp_labels[num_seeds_train:num_seeds_train+num_seeds_val]
        test_seed_set  = tmp_labels[num_seeds_train+num_seeds_val:]

        # sample negative nodes (not in the community)
        negative_samples = [i for i in range(new_num_nodes) if i not in tmp_labels]
        num_neg_train = int(len(negative_samples)*train_ratio)
        num_neg_val = int(len(negative_samples)*val_ratio)
        negative_samples = rng.permutation(negative_samples)
        train_neg_set = negative_samples[:num_neg_train]
        val_neg_set  = negative_samples[num_neg_train:num_neg_train+num_neg_val]
        test_neg_set = negative_samples[num_neg_train+num_neg_val:]
        
        train_mask, val_mask, test_mask = np.zeros(new_num_nodes), np.zeros(new_num_nodes), np.zeros(new_num_nodes)
        train_mask[train_seed_set] = 1; train_mask[train_neg_set] = 1
        val_mask[val_seed_set] = 1; val_mask[val_neg_set] = 1
        test_mask[test_seed_set] = 1; test_mask[test_neg_set] = 1

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    train_masks = np.array(train_masks).transpose()
    val_masks = np.array(val_masks).transpose()
    test_masks = np.array(test_masks).transpose()

    new_data.train_mask = torch.Tensor(train_masks).type(torch.bool)
    new_data.val_mask = torch.Tensor(val_masks).type(torch.bool)
    new_data.test_mask = torch.Tensor(test_masks).type(torch.bool)

    # generate node features
    if (embedding_dir is None) or (not os.path.exists(embedding_dir)):
        node_features = rng.normal(size=(new_data.num_nodes, feature_dim))
        new_data.x = torch.Tensor(node_features)
    else:
        node_embeddings = np.load(embedding_dir)
        new_data.x = torch.Tensor(node_embeddings)
    return new_data

def generate_network_v3(net_name = 'youtube', num_cmty=100, train_ratio=0.2, val_ratio=0.2, feature_dim=128, embedding_dir = None):
    rng = np.random.default_rng(1024)
    # load graph structure
    edge_row = []
    edge_col = []
    with open(data_dirs[net_name], 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            src, dst = line.split()
            edge_row.append(int(src)) 
            edge_col.append(int(dst))

    edge_row = np.array(edge_row)
    edge_index = np.array(edge_col)
    edge_index = np.stack([edge_row, edge_col])
    edge_index = torch.Tensor(edge_index).type(torch.long)
    data = Data(edge_index=edge_index)

    # load communities as labels
    labels = []; nodes_in_cmty = []
    with open(community_dirs[net_name], 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = [int(idx) for idx in line]
            labels.append(line)
            nodes_in_cmty.append(len(line))
    labels = np.array(labels)
    nodes_in_cmty = np.array(nodes_in_cmty)

    # extract the largest communities
    cmty_idxes = np.argsort(nodes_in_cmty)[-num_cmty:]
    labels = labels[cmty_idxes]
    nodes_in_cmty = nodes_in_cmty[cmty_idxes]

    # extract subgraph induced by the top community nodes
    new_node_idxes = set()
    for nodes in labels:
        new_node_idxes = new_node_idxes.union(set(nodes))
    new_node_idxes = np.sort(list(new_node_idxes))

    new_num_nodes = len(new_node_idxes)
    to_new_node_idx = np.zeros(data.num_nodes, dtype=np.long)
    to_new_node_idx[new_node_idxes] = np.arange(new_num_nodes)
    labels = [to_new_node_idx[line] for line in labels]

    community_labels = []
    for line in labels:
        tmp_labels = np.zeros(new_num_nodes)
        tmp_labels[line] = 1
        community_labels.append(tmp_labels)
    community_labels = np.array(community_labels).transpose()

    new_node_idxes = torch.Tensor(new_node_idxes).type(torch.long)
    new_data = data.subgraph(new_node_idxes)
    new_data.y = torch.Tensor(community_labels).type(torch.long)

    # generate training masks
    train_masks = []; val_masks = []; test_masks = []
    for tmp_labels in labels:
        tmp_train_ratio = train_ratio
        if len(tmp_labels) <= 100:
            tmp_train_ratio = 0.02
        else:
            tmp_train_ratio = train_ratio*10/len(tmp_labels)
        
        num_seeds_train = int(len(tmp_labels)*tmp_train_ratio)
        num_seeds_val = int(len(tmp_labels)*val_ratio)
        tmp_labels = rng.permutation(tmp_labels)

        train_seed_set = tmp_labels[:num_seeds_train]
        val_seed_set   = tmp_labels[num_seeds_train:num_seeds_train+num_seeds_val]
        test_seed_set  = tmp_labels[num_seeds_train+num_seeds_val:]

        # sample negative nodes (not in the community)
        negative_samples = [i for i in range(new_num_nodes) if i not in tmp_labels]
        num_neg_train = int(len(negative_samples)*train_ratio)
        num_neg_val = int(len(negative_samples)*val_ratio)
        negative_samples = rng.permutation(negative_samples)
        train_neg_set = negative_samples[:num_neg_train]
        val_neg_set  = negative_samples[num_neg_train:num_neg_train+num_neg_val]
        test_neg_set = negative_samples[num_neg_train+num_neg_val:]
        
        train_mask, val_mask, test_mask = np.zeros(new_num_nodes), np.zeros(new_num_nodes), np.zeros(new_num_nodes)
        train_mask[train_seed_set] = 1; train_mask[train_neg_set] = 1
        val_mask[val_seed_set] = 1; val_mask[val_neg_set] = 1
        test_mask[test_seed_set] = 1; test_mask[test_neg_set] = 1

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    train_masks = np.array(train_masks).transpose()
    val_masks = np.array(val_masks).transpose()
    test_masks = np.array(test_masks).transpose()

    new_data.train_mask = torch.Tensor(train_masks).type(torch.bool)
    new_data.val_mask = torch.Tensor(val_masks).type(torch.bool)
    new_data.test_mask = torch.Tensor(test_masks).type(torch.bool)

    # generate node features
    if (embedding_dir is None) or (not os.path.exists(embedding_dir)):
        node_features = rng.normal(size=(new_data.num_nodes, feature_dim))
        new_data.x = torch.Tensor(node_features)
    else:
        node_embeddings = np.load(embedding_dir)
        new_data.x = torch.Tensor(node_embeddings)
    return new_data