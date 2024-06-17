# %%
import torch
import numpy as np
from build_community_detections import generate_network
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.pagerank import pagerank_scipy
import torch_geometric.transforms as T
from utils.random_graphs import generate_sbm_graph
import os

class args:
    load = True
    num_tasks = 100
    dataset_name = "livejournal"
if args.load:
    data = torch.load(os.path.join(f"./data/com_{args.dataset_name}/", f'{args.dataset_name}_100_128_data.pt'))
else:
    transform = T.ToUndirected()
    data = generate_network(net_name=args.dataset_name)
    data = transform(data)
    # data = generate_sbm_graph(node_num=100)

# %%
'''Compute overlaps '''
overlaps = np.zeros((args.num_tasks, args.num_tasks))
for i in range(args.num_tasks):
    for j in range(args.num_tasks):
        if j > i:
            overlaps[i][j] = np.logical_and(data.y[:, i] ==1,  data.y[:, j]==1).sum().item()

overlaps_max =  np.max(overlaps, axis=1)
print(overlaps_max.mean(), overlaps_max.max()/data.num_nodes)

# %%
task_nodes = []
for task in [0, 2, 5, 9]:
    tmp_training_nodes = set(np.nonzero((data.y[:, task]==1).numpy())[0])
    task_nodes.append(tmp_training_nodes)

task_1_unique_nodes = list(task_nodes[0].difference(task_nodes[0].intersection(task_nodes[1])))
overlapping_nodes = list(task_nodes[0].intersection(task_nodes[1]))
task_2_unique_nodes = list(task_nodes[1].difference(task_nodes[1].intersection(task_nodes[0])))
task_3_unique_nodes = list(task_nodes[2])
task_4_unique_nodes = list(task_nodes[3])

to_new_node_dict = {}; count = 0
for node in task_1_unique_nodes + overlapping_nodes + task_2_unique_nodes + task_3_unique_nodes + task_4_unique_nodes:
        if node not in to_new_node_dict:
            to_new_node_dict[node] = count
            count +=1
new_nodes = task_1_unique_nodes + overlapping_nodes + task_2_unique_nodes + task_3_unique_nodes  + task_4_unique_nodes
new_node_idxes =  torch.Tensor(np.array(new_nodes)).type(torch.long)# torch.Tensor(np.arange(1000)).type(torch.long)
# new_node_idxes = torch.Tensor(np.arange(100)).type(torch.long)
new_data = data.subgraph(new_node_idxes)

# %%
''' Compute pagerank '''
G = to_scipy_sparse_matrix(new_data.edge_index, num_nodes=new_data.num_nodes)

show_pageranks_list = []
for i, task in enumerate([0, 2, 5, 9]):
    tmp_training_nodes = list(np.nonzero((data.y[:, task]==1).numpy())[0])    
    tmp_nodes = [to_new_node_dict[node] for node in tmp_training_nodes if node in to_new_node_dict]
    tmp_nodes.sort()
    pageranks = []
    for node in tmp_nodes:
        # print(node)
        personalization = {node: 1} # {tmp_node: 1/len(tmp_nodes) for tmp_node in tmp_nodes}
        try:
            pagerank = pagerank_scipy(G, alpha=0.2, personalization=personalization)
            pageranks.append(pagerank)
        except:
            continue
    pageranks = np.array(pageranks)
    show_pageranks = np.copy(pageranks)
    # show_pageranks[show_pageranks<1e-4] = 0
    show_pageranks[show_pageranks>0.1] = 10e-3
    # pageranks[:, np.arange(len(training_nodes))] = pageranks[:,training_nodes]
    show_pageranks_list.append(show_pageranks)

# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

# axs = {0: ax1, 1: ax2, 2: ax3, 3: ax4}
title_texts = [
    r'$\mathrm{Propagation~vectors~of~task~}1$',
    r'$\mathrm{Propagation~vectors~of~task~}2$',
    r'$\mathrm{Propagation~vectors~of~task~}3$',
    r'$\mathrm{Propagation~vectors~of~task~}4$',
]
i = 3
# for i, show_pageranks in enumerate(show_pageranks_list):
show_pageranks = show_pageranks_list[i][:100]
f, ax = plt.subplots(figsize=(10, 3))
ax.imshow(show_pageranks, vmin=1e-4, vmax=1e-2, cmap="YlGn")
ax.set_xticks([])
ax.set_yticks([])
# if i == 0:
ax.set_ylabel(r'$\mathrm{Task~}4$', fontsize = 40)
# ax.set_xlabel(r'$\mathrm{Nodes~in~the~graph}$', fontsize = 24)
# ax.title.set_text(title_texts[i])
# ax.title.set_fontsize(24)
f.tight_layout()
plt.savefig(f"./notebooks/figures/task_propagation_pagerank_{i}.pdf", format="pdf", dpi=1200)
plt.show()