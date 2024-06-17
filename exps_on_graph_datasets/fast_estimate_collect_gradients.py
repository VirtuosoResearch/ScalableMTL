import argparse
import os.path as osp

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.datasets import Yelp, AmazonProducts
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from utils.loader import OriginalGraphDataLoader, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, LeverageScoreEdgeSampler

from models import *
from utils.pagerank import pagerank_scipy
from utils.util import add_result_to_csv, precompute, generate_masked_labels
from minibatch_trainer import Trainer, MultitaskTrainer, MultitaskTrainerPooling
from build_community_detections import generate_network, generate_network_v2, generate_network_v3

name_to_samplers = {
    "no_sampler": OriginalGraphDataLoader,
    "node_sampler": GraphSAINTNodeSampler,
    "edge_sampler": GraphSAINTEdgeSampler,
    "rw_sampler": GraphSAINTRandomWalkSampler,
    "ls_sampler": LeverageScoreEdgeSampler
}

name_to_num_classes = {
    "yelp": 100,
    "amazon": 107,
    "protein": 112,
    "youtube": 100,
    "dblp": 100,
    "amazon": 100,
    "livejournal": 100,
    "arxiv": 40,
}


def get_task_gradients(model, train_loader, input_dim, task_idx, device, projection=None):
    '''
    Compute gradients: for a single task idx
    '''
    model.train()

    gradients = []; model_outputs = []; returned_gradients = []; labels = []
    # For decoupling trainer, the train loader only loads propagated features and labels
    for batch in train_loader:
        xs, y, train_mask = batch
        xs, y, train_mask = xs.to(device), y.to(device), train_mask.to(device)
        xs = [x for x in torch.split(xs, input_dim, -1)]
        
        outputs = model(xs, return_softmax=False)

        assert len(train_mask.shape) == 2
        task_train_mask = train_mask[:, task_idx]
        loss = outputs[task_train_mask][:, task_idx]
        model_outputs.append(loss.detach().cpu().numpy())
        labels.append(y[task_train_mask][:, task_idx].detach().cpu().numpy())

        # F.binary_cross_entropy_with_logits(outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx], reduction="none")
        for i in range(len(loss)):
            tmp_gradients = torch.autograd.grad(loss[i], model.parameters(), retain_graph=True, create_graph=False)
            tmp_gradients = torch.cat([gradient.view(-1) for gradient in tmp_gradients]).cpu().numpy() # flatten gradients
            gradients.append(tmp_gradients)
            
        if projection is not None:
            if len(gradients) > 100:
                gradients = np.array(gradients)
                gradients = np.matmul(gradients, projection)
                returned_gradients.append(gradients)
                gradients = []
    
    gradients = np.array(gradients)
    model_outputs = np.concatenate(model_outputs)
    labels = np.concatenate(labels)
        
    if projection is not None:
        if len(gradients) > 0:
            gradients = np.matmul(gradients, projection)
            returned_gradients.append(gradients)
        gradients = np.concatenate(returned_gradients, axis=0)
        return gradients, model_outputs, labels
    return gradients, model_outputs, labels

'''
Estimate approximation error
'''
def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset == "youtube":
        transform = T.ToUndirected()
        data_dir = f"./data/com_{args.dataset}/"
        if os.path.exists(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt')):
            data = torch.load(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
            print("Load data from file!")
        else:
            data = generate_network(net_name=args.dataset, num_cmty=args.num_communities, 
                    train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
                    feature_dim=args.feature_dim, embedding_dir=f"./verse/embeddings/{args.dataset}_top{args.num_communities}_default_ppr_{args.feature_dim}.npy")
            data = transform(data)
            data.y = data.y.type(torch.float)
            torch.save(data, os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
        name_to_num_classes[args.dataset] = args.num_communities
    elif args.dataset == "dblp" or args.dataset == "livejournal" or args.dataset == "orkut":
        transform = T.ToUndirected()
        data_dir = f"./data/com_{args.dataset}/"
        if os.path.exists(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt')):
            data = torch.load(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
            print("Load data from file!")
        else:
            data = generate_network_v2(net_name=args.dataset, num_cmty=args.num_communities, 
                    train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
                    feature_dim=args.feature_dim, embedding_dir=f"./verse/embeddings/{args.dataset}_top{args.num_communities}_default_ppr_{args.feature_dim}.npy")
            data = transform(data)
            data.y = data.y.type(torch.float)
            torch.save(data, os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
        name_to_num_classes[args.dataset] = args.num_communities
    elif args.dataset == "amazon":
        transform = T.ToUndirected()
        data_dir = f"./data/com_{args.dataset}/"
        if os.path.exists(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt')):
            data = torch.load(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
            print("Load data from file!")
        else:
            data = generate_network_v3(net_name=args.dataset, num_cmty=args.num_communities, 
                    train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
                    feature_dim=args.feature_dim, embedding_dir=f"./verse/embeddings/{args.dataset}_top{args.num_communities}_default_ppr_{args.feature_dim}.npy")
            data = transform(data)
            data.y = data.y.type(torch.float)
            torch.save(data, os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
        name_to_num_classes[args.dataset] = args.num_communities
    else:
        print("Non-valid dataset name!")
        exit()
    num_classes = name_to_num_classes[args.dataset]

    if args.task_idxes == -1:
        task_idxes = np.arange(num_classes)
    else:
        task_idxes = np.array(args.task_idxes)
    
    data.y = data.y[:, task_idxes]
    if len(data.train_mask.shape) == 2:
        data.train_mask = data.train_mask[:, task_idxes]
        data.val_mask  = data.val_mask[:, task_idxes]
        data.test_mask = data.test_mask[:, task_idxes]

    # Initialize mini-batch sampler
    decoupling = args.sample_method=="decoupling"
    if decoupling:
        data = precompute(data, args.num_layers)

        xs_train = torch.cat([x for x in data.xs], -1)
        y_train = data.y

        train_set = torch.utils.data.TensorDataset(xs_train, y_train, data.train_mask) \
            if not (args.model == "dmon" or args.model == "mincut")  else torch.utils.data.TensorDataset(xs_train, y_train, data.train_mask, torch.arange(xs_train.shape[0]))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, num_workers=1
        )
        test_loader = None

    # Initialize the model
    if args.model == "sign":
        model = SIGN_MLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    elif args.model == "gamlp":
        model = JK_GAMLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     input_drop=args.input_drop, att_dropout=args.attn_drop, pre_process=True, residual=True, alpha=args.alpha)
    else:
        raise NotImplementedError("No such model implementation!")
    print(model)
    model = model.to(device)
    input_dim = data.x.shape[1]
    gradient_dim = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_dim += param.numel()

    # load checkpoint 
    checkpoint_dir = f"./saved/{args.dataset}_sign_3_256_{args.num_communities}_all_run_{args.run}/model_best.pth"
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device))

    # collect gradients
    gradients_dir = f"./gradients/{args.dataset}_{args.project_dim}_{args.num_communities}/run_{args.run}"

    if not os.path.exists(gradients_dir):
        os.makedirs(gradients_dir)

    if args.create_projection:
        project_dim = args.project_dim
        matrix_P = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(np.float)
        matrix_P *= 1 / np.sqrt(project_dim)

        np.save(f"./gradients/{args.dataset}_{args.project_dim}_{args.num_communities}/projection_matrix_{args.run}.npy", matrix_P)
    else:
        idx = args.run % 10
        print("Loading projection matrix: ", idx)
        matrix_P = np.load(f"./gradients/{args.dataset}_{args.project_dim}_{args.num_communities}/projection_matrix_{idx}.npy")

    for task_idx in np.arange(10):
        gradients, outputs, labels = get_task_gradients(model, train_loader, input_dim, task_idx, device, projection=matrix_P)

        print("Saving gradients for task: ", task_idx, " shape: ", gradients.shape)
        np.save(f"{gradients_dir}/task_{task_idx}_gradients.npy", gradients)
        np.save(f"{gradients_dir}/task_{task_idx}_outputs.npy", outputs)
        np.save(f"{gradients_dir}/task_{task_idx}_labels.npy", labels)

    end = time.time()
    print("Training completes in {} seconds".format(end-start))

def add_decoupling_args(parser):
    # For SIGN
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)

    # For MOE
    parser.add_argument('--num_of_experts', type=int, default=10)

    # For DMoN
    parser.add_argument('--dmon_lam', type=float, default=1.0)
    return parser

def add_community_detection_args(parser):
    # num_cmty=100, train_ratio=0.2, val_ratio=0.2, feature_dim=64
    parser.add_argument("--num_communities", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--feature_dim", type=int, default=128)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--model', type=str, default='sign')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_edge_index', action="store_true")
    parser.add_argument('--evaluator', type=str, default="f1_score")
    parser.add_argument('--monitor', type=str, default="avg")
    parser.add_argument('--task_idxes', nargs='+', type=int, default=-1)
    parser.add_argument("--save_name", type=str, default="test")

    ''' Sampling '''
    parser.add_argument('--sample_method', type=str, default="decoupling")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--sample_coverage', type=int, default=0)

    ''' Model '''
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_bn', action="store_true")
    
    # GAT
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--input_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.1)

    '''Projection'''
    parser.add_argument("--create_projection", action="store_true")
    parser.add_argument("--project_dim", type=int, default=200)
    parser.add_argument("--run", type=int, default=0)

    parser = add_decoupling_args(parser)
    parser = add_community_detection_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)

