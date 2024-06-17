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

def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset == "yelp":
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'yelp')
        dataset = Yelp(path, transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=False)]))
        data = dataset[0]

        # Normalize Features
        features = data.x
        features = (features-features.mean(dim=0))/features.std(dim=0)
        data.x = features
    elif args.dataset == "amazon_products":
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'amazon_products')
        dataset = AmazonProducts(path, transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(remove_edge_index=False)]))
        data = dataset[0]

        # Normalize Features
        features = data.x
        features = (features-features.mean(dim=0))/features.std(dim=0)
        data.x = features
        data.y = data.y.type(torch.float)
    elif args.dataset == "arxiv":
        # arxiv
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToUndirected())
        data = dataset[0]

        split_idx = dataset.get_idx_split()

        # Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            if key == "valid": key = "val"
            data[f'{key}_mask'] = mask

        # Convert label to one-hot vectors
        labels = torch.zeros((data.num_nodes, dataset.num_classes), dtype=torch.float)
        labels[torch.arange(data.num_nodes), data.y[:, 0]] = 1
        data.y = labels
    elif args.dataset == "protein":
        dataset = PygNodePropPredDataset(name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr', remove_edge_index=False))
        data = dataset[0]

        # Move edge features to node features.
        data.y = data.y.type(torch.float)
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)

        split_idx = dataset.get_idx_split()

        # Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            if key == "valid": key = "val"
            data[f'{key}_mask'] = mask
    elif args.dataset == "youtube":
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
        print(data.x.shape, data.edge_index.shape)
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
    start = time.time()

    if args.task_idxes == -1:
        task_idxes = np.arange(num_classes)
    else:
        task_idxes = np.array(args.task_idxes)
    
    data.y = data.y[:, task_idxes]
    if len(data.train_mask.shape) == 2:
        data.train_mask = data.train_mask[:, task_idxes]
        data.val_mask  = data.val_mask[:, task_idxes]
        data.test_mask = data.test_mask[:, task_idxes]

    ''' Downsample training set'''
    if args.downsample < 1.0:
        if len(data.train_mask.shape) == 2:
            for idx in range(data.train_mask.shape[1]):
                masked_labels = generate_masked_labels(data.train_mask[:, idx], args.downsample)
                data.train_mask[:, idx][masked_labels] = False
            print("Training size: {}".format(data.train_mask[:, 0].sum().item()))
        else:
            masked_labels = generate_masked_labels(data.train_mask, args.downsample)
            data.train_mask[masked_labels] = False
            print("Training size: {}".format(data.train_mask.sum().item()))
        args.batch_size = int(args.batch_size/args.downsample)

    degrees = None; degree_thres = 0
    
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
    else:
        Z_dir = f"./save_z/{args.dataset}_z.npy"
        train_loader = name_to_samplers[args.sample_method](data, batch_size=args.batch_size,
                                    num_steps=args.num_steps, sample_coverage=args.sample_coverage,
                                    walk_length=args.walk_length, Z_dir=Z_dir)

        test_loader = NeighborSampler(data.clone().edge_index, sizes=[-1],
                                        batch_size=args.test_batch_size, shuffle=False,
                                        num_workers=1)

    # Initialize the model
    if args.model == "mlp":
        model = MLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout)
    elif args.model == "sage":
        model = SAGE(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn)
    elif args.model == "gin":
        model = GIN(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn)
    elif args.model == "sign":
        model = SIGN_MLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    elif args.model == "gamlp":
        model = JK_GAMLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     input_drop=args.input_drop, att_dropout=args.attn_drop, pre_process=True, residual=True, alpha=args.alpha)
    elif args.model == "moe":
        model = MixtureOfExperts(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn,
                     num_of_experts = args.num_of_experts)
    elif args.model == "dmon":
        model = DMoNGCN(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    elif args.model == "mincut":
        model = MinCutPoolGCN(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    else:
        raise NotImplementedError("No such model implementation!")
    print(model)
    model = model.to(device)

    log_metrics = {}
    for run in range(args.runs):
        # reintialize model and optimizer
        model.reset_parameters()

        load_model_dir = f"./saved/{args.load_model_dir}/model_best.pth"
        if os.path.exists(load_model_dir):
            print(f"Load model from {load_model_dir}")
            state_dict = torch.load(load_model_dir, map_location=device)
            state_dict.pop("project.layers.1.weight")
            state_dict.pop("project.layers.1.bias")
            model.load_state_dict(state_dict, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        task_idxes_str = "all" if args.task_idxes == -1 else "_".join([str(idx) for idx in args.task_idxes])
        if len(task_idxes_str) > 100:
            task_idxes_str = task_idxes_str[:100]
        
        save_dir = "saved" if args.task_idxes == -1 else f"saved_selected"
        
        if args.model == "dmon" or args.model == "mincut":
            trainer = MultitaskTrainerPooling(model, optimizer, data, train_loader, test_loader, device,
                            epochs=args.epochs, log_steps=args.log_steps, degrees=degrees, degree_thres=degree_thres, 
                            criterion = "multilabel", evaluator = args.evaluator, monitor=args.monitor, decoupling=decoupling,
                            checkpoint_dir=f"./{save_dir}/{args.dataset}_{args.model}_{args.num_layers}_{args.hidden_channels}_{args.num_communities}_{task_idxes_str}",
                            task_idxes=task_idxes, pool_loss_lam=args.dmon_lam)
        else:
            trainer = MultitaskTrainer(model, optimizer, data, train_loader, test_loader, device,
                            epochs=args.epochs, log_steps=args.log_steps, degrees=degrees, degree_thres=degree_thres, 
                            criterion = "multilabel", evaluator = args.evaluator, monitor=args.monitor, decoupling=decoupling,
                            checkpoint_dir=f"./{save_dir}/{args.dataset}_{args.model}_{args.num_layers}_{args.hidden_channels}_{args.num_communities}_{task_idxes_str}_run_{run}",
                            task_idxes=task_idxes)
        _, _ = trainer.train()
        trainer.load_checkpoint()
        
        if len(data.train_mask.shape) == 2:
            log = trainer.test_in_task_mask()
        else:
            log = trainer.test()
        
        for key, val in log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]
    print("Test accuracy: {:.4f}±{:.4f}".format(
            np.mean(log_metrics[f"test_{args.evaluator}"]), 
            np.std(log_metrics[f"test_{args.evaluator}"])
        ))
    print("Test accuracy for degree <={:2.0f}: {:.4f}±{:.4f}".format(
            degree_thres, 
            np.mean(log_metrics[f"test_longtail_{args.evaluator}"]), 
            np.std(log_metrics[f"test_longtail_{args.evaluator}"])
        ))
    
    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    for task_idx in task_idxes:
        # save validation results
        result_datapoint = {
            "Task": task_idx, 
            "Trained on": task_idxes,
        }
        for key, vals in log_metrics.items():
            if f"task_{task_idx}" in key:
                metric_name = "_".join(key.split("_")[2:])
                result_datapoint[metric_name] = np.mean(vals)
                result_datapoint[metric_name+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_{}.csv".format(args.save_name, args.dataset))
        add_result_to_csv(result_datapoint, file_name)
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
    parser.add_argument('--runs', type=int, default=5)
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
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_bn', action="store_true")
    
    # GAT
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--input_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.1)

    ''' Training '''
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--downsample', type=float, default=1.0)


    ''' Load model'''
    parser.add_argument('--load_model_dir', type=str, default="test")

    parser = add_decoupling_args(parser)
    parser = add_community_detection_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
