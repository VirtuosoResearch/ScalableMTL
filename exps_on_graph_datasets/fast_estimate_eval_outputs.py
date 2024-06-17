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
from sklearn.linear_model import LogisticRegression

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

def eval_output(model, train_loader, input_dim, task_idx, device, pretrain_state_dict, finetuned_state_dict):
    model.eval()
    diffs = 0; counts = 0
    for batch in train_loader:
        model.load_state_dict(pretrain_state_dict)
        xs, y, train_mask = batch
        xs, y, train_mask = xs.to(device), y.to(device), train_mask.to(device)
        xs = [x for x in torch.split(xs, input_dim, -1)]
        pretrain_outputs = model(xs, return_softmax=False)
        task_train_mask = train_mask[:, task_idx]
        pretrain_outputs = pretrain_outputs[task_train_mask][:, task_idx]
        
        pretrain_gradients = pretrain_outputs.clone()
        for i in range(len(pretrain_outputs)):
            tmp_gradient = torch.autograd.grad(pretrain_outputs[i], model.parameters(), retain_graph=True, create_graph=False)
            tmp_gradient = dict(
                (name, param) for (name, _), param in zip(model.named_parameters(), tmp_gradient)
            )
            dot_product = 0
            for key, param in model.named_parameters():
                dot_product += (tmp_gradient[key]*(finetuned_state_dict[key]-pretrain_state_dict[key])).sum().cpu().item()
            
            pretrain_gradients[i] = (dot_product)
        pretrain_outputs = pretrain_outputs.detach()

        model.load_state_dict(finetuned_state_dict, strict=False)
        finetuned_outputs = model(xs, return_softmax=False)
        finetuned_outputs = finetuned_outputs[task_train_mask][:, task_idx]

        # print((pretrain_outputs+pretrain_gradients-finetuned_outputs).abs() / (finetuned_outputs).abs())
        diff = (pretrain_outputs+pretrain_gradients-finetuned_outputs).abs() / torch.maximum(finetuned_outputs.abs(), pretrain_outputs.abs())
        counts += task_train_mask.sum().cpu().item()
        diffs += diff.square().sum().cpu().item()
        print(diffs)
    
    print(f"Average relative error for task {task_idx}", diffs/counts)
    return diffs/counts

def compute_norm(model, pretrain_weights, finetuned_weights=None):
    norm = 0
    for key, param in model.named_parameters():
        if finetuned_weights is None:
            norm += (torch.linalg.norm(pretrain_weights[key])).square().cpu().item()
        else:
            norm += (torch.linalg.norm(finetuned_weights[key]-pretrain_weights[key])).square().cpu().item()
    return norm

def generate_state_dict(model, state_dict, coef, device):
    # reshape coef
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        param_len = np.prod(param.shape)
        if "project" in key: 
            new_state_dict[key] = state_dict[key].clone()
            continue
        new_state_dict[key] = state_dict[key].clone() + \
            torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
        cur_len += param_len
    return new_state_dict

'''
Estimate approximation error
'''
def main(args):
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
    
    data.y = data.y[:]
    if len(data.train_mask.shape) == 2:
        data.train_mask = data.train_mask[:]
        data.val_mask  = data.val_mask[:]
        data.test_mask = data.test_mask[:]

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
                     num_classes, args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    elif args.model == "gamlp":
        model = JK_GAMLP(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     input_drop=args.input_drop, att_dropout=args.attn_drop, pre_process=True, residual=True, alpha=args.alpha)
    else:
        raise NotImplementedError("No such model implementation!")
    print(model)
    model = model.to(device)
    input_dim = data.x.shape[1]
    gradient_dim = 0
    for name, param in model.named_parameters():
        if "project" in name:
            continue
        if param.requires_grad:
            gradient_dim += param.numel()

    # load checkpoint 
    pretrain_checkpoint_dir = f"./saved/{args.dataset}_sign_3_256_all_run_{args.run}/model_best.pth"
    state_dict = torch.load(pretrain_checkpoint_dir, map_location=device)
    model.load_state_dict(state_dict)

    # fine-tuned checkpoint
    finetuned_checkpoint_dir = f"./saved_selected/{args.load_model_dir}/model_best.pth"
    finetuned_state_dict = torch.load(finetuned_checkpoint_dir, map_location=device)
    finetuned_weights = []
    for key, param in model.state_dict().items():
        if "project" not in key:
            # norm = torch.linalg.norm(finetuned_state_dict[key]-state_dict[key])
            finetuned_weights.append((finetuned_state_dict[key]-state_dict[key]).flatten())
    coef = torch.cat(finetuned_weights, 0).cpu().numpy()
    print("L2 norm", np.linalg.norm(coef, ord=2))

    # task_idxes = np.random.choice(100, size=5, replace=False)
    print("Selected task idxes", task_idxes)

    for scale in np.arange(10, 0, -1) :    # np.arange(1, 11, 1) 
        print("Perturbation scale: ", scale)
        total_diff = 0
        new_state_dict = generate_state_dict(model, state_dict, coef*scale/np.linalg.norm(coef, ord=2), device) 
        print("Norm:", compute_norm(model, state_dict, new_state_dict))
        for task_idx in task_idxes:
            diff = eval_output(model, train_loader, input_dim, task_idx,  device, state_dict, new_state_dict)
            total_diff += diff
        print("Average relative error", total_diff/len(task_idxes))





    # # collect gradients
    # gradients_dir = f"./gradients/{args.dataset}/run_{args.run}"
    # assert os.path.exists(gradients_dir)
    # matrix_P = np.load(f"./gradients/{args.dataset}/projection_matrix_{args.run}.npy")

    # task_to_gradients = {}
    # task_to_outputs = {}
    # task_to_labels = {}
    # for task_idx in task_idxes:
    #     gradients = np.load(f"{gradients_dir}/task_{task_idx}_gradients.npy")[:, :gradient_dim]
    #     outputs = np.load(f"{gradients_dir}/task_{task_idx}_outputs.npy")
    #     labels = np.load(f"{gradients_dir}/task_{task_idx}_labels.npy")
    #     task_to_gradients[task_idx] = gradients
    #     task_to_outputs[task_idx] = outputs
    #     task_to_labels[task_idx] = labels

    # ''' Train logistic regression '''
    # gradients = np.concatenate([task_to_gradients[task_idx] for task_idx in task_idxes], axis=0)
    # labels = np.concatenate([task_to_labels[task_idx] for task_idx in task_idxes], axis=0)
    # # labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
    # # mask = np.copy(labels)
    # # mask[labels == 0] = -1
    # # mask = mask.reshape(-1, 1)
    # # gradients = gradients*mask

    # train_num = int(len(gradients)*0.8)
    # train_gradients, train_labels = gradients[:train_num], labels[:train_num]
    # test_gradients, test_labels = gradients[train_num:], labels[train_num:]

    # clf = LogisticRegression(random_state=0, penalty='l2', C=100) # solver='liblinear' 
    # clf.fit(train_gradients, train_labels)
    # print(clf.score(test_gradients, test_labels))
    
    # # if projected:  # else: coef = clf.coef_.copy().flatten()
    # proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
    # coef = matrix_P @ proj_coef.flatten()
    # print("L2 norm", np.linalg.norm(coef))


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
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument('--load_model_dir', type=str, default="test")

    parser = add_decoupling_args(parser)
    parser = add_community_detection_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
