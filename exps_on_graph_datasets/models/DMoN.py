from models.layers.dmon_pool import DMoNPooling
from models.Precomputing_base import PrecomputingBase
from models.SIGN import FeedForwardNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, subgraph
from torch.utils.data import DataLoader
from torch_geometric.nn.dense import dense_mincut_pool


class DMoNGCN(PrecomputingBase):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True, mlp_layers=2, input_drop=0.3):
        super(DMoNGCN, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn)

        in_feats = self.num_feats
        out_feats = self.num_classes
        hidden = self.dim_hidden
        num_hops = self.num_layers + 1
        dropout = self.dropout

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, mlp_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      mlp_layers, dropout)
        
        self.pool = DMoNPooling([num_hops * hidden, ], out_feats)


    def forward(self, feats, edge_index, return_softmax=True):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))

        adj = to_dense_adj(edge_index, max_num_nodes=feats[0].shape[0])
        _, _, _, spectral_loss, ortho_loss, cluster_loss = self.pool(
            torch.cat(hidden, dim=-1).unsqueeze(0), adj
        )
        
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        

        if return_softmax:
            out = F.log_softmax(out, dim=1)
        return out, spectral_loss+ortho_loss+cluster_loss

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()
        self.pool.reset_parameters()

    @torch.no_grad()
    def inference(self, xs_all, device, edge_index, return_softmax=True):
        y_preds = []
        loader = DataLoader(range(xs_all[0].size(0)), batch_size=100000)
        for perm in loader:
            sub_edge_index, _ = subgraph(perm, edge_index, relabel_nodes=True)
            sub_edge_index = sub_edge_index.to(device)
            y_pred, _ = self.forward([x[perm].to(device) for x in xs_all], edge_index=sub_edge_index, return_softmax=return_softmax)
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds
    
class MinCutPoolGCN(PrecomputingBase):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True, mlp_layers=2, input_drop=0.3):
        super(MinCutPoolGCN, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn)

        in_feats = self.num_feats
        out_feats = self.num_classes
        hidden = self.dim_hidden
        num_hops = self.num_layers + 1
        dropout = self.dropout

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, mlp_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      mlp_layers, dropout)
        
        self.pool = dense_mincut_pool


    def forward(self, feats, edge_index, return_softmax=True):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        
        adj = to_dense_adj(edge_index, max_num_nodes=feats[0].shape[0])
        _, _, mincut_loss, ortho_loss = self.pool(
            torch.cat(hidden, dim=-1).unsqueeze(0), adj, F.softmax(out, dim=1)
        )

        if return_softmax:
            out = F.log_softmax(out, dim=1)
        return out, mincut_loss+ortho_loss

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

    @torch.no_grad()
    def inference(self, xs_all, device, edge_index, return_softmax=True):
        y_preds = []
        loader = DataLoader(range(xs_all[0].size(0)), batch_size=100000)
        for perm in loader:
            sub_edge_index, _ = subgraph(perm, edge_index, relabel_nodes=True)
            sub_edge_index = sub_edge_index.to(device)
            y_pred, _ = self.forward([x[perm].to(device) for x in xs_all], edge_index=sub_edge_index, return_softmax=return_softmax)
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds