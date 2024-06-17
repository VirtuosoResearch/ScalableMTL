import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from utils.metrics import evaluate_accuracy, evaluate_f1, evaluate_roc_auc
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.pagerank import pagerank_scipy

class Trainer:
    '''
    Training logic for semi-supervised node classification
    '''
    def __init__(self, model, optimizer, data, split_idx, evaluator, device,
                epochs, log_steps, checkpoint_dir, degrees, degree_thres, monitor="accuracy"):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.train_idx = split_idx['train']
        self.valid_idx = split_idx['valid']
        self.test_idx = split_idx['test']
        self.evaluator = evaluator
        self.device = device

        ''' Training config '''
        self.epochs = epochs
        self.log_steps = log_steps
        self.checkpoint_dir = checkpoint_dir
        self.degrees = degrees
        self.degree_thres = degree_thres

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.monitor = monitor

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        data = self.data.to(self.device)

        if hasattr(self.data, "adj_t"):
            outputs = self.model(data.x, data.adj_t, edge_weight = data.edge_weight)[self.train_idx]
        else:
            outputs = self.model(data.x, data.edge_index, edge_weight = data.edge_weight)[self.train_idx]

        labels = data.y.squeeze(1)[self.train_idx]

        loss = F.nll_loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        best_val_acc = test_acc = test_longtail_acc = 0

        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch(epoch)
            train_acc, valid_acc, tmp_test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, tmp_test_longtail_acc = self.test()

            monitor_metric = valid_acc if self.monitor == 'accuracy' else valid_longtail_acc
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                test_acc = tmp_test_acc
                test_longtail_acc = tmp_test_longtail_acc

                ''' Save checkpoint '''
                self.save_checkpoint()

            if epoch % self.log_steps == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, train loss: {train_loss:.4f}, '
                      f'Valid: {100 * valid_acc:.2f}%, valid loss: {valid_loss:.4f}, '
                      f'Test: {100 * tmp_test_acc:.2f}%, test_loss: {test_loss:.4f}, ' 
                      f'Valid longtail acc: {100*valid_longtail_acc:.2f}% '
                      f'Test longtail acc: {100*tmp_test_longtail_acc:.2f}%')
        return test_acc, test_longtail_acc

    def save_checkpoint(self, name = "model_best"):
        model_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saving current model: {name}.pth ...")

    def test(self):
        self.model.eval()
        
        data = self.data.to(self.device)
        if hasattr(self.data, "adj_t"):
            out = self.model(data.x, data.adj_t, edge_weight = data.edge_weight)
        else:
            out = self.model(data.x, data.edge_index, edge_weight = data.edge_weight)

        y_pred = out.argmax(dim=-1, keepdim=True)
        y_true = data.y
        train_acc = evaluate_accuracy(
            preds = y_pred[self.train_idx].detach().cpu().numpy(),
            labels = y_true[self.train_idx].detach().cpu().numpy(),
        )
        valid_acc = evaluate_accuracy(
            preds = y_pred[self.valid_idx].detach().cpu().numpy(),
            labels =  y_true[self.valid_idx].detach().cpu().numpy(),
        )
        test_acc = evaluate_accuracy(
            preds = y_pred[self.test_idx].detach().cpu().numpy(),
            labels =  y_true[self.test_idx].detach().cpu().numpy(),
        )

        train_loss = F.nll_loss(out[self.train_idx], data.y.squeeze(1)[self.train_idx]).cpu().item()
        valid_loss = F.nll_loss(out[self.valid_idx], data.y.squeeze(1)[self.valid_idx]).cpu().item()
        test_loss  = F.nll_loss(out[self.test_idx], data.y.squeeze(1)[self.test_idx]).cpu().item()

        ''' Evaluate longtail performance'''
        valid_degrees = self.degrees[self.valid_idx]
        longtail_valid_idx = self.valid_idx[valid_degrees<=self.degree_thres]
        valid_longtail_acc = evaluate_accuracy(
            preds = y_pred[longtail_valid_idx].detach().cpu().numpy(),
            labels = y_true[longtail_valid_idx].detach().cpu().numpy(),
        )

        test_degrees = self.degrees[self.test_idx]
        longtail_test_idx = self.test_idx[test_degrees<=self.degree_thres]
        test_longtail_acc = evaluate_accuracy(
                preds = y_pred[longtail_test_idx].detach().cpu().numpy(),
                labels = y_true[longtail_test_idx].detach().cpu().numpy(),
            )

        return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, test_longtail_acc

class MultilabelTrainer:
    '''
    Training logic for semi-supervised node classification
    1. change evaluation
    2. change model
    '''
    def __init__(self, model, optimizer, data, split_idx, evaluator, device,
                epochs, log_steps, checkpoint_dir, degrees, degree_thres, monitor="accuracy"):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.train_idx = split_idx['train']
        self.valid_idx = split_idx['valid']
        self.test_idx = split_idx['test']
        self.evaluator = evaluator
        self.device = device

        ''' Training config '''
        self.epochs = epochs
        self.log_steps = log_steps
        self.checkpoint_dir = checkpoint_dir
        self.degrees = degrees
        self.degree_thres = degree_thres

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.monitor = monitor

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        data = self.data.to(self.device)

        if hasattr(self.data, "adj_t"):
            outputs = self.model(data.x, data.adj_t, edge_weight = data.edge_weight, return_softmax=False)[self.train_idx]
        else:
            outputs = self.model(data.x, data.edge_index, edge_weight = data.edge_weight, return_softmax=False)[self.train_idx]

        labels = data.y[self.train_idx]

        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        best_val_acc = test_acc = test_longtail_acc = 0

        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch(epoch)
            train_acc, valid_acc, tmp_test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, tmp_test_longtail_acc = self.test()

            monitor_metric = valid_acc if self.monitor == 'accuracy' else valid_longtail_acc
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                test_acc = tmp_test_acc
                test_longtail_acc = tmp_test_longtail_acc

                ''' Save checkpoint '''
                self.save_checkpoint()

            if epoch % self.log_steps == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, train loss: {train_loss:.4f}, '
                      f'Valid: {100 * valid_acc:.2f}%, valid loss: {valid_loss:.4f}, '
                      f'Test: {100 * tmp_test_acc:.2f}%, test_loss: {test_loss:.4f}, ' 
                      f'Valid longtail acc: {100*valid_longtail_acc:.2f}% '
                      f'Test longtail acc: {100*tmp_test_longtail_acc:.2f}%')
        return test_acc, test_longtail_acc

    def save_checkpoint(self, name = "model_best"):
        model_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saving current model: {name}.pth ...")

    def test(self):
        self.model.eval()
        
        data = self.data.to(self.device)
        if hasattr(self.data, "adj_t"):
            out = self.model(data.x, data.adj_t, edge_weight = data.edge_weight, return_softmax=False)
        else:
            out = self.model(data.x, data.edge_index, edge_weight = data.edge_weight, return_softmax=False)

        y_pred = out
        y_true = data.y
        train_acc = evaluate_roc_auc(
            outputs = y_pred[self.train_idx].detach().cpu().numpy(),
            labels = y_true[self.train_idx].detach().cpu().numpy(),
        )
        valid_acc = evaluate_roc_auc(
            outputs = y_pred[self.valid_idx].detach().cpu().numpy(),
            labels = y_true[self.valid_idx].detach().cpu().numpy(),
        )
        test_acc = evaluate_roc_auc(
            outputs = y_pred[self.test_idx].detach().cpu().numpy(),
            labels = y_true[self.test_idx].detach().cpu().numpy(),
        )

        train_loss = F.binary_cross_entropy_with_logits(out[self.train_idx], data.y[self.train_idx]).cpu().item()
        valid_loss = F.binary_cross_entropy_with_logits(out[self.valid_idx], data.y[self.valid_idx]).cpu().item()
        test_loss  = F.binary_cross_entropy_with_logits(out[self.test_idx],  data.y[self.test_idx]).cpu().item()

        ''' Evaluate longtail performance'''
        valid_degrees = self.degrees[self.valid_idx]
        longtail_valid_idx = self.valid_idx[valid_degrees<=self.degree_thres]
        valid_longtail_acc = evaluate_roc_auc(
            outputs = y_pred[longtail_valid_idx].detach().cpu().numpy(),
            labels = y_true[longtail_valid_idx].detach().cpu().numpy(),
        )

        test_degrees = self.degrees[self.test_idx]
        longtail_test_idx = self.test_idx[test_degrees<=self.degree_thres]
        test_longtail_acc = evaluate_roc_auc(
            outputs = y_pred[longtail_test_idx].detach().cpu().numpy(),
            labels = y_true[longtail_test_idx].detach().cpu().numpy(),
        )

        return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, test_longtail_acc

''' Deprecated '''

class WeightedTrainer(Trainer):
    
    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor,
        group_num, group_weights, group_by="degree"):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

        assert group_weights.shape[0] == group_num
        self.group_num = group_num
        self.group_weights = torch.Tensor(group_weights).to(device)
        self.group_labels = self.split_groups(group_by=group_by)
        print(group_weights)

    def split_groups(self, group_by="degree"):
        '''
        Assign group labels to nodes by:
            degrees
            pagerank values
            dispersion terms
        '''
        group_labels = torch.zeros_like(self.data.y)

        if group_by == "degree":
            if hasattr(self.data, "adj_t"):
                degrees = self.data.adj_t.sum(0)
            else:
                degrees = torch_geometric.utils.degree(self.data.edge_index[1], self.data.x.size(0), dtype=self.data.x.dtype)

            metrics = degrees
        elif group_by == "pagerank":
            ''' Compute pagerank '''
            G = to_scipy_sparse_matrix(self.data.edge_index, num_nodes=self.data.num_nodes)
            pageranks = pagerank_scipy(G, alpha=0.15)
            pageranks = torch.Tensor(pageranks).to(self.device)
            metrics = pageranks

        def split_by_median(idxes):
            tmp_median = torch.median(metrics[idxes])
            group_1 = idxes[metrics[idxes]<=tmp_median]
            group_2 = idxes[metrics[idxes]>tmp_median]
            return group_1, group_2

        group_idxes = [torch.arange(metrics.size(0))]
        for _ in range(self.group_num-1):
            tmp_idxes = group_idxes[0]
            group_1, group_2 = split_by_median(tmp_idxes)
            group_idxes.pop(0)
            group_idxes.append(group_1); group_idxes.append(group_2)
        for i, idxes in enumerate(group_idxes):
            group_labels[idxes] = i
        return group_labels.squeeze()

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        group_labels = self.group_labels[self.train_idx]

        if hasattr(self.data, "adj_t"):
            outputs = self.model(self.data.x, self.data.adj_t)[self.train_idx]
        else:
            outputs = self.model(self.data.x, self.data.edge_index)[self.train_idx]

        labels = self.data.y.squeeze(1)[self.train_idx]

        losses = F.nll_loss(outputs, labels, reduction="none")
        group_losses  = scatter_mean(losses, index=group_labels, dim_size=self.group_num, dim=0)
        loss = torch.sum(group_losses*self.group_weights)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class LabelSmoothTrainer(Trainer):

    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, alpha, num_classes,
        monitor="accuracy"):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

        self.alpha = alpha
        self.smoothed_label = torch.ones(1, num_classes, dtype=torch.float).to(device) 
        self.smoothed_label = self.smoothed_label / num_classes

    def _label_smooth_loss(self, output, target):
        ce_loss = F.nll_loss(output, target)
        smooth_loss = torch.sum(- output * self.smoothed_label, dim=1).mean()
        loss = ce_loss * (1-self.alpha) + smooth_loss * self.alpha
        return loss

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        if hasattr(self.data, "adj_t"):
            outputs = self.model(self.data.x, self.data.adj_t)[self.train_idx]
        else:
            outputs = self.model(self.data.x, self.data.edge_index)[self.train_idx]

        labels = self.data.y.squeeze(1)[self.train_idx]

        loss = self._label_smooth_loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()