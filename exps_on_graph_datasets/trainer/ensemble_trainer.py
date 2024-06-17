import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.pagerank import pagerank_scipy

class EnsembleTrainer:
    '''
    Training logic for ensembled node classification
    '''

    def __init__(self, model, optimizer, datas, split_idx, evaluator, device,
                epochs, log_steps, checkpoint_dir, degrees, degree_thres, monitor="accuracy"):
        self.model = model
        self.optimizer = optimizer
        self.datas = datas
        self.train_idx = split_idx['train']
        self.valid_idx = split_idx['valid']
        self.test_idx = split_idx['test']
        self.evaluator = evaluator
        self.device = device

        ''' Training config '''
        self.epochs = epochs
        self.log_steps = log_steps
        self.checkpoint_dir = checkpoint_dir
        self.degree_thres = degree_thres
        self.degrees = degrees

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.monitor = monitor

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        
        avg_loss = 0
        for data in self.datas:
            data = data.to(self.device)
            if hasattr(data, "adj_t"):
                outputs = self.model(data.x, data.adj_t, edge_weight = data.edge_weight)[self.train_idx]
            else:
                outputs = self.model(data.x, data.edge_index, edge_weight = data.edge_weight)[self.train_idx]

            labels = data.y.squeeze(1)[self.train_idx]

            loss = F.nll_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            avg_loss += loss.item()
        return avg_loss/len(self.datas)

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
        
        print('Training finished: ',
            f'Test: {100 * test_acc:.2f}% ' 
            f'Longtail Test: {100*test_longtail_acc:.2f}%')
        return test_acc, test_longtail_acc

    def save_checkpoint(self, name = "model_best"):
        model_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Saving current model: {name}.pth ...")

    def test(self):
        self.model.eval()

        with torch.no_grad():
            y_preds = torch.zeros(self.datas[0].y.size(0), self.datas[0].y.max().item() + 1).to(self.device)
            for data in self.datas:
                data = data.to(self.device)
                if hasattr(data, "adj_t"):
                    out = self.model(data.x, data.adj_t, edge_weight = data.edge_weight)
                else:
                    out = self.model(data.x, data.edge_index, edge_weight = data.edge_weight)
                
                y_pred = out.argmax(dim=-1)
                y_preds[torch.arange(data.y.size(0)), y_pred] += 1

        y_pred = y_preds.argmax(dim=-1, keepdim=True)
        y_true = self.datas[0].y
        train_acc = self.evaluator.eval({
            'y_true': y_true[self.train_idx],
            'y_pred': y_pred[self.train_idx],
        })['acc']
        valid_acc = self.evaluator.eval({
            'y_true': y_true[self.valid_idx],
            'y_pred': y_pred[self.valid_idx],
        })['acc']
        test_acc = self.evaluator.eval({
            'y_true': y_true[self.test_idx],
            'y_pred': y_pred[self.test_idx],
        })['acc']

        train_loss = F.nll_loss(out[self.train_idx], self.datas[0].y.squeeze(1)[self.train_idx]).cpu().item()
        valid_loss = F.nll_loss(out[self.valid_idx], self.datas[0].y.squeeze(1)[self.valid_idx]).cpu().item()
        test_loss  = F.nll_loss(out[self.test_idx],  self.datas[0].y.squeeze(1)[self.test_idx]).cpu().item()

        ''' Evaluate longtail performance'''
        valid_degrees = self.degrees[self.valid_idx]
        longtail_valid_idx = self.valid_idx[valid_degrees<=self.degree_thres]
        valid_longtail_acc = self.evaluator.eval({
                'y_true': y_true[longtail_valid_idx],
                'y_pred': y_pred[longtail_valid_idx],
            })['acc']

        test_degrees = self.degrees[self.test_idx]
        longtail_test_idx = self.test_idx[test_degrees<=self.degree_thres]
        test_longtail_acc = self.evaluator.eval({
                'y_true': y_true[longtail_test_idx],
                'y_pred': y_pred[longtail_test_idx],
            })['acc']

        return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, test_longtail_acc
