from minibatch_trainer.multitask_trainer import MultitaskTrainer, name_to_evaluators, names_to_criterions

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

class TAGTrainer(MultitaskTrainer):
    
    def __init__(self, model, optimizer, data, train_loader, test_loader, device, epochs, log_steps,
        checkpoint_dir, degrees, degree_thres, task_idxes, criterion="multiclass", evaluator="accuracy", monitor="avg", decoupling=False,
        collect_gradient_step = 10, update_lr = 1e-4, record_step = 10, affinity_dir = '', affinity_type='tag', target_tasks = []):
        super().__init__(model, optimizer, data, train_loader, test_loader, device, epochs, log_steps, 
        checkpoint_dir, degrees, degree_thres, task_idxes, criterion, evaluator, monitor, decoupling)

        self.collect_gradient_step = collect_gradient_step
        self.update_lr = update_lr
        self.affinity_dir = affinity_dir

        self.global_step = 0
        self.record_step = record_step

        self.affinity_type = affinity_type

        tasks = self.task_idxes
        self.task_gains = {str(task): dict([(str(t), []) for t in tasks]) for task in tasks}

        self.target_tasks = target_tasks if len(target_tasks) > 0 else self.task_idxes
    
    def compute_loss_and_gradients(self, task_id, step=1):
        loss = 0; count=0
        
        assert self.decoupling # only applies to decoupling case
        for batch in self.train_loader:
            if count > step:
                break
            xs, y, train_mask = batch
            xs, y, train_mask = xs.to(self.device), y.to(self.device), train_mask.to(self.device)
            xs = [x for x in torch.split(xs, self.input_dim, -1)]
            self.optimizer.zero_grad()
            
            outputs = self.model(xs, return_softmax=self.criterion=="multiclass")
            labels = y.squeeze(1) if self.criterion == "multiclass"  else y

            if len(train_mask.shape) == 2:
                loss = 0; sample_count = 0
                for task_idx in range(train_mask.shape[1]):
                    task_train_mask = train_mask[:, task_idx]
                    loss += names_to_criterions[self.criterion](outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx])*task_train_mask.sum()
                    sample_count += task_train_mask.sum()
                loss = loss/sample_count
            else:
                loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])

            loss += 1
            count += 1

        loss = loss/count
        feature_gradients = grad(loss, self.model.parameters(), retain_graph=False, create_graph=False,
                             allow_unused=True)
        return loss.cpu().item(), feature_gradients

    def update_task_gains(self, step_gains):
        for task, task_step_gain in step_gains.items():
            for other_task in task_step_gain.keys():
                self.task_gains[task][other_task].append(task_step_gain[other_task])
        # self.save_task_gains()
        
    def save_task_gains(self):
        for task, gains in self.task_gains.items():
            for other_task in gains.keys():
                gains[other_task] = np.mean(gains[other_task])
        
        # save the task affinity
        with open(self.affinity_dir, "w") as f:
            task_affinity = json.dumps(self.task_gains)
            f.write(task_affinity)
        print("Saving TAG task affinity...")
        print(task_affinity)

    def compute_tag_task_gains(self):
        task_gain = {str(task): dict() for task in self.target_tasks}

        # 1. collect task losses
        task_losses = {}
        task_gradients = {}

        for task in self.task_idxes:
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(task, self.collect_gradient_step)
            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients
        
        for task in self.task_idxes:
            # 2. take a gradient step on the task loss
            encoder_weights = list(self.model.parameters())
            encoder_gradients = task_gradients[task]
            for i, weight in enumerate(encoder_weights):
                weight.data -= encoder_gradients[i].data * self.update_lr

            # 3. evaluate losses on the target task
            other_tasks =  self.target_tasks
            for target in other_tasks:
                update_loss, _ = self.compute_loss_and_gradients(target, self.collect_gradient_step)
                task_gain[str(target)][str(task)] =  1 - update_loss/task_losses[target]

            # 4. restore weights
            for i, weight in enumerate(encoder_weights):
                weight.data += encoder_gradients[i].data * self.update_lr

        return task_gain

    def compute_cs_task_gains(self):
        task_gain = {str(task): dict() for task in self.target_tasks}

        # 1. collect task losses and gradients
        task_losses = {}
        task_gradients = {}
    
        for task in self.task_idxes:
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(task, self.collect_gradient_step)
            tmp_gradients = tmp_gradients[:-1] # remove the predictor gradient
            tmp_gradients = torch.concat([gradient.view(-1) for gradient in tmp_gradients])
            
            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients
    
        # 2. compute cosine similarity
        other_tasks =  self.target_tasks
        for other_task in other_tasks:
            for task in self.task_idxes:
                task_gain[str(other_task)][str(task)] = F.cosine_similarity(
                    task_gradients[other_task], task_gradients[task], dim=0
                ).cpu().item()

        return task_gain

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0; steps = 0
        if self.decoupling:
            # For decoupling trainer, the train loader only loads propagated features and labels
            for batch in self.train_loader:
                xs, y, train_mask = batch
                xs, y, train_mask = xs.to(self.device), y.to(self.device), train_mask.to(self.device)
                xs = [x for x in torch.split(xs, self.input_dim, -1)]
                self.optimizer.zero_grad()
                
                outputs = self.model(xs, return_softmax=self.criterion=="multiclass")
                labels = y.squeeze(1) if self.criterion == "multiclass"  else y

                if len(train_mask.shape) == 2:
                    loss = 0; sample_count = 0
                    for task_idx in range(train_mask.shape[1]):
                        task_train_mask = train_mask[:, task_idx]
                        loss += names_to_criterions[self.criterion](outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx])*task_train_mask.sum()
                        sample_count += task_train_mask.sum()
                    loss = loss/sample_count
                else:
                    loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1

                # compute iter-task affinity
                if self.global_step % self.record_step == 0:
                    if self.affinity_type == 'tag':
                        step_task_gains = self.compute_tag_task_gains()
                    elif self.affinity_type == 'cs':
                        step_task_gains = self.compute_cs_task_gains()
                    self.update_task_gains(step_task_gains)
                self.global_step += 1
        else:
            for data in self.train_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                
                if hasattr(data, "adj_t") and data.adj_t is not None:
                    outputs = self.model(data.x, edge_index = data.adj_t, return_softmax=self.criterion=="multiclass")
                else:
                    outputs = self.model(data.x, edge_index = data.edge_index, return_softmax=self.criterion=="multiclass")
                labels = data.y.squeeze(1) if self.criterion == "multiclass"  else data.y
                train_mask = data.train_mask

                if len(train_mask.shape) == 2:
                    loss = 0; sample_count = 0
                    for task_idx in range(train_mask.shape[1]):
                        task_train_mask = train_mask[:, task_idx]
                        loss += names_to_criterions[self.criterion](outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx])*task_train_mask.sum()
                        sample_count += task_train_mask.sum()
                    loss = loss/sample_count
                else:
                    loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1
        
        return total_loss / steps