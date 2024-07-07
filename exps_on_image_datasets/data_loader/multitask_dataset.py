import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler

class MultitaskDataset(Dataset):

    def __init__(self, task_to_datasets):
        self.task_names = list(task_to_datasets.keys())
        self.datasets = list(task_to_datasets.values())
        self.task_to_datasets = task_to_datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        task_name, sample_id = idx
        # return {"task_name": task_name, "sample": self.task_to_datasets[task_name][sample_id]}
        return self.task_to_datasets[task_name][sample_id]


class MultitaskBatchSampler(BatchSampler):

    def __init__(
        self,
        task_to_datasets,
        batch_size
    ):
        self._task_to_datasets = task_to_datasets
        self._task_names = list(task_to_datasets.keys())
        self._datasets = list(task_to_datasets.values())
        self._batch_size = batch_size

        train_data_list = []
        for dataset in self._datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list
    
    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list)
        for local_task_idx in all_indices:
            task_id = self._task_names[local_task_idx]
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt = 0.5, extra_task_ratio = 0):
        '''
        Generate sampling indices of tasks
        mix_opt: whether shuffle the auxiliary task indices and main indices
        extra_task_ratio: ratio of auxiliary tasks to the main task (task 0)
        '''
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()
        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices

class MultitaskCollator:
    
    def __init__(self, task_to_collator: dict):
        self.task_to_collator = task_to_collator

    def collator_fn(self, batch):
        task_name = batch[0]["task_name"]   
        batch = [sample['sample'] for sample in batch]
        batch = self.task_to_collator[task_name](batch)
        return {"task_name": task_name, "data": batch}