import os
from pathlib import Path
import argparse
import collections
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import *
from utils import prepare_device, deep_copy

from torch.utils.data import DataLoader
from data_loader.multitask_dataset import MultitaskBatchSampler, MultitaskCollator, MultitaskDataset 
from model.modeling_vit import VisionTransformer, CONFIGS

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    assert config["data_loader"]["type"] == "DomainNetDataLoader"
    task_to_train_datasets = {}
    task_to_train_dataloaders = {}
    task_to_valid_dataloaders = {}
    task_to_test_dataloaders = {}

    for domain in args.domains:
        config['data_loader']['args']['domain'] = domain
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
        task_to_train_datasets[domain] = train_data_loader.dataset
        task_to_train_dataloaders[domain] = train_data_loader
        task_to_valid_dataloaders[domain] = valid_data_loader
        task_to_test_dataloaders[domain] = test_data_loader
        print("Domain: {} Train Size: {} Valid Size: {} Test Size: {}".format(
            domain, len(train_data_loader.sampler), len(valid_data_loader.sampler), len(test_data_loader.sampler))
        )
    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, config['data_loader']['args']['batch_size'])
    # multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
    )
    print("Multitask Train Size: {}".format(len(multitask_train_dataset)))
    
    # build model architecture, then print to console
    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
        model.load_from(np.load(args.vit_pretrained_dir))
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    results = {}
    for run in range(args.runs):
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        checkpoint_dir = os.path.join(
        "./saved", 
        "{}_{}_{}_run_{}".format(config["arch"]["type"], config["data_loader"]["type"], 
                                 "_".join(args.domains),
                                 run))
        trainer = MultitaskTrainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=multitask_train_dataloader,
                        valid_data_loaders=task_to_valid_dataloaders,
                        test_data_loaders=task_to_test_dataloaders,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir = checkpoint_dir)

        trainer.train()
        test_metrics = trainer.test()
        for key, value in test_metrics.items():
            if key not in metrics:
                results[key] = []
            results[key].append(value)
    for key, values in results.items():
        print("Test {}:  {:1.4f} +/- {:1.4f}".format(key, np.mean(values), np.std(values)))

    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    for domain in args.domains:
        # save validation results
        result_datapoint = {
            "Task": domain, 
            "Trained on": " ".join(args.domains),
        }
        for key, vals in results.items():
            if domain in key:
                metric_key = "_".join(key.split("_")[1:])
                result_datapoint[metric_key] = np.mean(vals)
                result_datapoint[metric_key+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_valid.csv".format(args.save_name))
        add_result_to_csv(result_datapoint, file_name)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--load_robust', action="store_true")
    args.add_argument('--robust_model_dir', type=str, default="./robust_models/resnet18_l2_eps1.ckpt")
    args.add_argument('--data_frac', type=float, default=1.0)
    args.add_argument('--domains', type=str, nargs="+", default=["clipart", "infograph", "painting", "quickdraw", "real", "sketch"])
    
    args.add_argument('--is_vit', action="store_true")
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="checkpoints/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    
    args.add_argument('--save_name', type=str, default="multitask_domainnet")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--weight_decay'], type=float, target="optimizer;args;weight_decay"),
        CustomArgs(['--reg_method'], type=str, target='reg_method'),
        CustomArgs(['--reg_norm'], type=str, target='reg_norm'),
        CustomArgs(['--reg_extractor'], type=float, target='reg_extractor'),
        CustomArgs(['--reg_predictor'], type=float, target='reg_predictor'),
        CustomArgs(['--scale_factor'], type=float, target="scale_factor"),
        CustomArgs(['--domain'], type=str, target="data_loader;args;domain"),
        CustomArgs(['--sample'], type=int, target="data_loader;args;sample"),
        CustomArgs(['--early_stop'], type=int, target="trainer;early_stop"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
