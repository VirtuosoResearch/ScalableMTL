### Overview

This code implements a gradient-based estimation method for computing task affinity, namely Grad-TAE. The experiments involve multi-label classification on graphs using community detection labels, fine-tuning language models with multiple instructions, and image classification data sets of various domains. 



### Usage

Our algorithm contains four steps:

1. Multitask training on all tasks to obtain meta-initializations 

2. Computing and projecting gradients of every training sample on the meta-initializations

3. Estimating the fine-tuned model parameters for a subset of tasks by solving a linear regression on the projected gradients to align with task labels 

4. Clustering task affinity scores to generate task groups


#### **Multitask training to obtain meta-initializations**

This step trains a multitask learning model on the combination of all tasks. 

- For graph datasets, use `train_node_pred_multitask.py` to train the model. Modify the `--dataset` to specify the dataset. This script will save the model checkpoints under a `./saved/` folder. Please create one before usage. 
- For image classification tasks, see `./exps_on_image_datasets/scripts/train_multitask.sh` for a bash script example. 
- For fine-tuning language models,  use `train_multi_instruction.py` to fine-tune a model on all instructions. 

See `./exps_on_text_datasets/scripts/train_multi_instructions.sh` for a bash script example.

- Modify the `--task_name` to specify the dataset name. 
- Specify the `--template_idx` from 1 to 100 in order to train on all the instructions. 



#### **Computing and projecting gradients on meta-initializations**

- For graph datasets, use `fast_estimate_collect_gradients.py`. Specify `--project_dim` as the number of projections. This file will save the projection matrix and all projected gradients under a `./gradients/` folder. Please create the folder before usage. 

- For fine-tuning language models, use `fast_estimate_collect_gradients.py`. 
- For image classification tasks, see `./exps_on_image_datasets/scripts/compute_gradients.sh` for a bash script example. 

 See `exps_on_text_datasets/scripts/collect_gradients.sh` for a bash script example. 

- Use `--load_model_dir` to specify a saved checkpoint directory as the base model. 



#### Estimate the fine-tuned model parameters for a subset of tasks

- For graph datasets, use `fast_estimate_linear_model.py`. Specify `--save_name` for the file to save the evaluation results of estimated models. 
  - Inside the file, one can modify the subsets collection file under `./sampled_tasks/sample_{dataset}_128_10.txt` to specify the sampled subsets of tasks. Usually, it should be randomly sampled subsets. 

- For fine-tuning language models, use `fast_estimate_linear_model.py`. See `exps_on_text_datasets/scripts/solve_linear_regression.sh` for a bash script example. 

- For image classification tasks, see `./exps_on_image_datasets/scripts/compute_logistic_regression.sh` for a bash script example. 



#### Clustering task affinity scores to generate task groups 

- We provide an implementation of our SDP-based clustering algorithm in `exps_on_graph_datasets/run_clustering.py`. 

- One can load a computed task affinity score matrix and apply the clustering on top of the matrix. Then, we train one model on each subset of tasks. 



### Requirements

We list the package requirements under each folder. Install related packages within each corresponding folder based on the following: 

```
pip install -r requirements.txt
```



### Data Preparation

**Community detection.** We provide the datasets for conducting community detection named `data.zip` under the `./data/` folder used. Unzip the files under the folder, and then one can directly load them in the code.

**Language model fine-tuning.** The code will automatically download the datasets. Please specify the name of the dataset while using it.  

**Image classification.** We provide image classification data sets sampled from DomainNet named `domain_net.zip`. Download the zip file from this [link](https://drive.google.com/file/d/1OmeNf_sWHLUQhSIA2ICvm4jj4xyadG16/view?usp=sharing). Unzip the file under the `./data/` folder, then one can directly load them in the code.

### Citation

If you find this repository useful or happen to use it in a research paper, please cite our work using the following bib information.

```
@article{li2024scalable,
  title={Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity},
  author={Li, Dongyue and Sharma, Aneesh and Zhang, Hongyang R},
  journal={SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
