import os 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import argparse
import numpy as np
import shutil

def main(args):
    task_list = np.arange(args.num_tasks)
    task_list = [str(task) for task in task_list]

    # load existing cluster structures 
    cluster_dir = os.path.join("./clusters", "{}.txt".format(args.current_cluster_dir))
    cluster_structures = []
    if os.path.exists(cluster_dir):
        with open(cluster_dir, "r") as f:
            for line in f.readlines():
                cur_cluster = line.strip().split(" ")
                cluster_structures.append(cur_cluster)
    existing_tasks = []
    for cluster_structure in cluster_structures:
        existing_tasks += cluster_structure

    new_tasks = [task for task in task_list if task not in existing_tasks]
    
    num_samples = args.num_samples
    max_task_num = args.max_task_num
    min_task_num = args.min_task_num
    for _ in range(num_samples):
        # create a set of trained task combinations
        sampled_task_dir = os.path.join("./sampled_tasks", "{}.txt".format(args.task_set_name))
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()
            
        with open(sampled_task_dir, "r") as f:
            sampled_tasks = set()
            for line in f.readlines():
                sampled_tasks.add(line.rstrip("\n"))

        # train on a new task combination
        with open(sampled_task_dir, "a") as f:
            # randomly choose one cluster
            if len(cluster_structures) == 0:
                tmp_other_tasks = new_tasks[:]
            else:
                tmp_cluster_idx = np.random.randint(low=0, high=len(cluster_structures))
                tmp_other_tasks = cluster_structures[tmp_cluster_idx][:]
                tmp_other_tasks += new_tasks

            tmp_other_task_num = np.random.randint(
                low=min_task_num, 
                high=max_task_num+1
            )
            tmp_sampled_other_tasks = np.random.choice(tmp_other_tasks, size=tmp_other_task_num, replace=False)
            
            tmp_sampled_tasks = tmp_sampled_other_tasks
            tmp_sampled_tasks.sort()
            tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            
            if tmp_sampled_tasks in sampled_tasks:
                continue
            print(tmp_sampled_tasks)
            
            os.system("CUDA_VISIBLE_DEVICES={} python train_multi_instruction.py \
                        --do_train \
                        --do_eval \
                        --predict_with_generate \
                        --model_name_or_path {} \
                        --max_source_length 512 \
                        --max_target_length 128 \
                        --pad_to_max_length True \
                        --generation_max_length 128 \
                        --data_dir data/splits/default \
                        --task_dir data/tasks \
                        --overwrite_output_dir \
                        --cache_dir ./cache/ \
                        --overwrite_cache \
                        --per_device_train_batch_size 8 \
                        --per_device_eval_batch_size 8 \
                        --gradient_accumulation_steps 1 \
                        --learning_rate {} \
                        --lr_scheduler_type linear \
                        --max_steps {} \
                        --warmup_steps 500 \
                        --logging_strategy steps \
                        --logging_steps 500 \
                        --evaluation_strategy steps \
                        --eval_steps {} \
                        --save_strategy steps \
                        --save_steps {} \
                        --metric_for_best_model eval_exact_match\
                        --greater_is_better True \
                        --downsample {} \
                        --task_name {} \
                        --template_idx {} \
                        --output_dir saved/ \
                        --load_best_model_at_end \
                        --disable_tqdm True \
                        --runs {}\
                        --train_lora True --lora_rank {} --lora_alpha {}\
                        --save_name {} --fp16 True".format(
                            args.device, args.model_name_or_path, args.lr, args.max_steps, args.eval_steps, args.save_steps,
                            args.downsample, args.dataset, tmp_sampled_tasks, args.runs, 
                            args.lora_rank, args.lora_alpha, args.save_name
            ))
            instruction_idxs = "[" + ", ".join(tmp_sampled_tasks.split(" ")) + "]"
            output_dir = os.path.join("saved", "{}_{}_{}".format(args.dataset, instruction_idxs, args.model_name_or_path))
            # delete the output_dir
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            sampled_tasks.add(tmp_sampled_tasks)
            f.write(tmp_sampled_tasks + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--target_tasks", nargs='+', type=str, default=[])

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--min_task_num", type=int, default=3)
    parser.add_argument("--max_task_num", type=int, default=3)

    parser.add_argument("--dataset", type=str, default="rte")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--downsample", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=32)

    parser.add_argument("--task_set_name", type=str, default="sampled_tasks")
    parser.add_argument("--save_name", type=str, default="sampled_tasks")
    parser.add_argument("--current_cluster_dir", type=str, default="none")

    args = parser.parse_args()
    main(args)