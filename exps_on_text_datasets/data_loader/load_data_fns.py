from promptsource.templates import DatasetTemplates, Template
from task_constants import task_to_benchmark, task_to_null_template
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from data_loader import DataCollatorForNI, Seq2SeqCollator, CLMCollator
import numpy as np

data_dir = "./data/splits/default/"
task_dir = "./data/tasks/"
cache_dir = "./cache/"

class data_args:
    pad_to_max_length = False
    max_source_length = 1024
    max_target_length = 128
    ignore_pad_token_for_loss = True
    add_task_name = True
    add_task_definition = True
    num_pos_examples = 1
    num_neg_examples = 1
    add_explanation = True
    tk_instruct = False
    batch_size = 8

def load_ni_task(task_name, model, tokenizer):
    raw_datasets = load_dataset(
        "ni_single_dataset.py", 
        data_dir = data_dir, 
        task_dir = task_dir, 
        cache_dir = cache_dir,
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=500,
        task_name = task_name
    )

    dataset_size = len(raw_datasets["train"])
    train_size = int(dataset_size * 0.6)
    val_size = int(dataset_size * 0.2)
    rng = np.random.default_rng(42)
    indices = rng.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.select(train_indices)

    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.select(val_indices)

    predict_dataset = raw_datasets["test"]
    predict_dataset = predict_dataset.select(test_indices)

    print("train dataset size: {} \nvalidation dataset size: {} \ntest dataset size: {}".format(len(train_dataset), len(eval_dataset), len(predict_dataset)))

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    collator = DataCollatorForNI(
        tokenizer, model,
        padding="max_length" if data_args.pad_to_max_length else "longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        add_task_name=data_args.add_task_name,
        add_task_definition=data_args.add_task_definition,
        num_pos_examples=data_args.num_pos_examples,
        num_neg_examples=data_args.num_neg_examples,
        add_explanation=data_args.add_explanation,
        tk_instruct=data_args.tk_instruct
    )

    from torch.utils.data.dataloader import DataLoader

    train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collator, batch_size=data_args.batch_size
        )
    eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=collator, batch_size=data_args.batch_size
        )
    predict_dataloader = DataLoader(
            predict_dataset, shuffle=False, collate_fn=collator, batch_size=data_args.batch_size
        )

    return train_dataloader, eval_dataloader, predict_dataloader


def load_promptsource_task(task_name, template_idx, model, tokenizer, collator_class):
    benchmark_name = task_to_benchmark[task_name] # Test set does not have labels
    raw_datasets = load_dataset(benchmark_name, task_name)

    dataset_templates = DatasetTemplates(benchmark_name, task_name)
    keys = list(dataset_templates.name_to_id_mapping.keys())
    template = dataset_templates[keys[template_idx]]
    if template_idx < 0: 
        template = Template(name="null", jinja=task_to_null_template[task_name], reference="", answer_choices = template.answer_choices)

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    collator = collator_class(
        tokenizer, model, template=template,
        padding="max_length" if data_args.pad_to_max_length else "longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        add_task_name=data_args.add_task_name,
        add_task_definition=data_args.add_task_definition,
        num_pos_examples=data_args.num_pos_examples,
        num_neg_examples=data_args.num_neg_examples,
        add_explanation=data_args.add_explanation,
        tk_instruct=data_args.tk_instruct
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]

    from torch.utils.data.dataloader import DataLoader

    train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collator, batch_size=data_args.batch_size
        )
    eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=collator, batch_size=data_args.batch_size
        )
    predict_dataloader = DataLoader(
            predict_dataset, shuffle=False, collate_fn=collator, batch_size=data_args.batch_size
        )

    return train_dataloader, eval_dataloader, predict_dataloader