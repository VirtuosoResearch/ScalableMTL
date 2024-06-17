"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
import nltk
import numpy as np
from datasets.utils import disable_progress_bar
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from data_loader import Seq2SeqCollator, CLMCollator, Seq2SeqMultiInstructionCollator, CLMMultiInstructionCollator
from trainer import NITrainer, DecoderOnlyTrainer, DenserEvalCallback
from compute_metrics import compute_metrics, compute_grouped_metrics
from promptsource.templates import DatasetTemplates, Template
from task_constants import task_to_benchmark, task_to_null_template, task_is_generative_task
from utils.adjustment import add_adjustment_term, split_gpt_self_attention
from utils.template_utils import asssign_template
from copy import deepcopy
from transformers.optimization import Adafactor, AdafactorSchedule
from utils.util import add_result_to_csv

from peft import get_peft_model, PrefixTuningConfig, TaskType, PromptTuningConfig, PromptTuningInit, LoraConfig
import shutil

import os
os.environ["WANDB_DISABLED"] = "true"

disable_progress_bar()
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    New arguments
    """
    task_name : str = field(
        default=None, metadata={"help": "The name of the task to train on."}
    )
    template_idxs : int = field(
        default=-1, metadata={
            "nargs": "+",
            "help": "The index of the template to use."
            }
    )
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    train_adjustment: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, train the adjustment terms."}
    )
    freeze_module : Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, freeze the pretrained parameters."}
    )
    module_name : Optional[str] = field(
        default="query",
        metadata={"help": "If specified, finetune the pretrained model."}
    )
    freeze_layer : Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, freeze the pretrained parameters."}
    )
    layer_index : Optional[int] = field(
        default=0,
        metadata={"help": "If specified, finetune the pretrained model."}
    )
    runs : Optional[int] = field(
        default=3,
        metadata={"help": "The number of random seeds"}
    )
    downsample : Optional[int] = field(
        default=-1,
        metadata={"help": "If specified, downsample the training set."}
    )
    use_Adafactor_optm: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, use Adafactor optimizer"}
    )
    is_generative_task: Optional[bool] = field(
        default=False,
        metadata={"help": "If the task is a generative task"}
    )
    save_name: Optional[str] = field(
        default="test",
        metadata={"help": "The name of the model to save."}
    )

    train_prefix : Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, train the prefix."}
    )

    train_prompt_tuning: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, train the prompt tuning."}
    )

    train_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, train Lora."}
    )

    lora_rank: Optional[int] = field(
        default=32,
        metadata={"help": "If specified, train Lora."}
    )

    lora_alpha: Optional[float] = field(
        default=32,
        metadata={"help": "If specified, train Lora."}
    )


    run: Optional[int] = field(
        default=0,
        metadata={"help": "The run number"}
    )

    create_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "If specified, create projection matrix."}
    )

    project_dim: Optional[int] = field(
        default=100,
        metadata={"help": "The dimension of the projection matrix."}
    )

    load_model_dir: Optional[str] = field(
        default="test",
        metadata={"help": "The directory to load the model."}
    )

def prepare_inputs(batch, device):
    for t in batch:
        if torch.is_tensor(batch[t]):
            batch[t] = batch[t].to(device)
    return batch

def get_task_gradients(model, tokenizer, train_loader, device, projection=None):
    '''
    Compute gradients: for a single task idx
    '''
    model.train()

    parameters = [param for param in model.parameters() if param.requires_grad]

    gradients = []; model_outputs = []; returned_gradients = []; labels = []
    # For decoupling trainer, the train loader only loads propagated features and labels
    for batch in train_loader:
        batch = prepare_inputs(batch, device)
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        batch_size = labels.shape[0]

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(logits[np.arange(batch_size), 0], labels[np.arange(batch_size), 0])
        print(loss.mean())

        for i in range(len(loss)):
            tmp_gradients = torch.autograd.grad(loss[i], parameters, retain_graph=True, create_graph=False)
            tmp_gradients = torch.cat([gradient.view(-1) for gradient in tmp_gradients]).cpu().numpy() # flatten gradients
            gradients.append(tmp_gradients)
            
        if projection is not None:
            if len(gradients) > 100:
                gradients = np.array(gradients)
                gradients = np.matmul(gradients, projection)
                returned_gradients.append(gradients)
                gradients = []
    
    gradients = np.array(gradients)
    model_outputs = np.concatenate(model_outputs)
    labels = np.concatenate(labels)
        
    if projection is not None:
        if len(gradients) > 0:
            gradients = np.matmul(gradients, projection)
            returned_gradients.append(gradients)
        gradients = np.concatenate(returned_gradients, axis=0)
        return gradients, model_outputs, labels
    return gradients, model_outputs, labels

def generate_state_dict(model, state_dict, coef, device):
    # reshape coef
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        new_state_dict[key] = state_dict[key].clone() + \
            torch.FloatTensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
        cur_len += param_len
    return new_state_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NITrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
            "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/t5-v1_1-base"
        ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    results = {}
    logging_metrics = ["loss", "accuracy", "exact_match", "rouge1", "rougeL", "rouge2"]

    # idx_dir = "_".join([str(idx) for idx in data_args.template_idxs])[:100]
    # training_args.output_dir = os.path.join(training_args.output_dir,
    #                 "{}_{}_{}_run_{}".format(
    #                     data_args.task_name, idx_dir, 
    #                     model_args.model_name_or_path.replace("/", "_") ,run)
    #                 )

    # Set seed before initializing model.
    seed = np.random.randint(0, 1e6)
    set_seed(seed)
    training_args.seed = seed

    # split dataset into train, validation, test
    task_name = data_args.task_name
    benchmark_name = task_to_benchmark[task_name] # Test set does not have labels
    training_template_idxs = data_args.template_idxs

    if benchmark_name is not None:
        raw_datasets = load_dataset(benchmark_name, task_name)
        dataset_templates = DatasetTemplates(benchmark_name, task_name)
    else:
        raw_datasets = load_dataset(task_name)
        dataset_templates = DatasetTemplates(task_name)
    
    keys = list(dataset_templates.name_to_id_mapping.keys())
    templates = [dataset_templates[key] for key in keys]

    template = templates[0]
    # unifying labels space
    answer_choices = template.answer_choices
    for cur_template in  [dataset_templates[key] for key in keys]:
        cur_template.answer_choices = answer_choices
    
    # add generated instructions
    task_templates = json.load(open("task_templates.json", "r"))
    for idx, cur_jinja in enumerate(task_templates[task_name]):
        cur_template = Template(name="null", jinja=cur_jinja, reference="", answer_choices = template.answer_choices)
        templates.append(cur_template)
    print("number of templates: {}".format(len(templates)))

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]

    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]

    print("train dataset size: {} \nvalidation dataset size: {} \ntest dataset size: {}".format(len(train_dataset), len(eval_dataset), len(predict_dataset)))

    # mix templates in training datasets
    multitask_train_dataset = []
    task_to_train_datasets = {}
    for idx in training_template_idxs:
        tmp_dataset = deepcopy(train_dataset)
        tmp_dataset = tmp_dataset.map(
                asssign_template(idx), batched=True
            )
        task_to_train_datasets[idx] = tmp_dataset
        multitask_train_dataset.append(tmp_dataset)
    multitask_train_dataset = datasets.concatenate_datasets(multitask_train_dataset)
    multitask_train_dataset = multitask_train_dataset.shuffle(seed=42)
    multitask_train_dataset = multitask_train_dataset.flatten_indices()
    print("multitask train dataset size: {}".format(len(multitask_train_dataset)))
    
    # Prepare validation and test dataset
    multitask_eval_dataset = []
    for idx in training_template_idxs:
        tmp_dataset = deepcopy(eval_dataset)
        tmp_dataset = tmp_dataset.map(
                asssign_template(idx), batched=True
            )
        multitask_eval_dataset.append(tmp_dataset)
    multitask_eval_dataset = datasets.concatenate_datasets(multitask_eval_dataset)

    eval_dataset = eval_dataset.map(asssign_template(-1), batched=True)
    predict_dataset = predict_dataset.map(asssign_template(-1), batched=True)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_name_or_path in [
            "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        ]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if training_args.train_prefix:
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if training_args.train_prompt_tuning:
            peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=20,
                prompt_tuning_init_text=template.jinja,
                tokenizer_name_or_path=model_args.model_name_or_path,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        if training_args.train_lora:
            config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()

    device = torch.device("cuda")
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
        
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = Seq2SeqMultiInstructionCollator(
        tokenizer, model, templates=templates,
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
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 

    task_to_train_dataloaders = {}
    for idx, task_dataset in task_to_train_datasets.items():
        train_data_loader = torch.utils.data.DataLoader(task_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator, shuffle=True)
        task_to_train_dataloaders[idx] = train_data_loader

    load_model_dir = "./saved/" + training_args.load_model_dir + "/pytorch_model.bin"
    state_dict = torch.load(load_model_dir, map_location=device)

    # collect gradients
    gradients_dir = f"./gradients/{data_args.task_name}/run_{training_args.run}"
    assert os.path.exists(gradients_dir)
        
    matrix_P = np.load(f"./gradients/{data_args.task_name}/projection_matrix_{int(training_args.run%10)}.npy")
    
    from sklearn.linear_model import LogisticRegression

    # Metric
    def compute_ni_metrics(dataset, preds, save_prefix=None):
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [[template.apply(e)[1]] for e in dataset] 
        result = compute_metrics(predictions=decoded_preds, references=references)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result

    # Initialize our Trainer
    trainer = NITrainer(
        model=model,
        args=training_args,
        train_dataset=multitask_train_dataset if training_args.do_train else None,
        eval_dataset=multitask_eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    with open(f"./sampled_tasks/sample_{data_args.task_name}.txt", "r") as f:
        for line in f.readlines():
            task_idxes = line.strip().split()
            task_idxes = [int(task_idx) for task_idx in task_idxes]
            print("Selected task idxes", task_idxes)

            task_to_gradients = {}
            task_to_outputs = {}
            for task_idx in task_idxes:
                gradients = np.load(f"{gradients_dir}/task_{task_idx}_gradients.npy")
                outputs = np.load(f"{gradients_dir}/task_{task_idx}_outputs.npy")
                task_to_gradients[task_idx] = gradients
                task_to_outputs[task_idx] = outputs

            ''' Train logistic regression '''
            gradients = np.concatenate([task_to_gradients[task_idx] for task_idx in task_idxes], axis=0)
            gradients = gradients @ matrix_P
            labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
            mask = np.copy(labels)
            mask[labels == 0] = -1
            mask = mask.reshape(-1, 1)
            gradients = gradients*mask
            

            train_num = int(len(gradients)*0.8)
            train_gradients, train_labels = gradients[:train_num], labels[:train_num]
            test_gradients, test_labels = gradients[train_num:], labels[train_num:]

            clf = LogisticRegression(random_state=0, penalty='l2', C=1e-2) # solver='liblinear' 
            clf.fit(train_gradients, train_labels)
            print(clf.score(test_gradients, test_labels))
            
            # if projected:  # else: coef = clf.coef_.copy().flatten()
            proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
            print("L2 norm", np.linalg.norm(proj_coef))
            print(matrix_P.shape, proj_coef.shape)
            coef = matrix_P @ proj_coef.flatten()
        
            print("L2 norm", np.linalg.norm(coef))

            new_state_dict = generate_state_dict(model, state_dict, coef, device)
            model.load_state_dict(state_dict)
            model.load_state_dict(new_state_dict, strict=False)

            # Evaluation
            max_length = (
                training_args.generation_max_length
                if training_args.generation_max_length is not None
                else data_args.max_target_length
            )
            num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

            all_metrics = {}
            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                ''' Evaluate for all possible templates '''
                for template_idx in task_idxes:
                    eval_dataset = eval_dataset.map(asssign_template(template_idx), batched=True)

                    trainer.eval_dataset = eval_dataset
                    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
                    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                    new_metrics = {}
                    for key, val in metrics.items():
                        if key.startswith("eval"):
                            key = key.replace("eval", f"eval_{template_idx}")
                            new_metrics[key] = val

                    all_metrics.update(new_metrics)

            for key, val in all_metrics.items():
                for metric_name in logging_metrics:
                    if (metric_name in key) and (key not in results):
                        results[key] = [val, ]
                    elif (metric_name in key) and (key in results):
                        results[key].append(val)

            for key, val in results.items():
                print("{}: {:.4f} +/- {:.4f}".format(key, np.mean(val), np.std(val)))

            # save results into .csv
            file_dir = os.path.join("./results/", training_args.save_name)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
            
            for template_idx in task_idxes:
                # save validation results
                if training_args.do_eval:
                    result_datapoint = {
                        "Task": template_idx, 
                        "Trained on": task_idxes,
                    }
                    for key, vals in results.items():
                        if f"eval_{template_idx}_" in key:
                            key = key.replace(f"eval_{template_idx}", "eval")
                            result_datapoint[key] = np.mean(vals)
                            result_datapoint[key+"_std"] = np.std(vals)
                    file_name = os.path.join(file_dir, "{}_valid.csv".format(training_args.save_name))
                    add_result_to_csv(result_datapoint, file_name)


if __name__ == "__main__":
    main()