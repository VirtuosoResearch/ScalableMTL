# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, task_name = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.task_name: str = task_name
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task


class NaturalInstructionsSingleTask(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")]
                    },
                    "Instance License": [datasets.Value("string")]
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir
        task_name = self.config.task_name

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "task_dir": task_dir, 
                    "task_name": task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "task_dir": task_dir,
                    "task_name": task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "task_dir": task_dir, 
                    "task_name": task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "test"
                }),
        ]

    def _generate_examples(self, task_dir=None, task_name=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks {task_name}")
        
        task_path = os.path.join(task_dir, task_name + ".json")
        with open(task_path, encoding="utf-8") as task_f:
            s = task_f.read()
            task_data = json.loads(s)
            task_data["Task"] = task_name
            if "Instruction Source" in task_data:
                task_data.pop("Instruction Source")
            all_instances = task_data.pop("Instances")
            instances = all_instances
            if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                random.shuffle(instances)
                instances = instances[:max_num_instances_per_task]
            for idx, instance in enumerate(instances):
                example = task_data.copy()
                example["id"] = instance["id"]
                example["Instance"] = instance
                yield f"{task_name}_{idx}", example

