import numpy as np
import copy

class asssign_template:

    def __init__(self, template_id):
        self.template_id = template_id

    def __call__(self, examples):
        example_keys = list(examples.keys())
        num_samples = len(examples[example_keys[0]])
        examples["template_id"] = np.ones(num_samples, dtype=int) * self.template_id
        return examples
    
class apply_template:
    def __init__(self, template):
        self.template = template

    def __call__(self, examples):
        example_keys = list(examples.keys())
        num_samples = len(examples[example_keys[0]])
        if "input" in example_keys:
            examples["old_input"] = copy.deepcopy(examples["input"])
            example_keys.remove("input")
            example_keys.append("old_input")
        examples["input"] = []
        examples["output"] = []
        for i in range(num_samples):
            example = {key: examples[key][i] for key in example_keys if len(examples[key])>i}
            converted_instance = self.template.apply(example)
            examples["input"].append(converted_instance[0])
            examples["output"].append(converted_instance[1])
        return examples

# def sample_templates(examples):
#     ''' Randomly sample templates to each example '''
#     example_keys = list(examples.keys())
#     num_samples = len(examples[example_keys[0]])

#     # sample templates
#     template_ids = np.random.choice(len(templates), num_samples, replace=True)
#     examples["template_id"] = template_ids
#     return examples