import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)

@dataclass
class CLMCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    template: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False  # deprecated
    add_task_definition: bool = True  # deprecated
    num_pos_examples: int = 0  # deprecated
    num_neg_examples: int = 0  # deprecated
    add_explanation: bool = False  # deprecated
    tk_instruct: bool = False  # deprecated
    text_only: bool=False # deprecated

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = []
        assert self.template is not None
        for instance in batch:
            converted_instance = self.template.apply(instance)
            converted_batch.append({"input": converted_instance[0], "output": converted_instance[1]})            
        
        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []
        for instance in converted_batch:
            label = instance["output"]
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))

        inputs = [source + "\n" + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length+1] = self.label_pad_token_id
            
        return model_inputs