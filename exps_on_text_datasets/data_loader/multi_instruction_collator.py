import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)

@dataclass
class Seq2SeqMultiInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    templates: Optional[Any] = None
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
        
        for instance in batch:
            template = self.templates[instance["template_id"]]
            converted_instance = template.apply(instance)
            converted_batch.append({"input": converted_instance[0], "output": converted_instance[1]})            
        
        # prepare input sources
        sources = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        labels = [instance["output"] for instance in converted_batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs
    
@dataclass
class Seq2SeqMultiInstructionEnsembleCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    templates: Optional[Any] = None
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
    training: bool=True

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = []
        
        for instance in batch:
            template = self.templates[instance["template_id"]]
            converted_instance = template.apply(instance)
            converted_batch.append({"input": converted_instance[0], "output": converted_instance[1]})            
        
        # prepare input sources
        sources = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        labels = [instance["output"] for instance in converted_batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
        
        if self.training:
            model_inputs["prompt_idxs"] = [instance["template_id"] for instance in batch]
        else:
            model_inputs["prompt_idxs"] = [-1 for instance in batch]

        return model_inputs
    


@dataclass
class CLMMultiInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    templates: Optional[Any] = None
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
        for instance in batch:
            template = self.templates[instance["template_id"]]
            converted_instance = template.apply(instance)
            converted_batch.append({"input": converted_instance[0], "output": converted_instance[1]})         
        
        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            source = source.replace("\n", " ")
            source = " ".join(source.split())
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []; label_lengths = []
        for instance in converted_batch:
            label = instance["output"]
            label = label.replace("\n", " ")
            label = " ".join(label.split())
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))
            label_lengths.append(min(len(tokenized_label), self.max_target_length))

        inputs = [source + " " + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id            
        return model_inputs