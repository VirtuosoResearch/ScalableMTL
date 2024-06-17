from typing import Any
import torch 
import torch.nn as nn

class WeightAdjustment:
    '''
    Add one ajustment term and a gate to a layer of a model.
    '''

    def __init__(self, name) -> None:
        self.name = name

    def compute_weight(self, module):
        adjustment = getattr(module, self.name + '_adjustment')
        gate = getattr(module, self.name + '_gate')
        weight = getattr(module, self.name + "_pretrained")
        return weight + adjustment * torch.sigmoid(gate)

    @staticmethod
    def apply(module, name: str):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAdjustment) and hook.name == name:
                raise RuntimeError("Cannot register two adjustment hooks on "
                                   "the same parameter {}".format(name))

        fn = WeightAdjustment(name)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        module.register_parameter(name + "_pretrained", nn.Parameter(weight.data))
        module.register_parameter(name + '_adjustment', nn.Parameter(torch.randn_like(weight.data)))
        module.register_parameter(name + '_gate', nn.Parameter(torch.randn(size=(1, ))))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn


    def __call__(self, module, inputs) -> Any:
        setattr(module, self.name, self.compute_weight(module))

def add_adjustment_term(module, name):
    WeightAdjustment.apply(module, name)
    return module


class SplitGPTSelfAttention:
    '''
    Add one ajustment term and a gate to a layer of a model.
    '''

    def __init__(self, name) -> None:
        self.name = name

    def compute_weight(self, module):
        query = getattr(module, self.name + '_q')
        key = getattr(module, self.name + '_k')
        value = getattr(module, self.name + "_v")
        return torch.concatenate([query, key, value], dim=1)

    @staticmethod
    def apply(module, name: str):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAdjustment) and hook.name == name:
                raise RuntimeError("Cannot register two adjustment hooks on "
                                   "the same parameter {}".format(name))

        fn = SplitGPTSelfAttention(name)
        weight = getattr(module, name)
        embedding_dim = weight.shape[0]
        # remove w from parameter list
        del module._parameters[name]

        module.register_parameter(name + "_q", nn.Parameter(weight.data[:, :embedding_dim]))
        module.register_parameter(name + '_k', nn.Parameter(weight.data[:, embedding_dim:embedding_dim*2]))
        module.register_parameter(name + '_v', nn.Parameter(weight.data[:, embedding_dim*2:embedding_dim*3]))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn


    def __call__(self, module, inputs) -> Any:
        setattr(module, self.name, self.compute_weight(module))

def split_gpt_self_attention(module, name):
    SplitGPTSelfAttention.apply(module, name)
    return module