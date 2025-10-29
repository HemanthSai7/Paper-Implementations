import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn import functional as F
from safetensors.torch import save_file

import math
from dataclasses import dataclass
from lora_config import LoraConfig
from base_lora import *

__all__ = ["LoRAModel"]


class LoRAModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        config: LoraConfig,
    ):
        super(LoRAModel, self).__init__()

        self.lora_model = model
        self.config = config

        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]

        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        original_trainable_parameters = self._count_trainable_parameters()
        print(f"Original trainable parameters: {original_trainable_parameters}")

        self._disable_all_grads()
        self._apply_lora(self.lora_model)
        self._togle_bias_grad()

        lora_trainable_parameters = self._count_trainable_parameters()
        print_string = ""
        print_string += f"Initial Parameters : {original_trainable_parameters} || "
        print_string += f"LoRA Parameters : {lora_trainable_parameters} || "
        print_string += f"Trainable Proportion : {round(lora_trainable_parameters*100/original_trainable_parameters, 2)}%"

        print(print_string)

    def _count_trainable_parameters(self):

        total_trainable_parameters = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_trainable_parameters += param.numel()
        return total_trainable_parameters
    
    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])
    
    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])
    
    def _apply_lora(self, module):

        for name, child in module.named_children():
            if self._target_module_name_check(name):
                if isinstance(child, nn.Linear):
                    new_layer = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=True if child.bias is not None else False,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        use_rslora=self.config.use_rslora,
                    )
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)
                elif isinstance(child, nn.Conv2d):
                    new_layer = LoRAConv2D(in_channels=child.in_channels,
                                           out_channels=child.out_channels,
                                           kernel_size=child.kernel_size,
                                           stride=child.stride,
                                           padding=child.padding,
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=config.lora_alpha,
                                           lora_dropout=config.lora_dropout,
                                           use_rslora=config.use_rslora)
                    print(f"State dict keys: {list(child.state_dict().keys())}")
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):
                    new_layer = LoRAEmbedding(num_embeddings=child.num_embeddings,
                                           embedding_dim=child.embedding_dim,
                                           rank=self.config.rank,
                                           lora_alpha=self.config.lora_alpha,
                                           lora_dropout=self.config.lora_dropout,
                                           use_rslora=self.config.use_rslora)
                    print(f"State dict keys: {list(child.state_dict().keys())}")
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)
                
                elif isinstance(child, nn.Conv1d):
                    new_layer = LoRAConv1d(in_channels=child.in_channels,
                                           out_channels=child.out_channels,
                                           kernel_size=child.kernel_size,
                                           stride=child.stride,
                                           padding=child.padding,
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=config.lora_alpha,
                                           lora_dropout=config.lora_dropout,
                                           use_rslora=config.use_rslora)
                    print(f"State dict keys: {list(child.state_dict().keys())}")
                    # parametrize.remove_parametrizations(new_layer, "weight", leave_parametrized=True)
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            if ((len(list(child.children()))) > 0) and not self._exclude_module_name_check(name):
                self._apply_lora(child)
    
    def _disable_all_grads(self):
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _togle_bias_grad(self):
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and self._target_module_name_check(name):
                        param.requires_grad = True

    def merge_weights(self, module):
        for name, child in module.named_children():
            if isinstance(child, (LoRALinear, LoRAEmbedding, LoRAConv2D, LoRAConv1d)):
                merged_layer = child.merge_weights()
                setattr(module, name, merged_layer)

            if len(list(child.children())) > 0:
                self.merge_weights(child)

    def save_model(self, path, merge_weights=False):
        def _detach_cpu(param):
            return param.detach().cpu()
        
        if merge_weights:
            self.merge_weights(self.lora_model)
            state_dict = {name.replace("lora_model.", ""):_detach_cpu(param) for (name, param) in self.named_parameters()}
        else:
            state_dict = {name:_detach_cpu(param) for (name, param) in self.named_parameters() if param.requires_grad}

        save_file(state_dict, path)

    def forward(self, *inputs, **kwargs):
        return self.lora_model(*inputs, **kwargs)

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification
    from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC

    target_modules = ["linear_q", "linear_k", "linear_v", "linear_out", "conv","word_embeddings"]
    exclude_modules = ["classifier"]

    config = LoraConfig(target_modules=target_modules, exclude_modules=exclude_modules, bias="lora_only")
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
    # model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
    print(model)

    lora_model = LoRAModel(model, config)
    # for name, param in lora_model.named_parameters():
    #     print(name, param.requires_grad)
    print(lora_model)
    lora_model.save_model("path", merge_weights=True)


