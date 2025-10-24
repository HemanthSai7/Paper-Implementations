import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
from safetensors.torch import save_file
from typing import Optional, Literal, Union, List

__all__ = ["LoRALinear","LoRAEmbedding","LoRAConv2D"]


class LoRALayerBase:
    """
    Simple implementation of [Low Rank Adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf)

    The forward pass combines the frozen linear output with the LoRA adaptation:
    
        y = xW + x(AB) * (α / r)
    
    where:
        - W is the frozen pretrained weight matrix
        - A ∈ ℝ^(in_features x rank)
        - B ∈ ℝ^(rank x out_features)
        - α is the scaling factor
        - r is the rank of the low-rank matrices
    """
    def __init__(
        self,
        rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        use_rslora: bool = False, #
    ):
        """Base class for LoRA layers.

        Args:
            rank (int): Rank of the low-rank matrix.
            lora_alpha (int): Alpha value for scaling the low-rank matrix.
            lora_dropout (float): Dropout probability for the low-rank matrix.
            use_rslora (bool): [Use rank stabilised loRA](https://arxiv.org/pdf/2312.03732)
        """
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.use_rslora = use_rslora

        self.scaling = self.lora_alpha / self.rank ** 0.5 if use_rslora else self.lora_alpha / self.rank # alpha/rank

    

class LoRALinear(nn.Linear, LoRALayerBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        **kwargs,
    ):  
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )
        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank)) # Initialize A matrix with R^(in, r)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) # Initialize B matrix with R^(r, out)
        nn.init.kaiming_uniform_(self.lora_A, self.lora_B)

    def forward(self, x: torch. Tensor) -> torch.Tensor:
        orig_layer_out = F.linear(x, self.weight, self.bias)

        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult

        output = orig_layer_out + low_rank_out
        return output
    

class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        use_rslora: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(
            self,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            **kwargs
        )
        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(embedding_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, num_embeddings))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_layer_out = F.embedding(
            input=x,
            weight=self.weight,
            padding_idx=self.padding_idx,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        low_rank_A_output = F.embedding(
            input=x,
            weight=self.lora_A,
            padding_idx=self.padding_idx,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling
        output = orig_layer_out + low_rank_output

        return output


class LoRAConv2D(nn.Conv2d, LoRALayerBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=bias,
                           **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora)

        self.weight.requires_grad = False

        self.lora_A = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.lora_B = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        orig = F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        print(orig.shape)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling
        print(lora_out.shape)
        return orig + lora_out

    
# if __name__ == "__main__":
#     x = torch.randn(2, 3, 32, 32)  # (batch, channels, height, width)
#     lora_conv2d = LoRAConv2D(3, 256,3, rank=8)
    
#     output = lora_conv2d(x)  # This calls forward() and triggers the prints
#     print(f"Output shape: {output.shape}")