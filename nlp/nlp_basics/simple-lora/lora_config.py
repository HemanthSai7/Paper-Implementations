from dataclasses import dataclass
from typing import Optional, Union, List, Literal

@dataclass
class LoraConfig:
    rank: int = 8
    target_modules: Optional[Union[List[str], str]] = None
    exclude_modules: Optional[Union[List[str], str]] = None
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
