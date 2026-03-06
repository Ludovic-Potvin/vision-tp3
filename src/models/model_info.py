from torch import nn
from dataclasses import dataclass
from typing import Any

@dataclass
class ModelInfo:
    weights: Any
    model: Any
    mean: Any
    std: Any


