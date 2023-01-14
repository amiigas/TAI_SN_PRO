from dataclasses import dataclass

import torch
from torchmetrics import SpearmanCorrCoef

from src.models import BaseProteinModel, ProteinMLP


@dataclass
class Config:
    data_dir: str = "data/int_mapped_zero_padded"
    model: BaseProteinModel = ProteinMLP
    input_size: int = 8798
    seed: int = 7777
    device: str = "cuda"
    batch_size: int = 64
    dropout: float = 0.0
    activation: str = "sigmoid"
    loss_func: torch.nn.Module = torch.nn.MSELoss()
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 50
    metric: torch.nn.Module = SpearmanCorrCoef()
