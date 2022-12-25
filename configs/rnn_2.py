from dataclasses import dataclass

import torch
from torchmetrics import SpearmanCorrCoef

from src.models import BaseProteinModel, ProteinRNN


@dataclass
class Config:
    data_dir: str = "data/one_hot_zero_padded"
    model: BaseProteinModel = ProteinRNN
    input_size: int = 20
    seed: int = 7777
    device: str = "cuda"
    batch_size: int = 64
    hidden_dim: int = 24
    n_layers: int = 1
    loss_func: torch.nn.Module = torch.nn.MSELoss()
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 100
    metric: torch.nn.Module = SpearmanCorrCoef()
