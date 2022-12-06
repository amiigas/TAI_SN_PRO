from dataclasses import dataclass

import torch
from torchmetrics import SpearmanCorrCoef


@dataclass
class Config:
    seed: int = 7777
    device: str = "cuda"
    batch_size: int = 32
    hidden_dim: int = 12
    n_layers: int = 1
    loss_func: torch.nn.Module = torch.nn.MSELoss()
    learning_rate: float = 0.01
    momentum: float = 0.9
    epochs: int = 100
    metric: torch.nn.Module = SpearmanCorrCoef()
