import os
import random
import sys
import importlib

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets import IntMappedZeroPaddedDataset
from src.models import ProteinMLP


def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_config(config_name):
    module = importlib.import_module(f"configs.{config_name}")
    return module.Config


if __name__ == "__main__":
    cfg = parse_config(sys.argv[1])
    
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {cfg.device} device")

    seed_everything(cfg.seed)

    X = np.load("./data/int_mapped_zero_padded/inputs.npy")
    Y = np.load("./data/int_mapped_zero_padded/targets.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    train_dataset = IntMappedZeroPaddedDataset(X_train, y_train)
    test_dataset = IntMappedZeroPaddedDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = ProteinMLP(dropout=cfg.dropout, activation=cfg.activation)
    model.to(cfg.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    model.fit(
        train_dataloader=train_dataloader,
        valid_dataloader=test_dataloader,
        loss_func=cfg.loss_func,
        metric=cfg.metric,
        optimizer=optimizer,
        epochs=cfg.epochs
    )