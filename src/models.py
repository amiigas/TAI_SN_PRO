import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ProteinMLP(nn.Module):
    def __init__(self, input_size=8798, dropout=0.0, activation="sigmoid", device="cpu") -> None:
        super(ProteinMLP, self).__init__()
        self._tb_writer = SummaryWriter(log_dir=f"runs/{self.__class__.__name__}")
        self.device = device
        
        if activation == "sigmoid":
            self._activation_func = nn.Sigmoid()
        elif activation == "relu":
            self._activation_func = nn.ReLU()

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.Dropout(dropout),
            self._activation_func,
            nn.Linear(1000, 512),
            nn.Dropout(dropout),
            self._activation_func,
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            self._activation_func,
            nn.Linear(128, 1)
        )

        self.to(self.device)

    def forward(self, x):
        return self.linear_sigmoid_stack(x)

    def fit(self, train_dataloader, valid_dataloader, loss_func, metric, optimizer, epochs):
        for e in range(epochs):
            # train
            self.train()
            total_train_loss = 0.0
            for X, y in tqdm(train_dataloader, desc=f"Epoch {e}"):
                pred = self(X)
                loss = loss_func(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            total_train_loss /= len(train_dataloader)
            self._tb_writer.add_scalar("loss/train", total_train_loss, e)
            
            # valid
            self.eval()
            total_valid_loss = 0.0
            valid_metric = metric
            with torch.no_grad():
                for X, y in tqdm(valid_dataloader, desc=f"Validating"):
                    pred = self(X)

                    loss = loss_func(pred, y)
                    valid_metric.update(pred.flatten(), y.flatten())
                    total_valid_loss += loss.item()

            total_valid_loss /= len(valid_dataloader)
            total_metric = valid_metric.compute()
            self._tb_writer.add_scalar("loss/valid", total_valid_loss, e)
            self._tb_writer.add_scalar("metric/valid", total_metric, e)

            valid_metric.reset()

        self._tb_writer.flush()



class ProteinRNN(nn.Module):
    def __init__(self, input_size=1, hidden_dim=12, n_layers=1, device="cpu") -> None:
        super(ProteinRNN, self).__init__()
        self._tb_writer = SummaryWriter(log_dir=f"runs/{self.__class__.__name__}")
        self.device = device

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        self.fc = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def forward(self, x):
        hidden = torch.zeros(self.n_layers, x.shape[0], self.hidden_dim).to(self.device)
        x = torch.unsqueeze(x, 2)
        out, _ = self.rnn(x, hidden)
        out = out[:,-1, :]
        out = self.fc(out)
        
        return out

    def fit(self, train_dataloader, valid_dataloader, loss_func, metric, optimizer, epochs):
        for e in range(epochs):
            # train
            self.train()
            total_train_loss = 0.0
            for X, y in tqdm(train_dataloader, desc=f"Epoch {e}"):
                pred = self(X)
                loss = loss_func(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            total_train_loss /= len(train_dataloader)
            self._tb_writer.add_scalar("loss/train", total_train_loss, e)
            
            # valid
            self.eval()
            total_valid_loss = 0.0
            valid_metric = metric
            with torch.no_grad():
                for X, y in tqdm(valid_dataloader, desc=f"Validating"):
                    pred = self(X)

                    loss = loss_func(pred, y)
                    valid_metric.update(pred.flatten(), y.flatten())
                    total_valid_loss += loss.item()

            total_valid_loss /= len(valid_dataloader)
            total_metric = valid_metric.compute()
            self._tb_writer.add_scalar("loss/valid", total_valid_loss, e)
            self._tb_writer.add_scalar("metric/valid", total_metric, e)

            valid_metric.reset()

        self._tb_writer.flush()
