import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.module import Module

class Projection(Module):
    def __init__(self, input_dim, h_dim, output_dim, dropout_rate=0.0):
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                                       nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                                       nn.Linear(h_dim, output_dim), nn.ReLU())

    def forward(self, data): 
        data = self.regressor(data)
        return data
