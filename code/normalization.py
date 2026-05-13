import torch
import torch.nn as nn

class Normalization(nn.Module): # Based off the RevIN layer
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        return x_norm, mean, std

    def denorm(self, x, mean, std):
        return x * std + mean