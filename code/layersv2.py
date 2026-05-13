import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, P, S, D):
        super().__init__()
        self.P = P  # Patch length [cite: 139]
        self.S = S  # Stride [cite: 139]
        # Trainable linear projection [cite: 144]
        self.value_embedding = nn.Linear(P, D)

    def forward(self, x):
        # x: [Batch, Channels, L]
        # Padding to ensure we can divide by stride [cite: 139]
        # x: [Batch, Channels, Number_of_Patches, Patch_Length]
        x = self.value_embedding(x)
        # x: [Batch, Channels, Number_of_Patches, D]
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, D, max_len=5000):
        super().__init__()
        # Learnable additive position encoding [cite: 144]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, D))

    def forward(self, x):
        # x: [Batch, Channels, N, D]
        B, N, D = x.shape
        return x + self.pos_embed[:, :N, :]
        #return x + self.pos_embed[:, :x.size(1), :]