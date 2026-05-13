import torch
import torch.nn as nn
from .layersv2 import PatchEmbedding, PositionalEmbedding
from .normalization import Normalization

class PatchTSTTwo(nn.Module):
    def __init__(self, M, L, T, P=16, S=8, D=128, n_heads=16, n_layers=3, dropout=0.2):
        super().__init__()
        self.P, self.S = P, S

        self.normalization = Normalization(M)

        self.patch_embed = PatchEmbedding(P, S, D)
        self.pos_embed = PositionalEmbedding(D)

        # Official repo: patch_num = (L - P) // S + 1, then +1 for end padding
        self.num_patches = (L - P) // S + 1 + 1  # the +1 is for end padding

        # Pad end of sequence by S timesteps (matching official repo)
        self.padding_patch_layer = nn.ReplicationPad1d((0, S))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=D*2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # head_nf = D * num_patches, applied per-variable (on last 2 dims of [B, M, N, D])
        self.head_nf = D * self.num_patches
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.head_nf, T)

    def forward(self, x):
        B, M, L = x.shape  # [Batch, Vars, Seq_Len]

        x, mean, std = self.normalization(x)

        # 1. Reshape to [B*M, L] and pad end by S (official repo approach)
        x = x.reshape(B * M, L)
        x = self.padding_patch_layer(x)        # [B*M, L+S]

        # 2. Unfold into patches
        x = x.unfold(dimension=-1, size=self.P, step=self.S)  # [B*M, N, P]

        # 3. Embed patches
        x = self.patch_embed(x)   # [B*M, N, D]
        x = self.pos_embed(x)

        # 4. Transformer
        z = self.transformer_encoder(x)  # [B*M, N, D]

        # 5. Reshape to [B, M, N, D], flatten N and D, then project to T
        z = z.reshape(B, M, self.num_patches, -1)  # [B, M, N, D]
        z = self.flatten(z)                         # [B, M, N*D]
        z = self.linear(z)                       # [B, M, T]
        z = self.normalization.denorm(z, mean, std)
        return z