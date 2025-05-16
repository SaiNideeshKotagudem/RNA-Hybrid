# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BPPGCN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, edge_index, edge_attr, num_nodes):
        x = torch.ones((num_nodes, 1), device=edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

class SqueezeformerBlock(nn.Module):
    def __init__(self, input_dim, squeeze_factor=4):
        super().__init__()
        self.downsample = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=squeeze_factor)
        self.upsample = nn.ConvTranspose1d(input_dim, input_dim, kernel_size=1, stride=squeeze_factor)
        self.transformer = TransformerBlock(input_dim)

    def forward(self, x):
        x_ = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        x_ = self.transformer(x_)
        x_ = self.upsample(x_.transpose(1, 2)).transpose(1, 2)
        return x_

class FusionGate(nn.Module):
    def __init__(self, input_dim, num_streams=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * num_streams, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_streams),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs):
        concat = torch.cat(inputs, dim=-1)
        weights = self.gate(concat)
        fused = sum(w.unsqueeze(-1) * s for w, s in zip(weights.chunk(len(inputs), dim=-1), inputs))
        return fused

class RNAHybridModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=3, max_len=200):
        super().__init__()
        self.pos_encoder = DynamicPositionalEncoding(input_dim, max_len=max_len)

        self.gru = GRUBlock(input_dim, hidden_dim)
        self.transformer = TransformerBlock(input_dim)
        self.squeezeformer = SqueezeformerBlock(input_dim)

        self.fusion = FusionGate(input_dim * 2, num_streams=3)

        self.bpp_gcn = BPPGCN()
        self.bpp_attention_proj = nn.Linear(hidden_dim, input_dim * 2)

        self.output = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, output_dim)  # Predict x, y, z or 40x3 + 2
        )

    def forward(self, x, bpp_edge_index, bpp_edge_attr, bpp_num_nodes):
        x = self.pos_encoder(x)

        gru_out = self.gru(x)
        transformer_out = self.transformer(x)
        squeeze_out = self.squeezeformer(x)

        fused = self.fusion([gru_out, transformer_out, squeeze_out])

        bpp_feat = self.bpp_gcn(bpp_edge_index, bpp_edge_attr, bpp_num_nodes)
        bpp_proj = self.bpp_attention_proj(bpp_feat).unsqueeze(0).expand(x.size(0), -1, -1)

        fused = fused + bpp_proj
        return self.output(fused)

class MaskedSNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        mse = (pred - target) ** 2
        snr = (target ** 2) / (mse + self.eps)
        masked_snr = snr * mask
        loss = -torch.log(masked_snr + self.eps)
        return loss.sum() / mask.sum()
