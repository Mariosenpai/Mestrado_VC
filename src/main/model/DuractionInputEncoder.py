import torch
import torch.nn as nn
import math


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Encoder para gerar dp_input
# -----------------------------
class DPInputEncoder(nn.Module):
    def __init__(
        self,
        n_mels=80,
        d_model=80,
        n_heads=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        # Projeção inicial mel → d_model
        self.input_projection = nn.Linear(n_mels, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, mel):
        """
        mel: (B, T, n_mels) ou (T, n_mels)
        retorna: dp_input (B, T, d_model)
        """

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        x = self.input_projection(mel)
        x = self.pos_encoding(x)
        dp_input = self.encoder(x)

        return dp_input