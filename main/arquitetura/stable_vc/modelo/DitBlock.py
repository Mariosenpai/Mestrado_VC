import torch.nn as nn

from bibliotecas_externas.Amphion.models import FiLM
from main.arquitetura.stableVC.modelo.DualAGC import DualAGC


class DiTBlock(nn.Module):
    def __init__(self, d_model, d_timbre, d_style,d_global_speak, n_heads, d_ff, d_norm=-1, dropout=0.1, device='cpu'):
        super(DiTBlock, self).__init__()
        self.d_model = d_model
        if d_norm == -1:
            d_norm = d_model

        self.d_norm = d_norm

        self.proj = nn.Linear(d_norm, d_model).to(device)

        # FiLM para condicionamento temporal: gera escala (gamma) e deslocamento (beta)
        # Estou usando a função do Amphion pois é a mesma usadada no artigo
        self.film = FiLM(d_model, d_model).to(device)

        # Normalização e autoatenção
        self.norm1 = nn.RMSNorm(d_norm).to(device)

        # ou nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout).to(device)

        # DualAGC para integrar timbre e estilo
        self.norm2 = nn.RMSNorm(d_norm).to(device)

        self.dual_agc = DualAGC(d_model, d_timbre, d_style, d_global_speak,device)

        # Rede Feed-Forward (FFN)
        self.norm3 = nn.RMSNorm(d_norm).to(device)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        ).to(device)

    def forward(self, content, timbre, style, global_speaker, time_embedding):
        """

        """

        # 1. FiLM: Condicionamento temporal
        # Gera parâmetros gamma e beta a partir do time_embedding e os aplica a x.
        x = self.film(content, time_embedding)

        # 2. Camada de autozatenção
        #x = self.proj(x)
        x_norm = self.norm1(x)
        # Note que nn.MultiheadAttention espera (query, key, value)
        attn_output, _ = self.self_attn(x_norm.transpose(0, 1),
                                        x_norm.transpose(0, 1),
                                        x_norm.transpose(0, 1))
        x = x + attn_output.transpose(0, 1)

        # 3. DualAGC: Incorpora informações de timbre e estilo
        x = x + self.dual_agc(x, timbre, style, global_speaker)

        # 4. Rede Feed-Forward
        x_norm3 = self.norm3(x)
        x = x + self.ffn(x_norm3)

        return x
