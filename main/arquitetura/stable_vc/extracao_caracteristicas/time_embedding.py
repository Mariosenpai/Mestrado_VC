import math

import torch
from torch import nn


def extracao_time_embedding(d_model: int, num_steps_difusion: int, batch_size: int, device: str = "cuda") -> torch.Tensor:
    '''

    :param device:
    :param d_model: saida do modelo
    :param num_steps_difusion:
    :param batch_size:
    :return: tensor [batch_size, d_model] exemplo [1, 128]
    '''

    # Um processo de difusao
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, embed_dim: int):
            super(SinusoidalTimeEmbedding, self).__init__()
            self.embed_dim = embed_dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            """
            Recebe um tensor t de forma (batch,) com os time steps (como float)
            e retorna o embedding sinusoidal de forma (batch, embed_dim).
            """
            half_dim = self.embed_dim // 2
            # Calcula o termo de divisão (log(10000) / (half_dim - 1))
            div_term = math.log(10000) / (half_dim - 1)
            # Cria um vetor com frequências decaindo exponencialmente
            exp_term = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -div_term)
            # Multiplica o time steps (expandindo para (batch, 1)) pelas frequências
            pos = t.unsqueeze(1) * exp_term.unsqueeze(0)  # (batch, half_dim)
            # Concatena seno e cosseno para formar o embedding
            emb = torch.cat([torch.sin(pos), torch.cos(pos)], dim=1)  # (batch, embed_dim)
            return emb

    # Simula alguns time steps para cada exemplo do batch
    # Por exemplo, podem ser valores contínuos representando o passo atual do processo
    t = torch.randint(0, num_steps_difusion, (batch_size,), dtype=torch.float32)  # time steps aleatórios

    # Gera o time embedding
    time_embedding_layer = SinusoidalTimeEmbedding(d_model)
    time_emb = time_embedding_layer(t).to(device)

    return time_emb