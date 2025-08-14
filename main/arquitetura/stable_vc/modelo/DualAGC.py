import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualAGC(nn.Module):
    def __init__(self, d_model, d_timbre, d_style, d_global_speak, device="cpu"):
        super(DualAGC, self).__init__()
        self.d_style = d_style
        self.d_timbre = d_timbre
        self.d_model = d_model

        # Projeção do conteúdo para gerar a consulta (query)
        self.norm = nn.RMSNorm(d_model).to(device)
        self.query_proj = nn.Linear(d_model, d_model).to(device)

        # Projeções para a branch de timbre (key e value)
        self.timbre_key_proj = nn.Linear(d_timbre, d_model).to(device)
        self.timbre_value_proj = nn.Linear(d_timbre, d_model).to(device)
        self.timbre_proj = nn.Linear(d_model*2, d_model).to(device)

        # Projeções para a branch de estilo (key e value)
        self.style_key_proj = nn.Linear(d_style, d_model).to(device)
        self.style_value_proj = nn.Linear(d_style, d_model).to(device)

        # Projeção para o embedding global do locutor (para estabilidade do timbre)
        self.global_speaker_proj = nn.Linear(d_global_speak, d_model).to(device)

        # Parâmetro de gate adaptativo, inicializado em 0
        self.alpha = nn.Parameter(torch.zeros(1)).to(device)

    def forward(self, content, timbre, style, global_speaker):
        """

        """
        # 1. Projeção do conteúdo para obter Query
        x_norm = self.norm(content)
        Q = self.query_proj(x_norm)

        # 2. Projeção do timbre para Key e Value
        K_timbre = self.timbre_key_proj(timbre)
        V_timbre = self.timbre_value_proj(timbre)

        # 3. Projeção do estilo para Key e Value
        K_style = self.style_key_proj(style)
        V_style = self.style_value_proj(style)

        # 4. Processamento do embedding global do locutor
        global_speaker = self.global_speaker_proj(global_speaker)  # (batch, d_model)

        # Concatena o embedding global às chaves e valores do timbre
        K_timbre = torch.cat([global_speaker, K_timbre], dim=1)  # (batch, seq_len+1, d_model)
        V_timbre = torch.cat([global_speaker, V_timbre], dim=1)  # (batch, seq_len+1, d_model)
        # normalização dos valores para d_model depois da concatenação
        K_timbre = self.timbre_proj(K_timbre)
        V_timbre = self.timbre_proj(V_timbre)

        # 5. Atenção para Timbre
        d_k = self.d_model
        attn_scores_timbre = torch.matmul(Q, K_timbre.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights_timbre = F.softmax(attn_scores_timbre, dim=-1)
        timbre_attn = torch.matmul(attn_weights_timbre, V_timbre)

        # 6. Atenção para Estilo
        attn_scores_style = torch.matmul(Q, K_style.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights_style = F.softmax(attn_scores_style, dim=-1)
        style_attn = torch.matmul(attn_weights_style, V_style)

        # 7. Combinação via gate adaptativo
        output = timbre_attn + torch.tanh(self.alpha) * style_attn

        return output
