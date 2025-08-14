import math

import torch
from torch import nn

from main.arquitetura.stableVC.extracao_caracteristicas.SpeakerNet import SpeakerNet
import torchaudio


class TimbreExtractor(nn.Module):
    """
        Também pode ser dito como o MEL EXTRATOR
    """
    def __init__(self, n_mels=80, emb_dim=128, timbre_dim=128, device='cpu'):
        super(TimbreExtractor, self).__init__()
        self.device = device

        # Extrator de embedding global do locutor
        self.speaker_embedding = SpeakerNet(nOut=emb_dim, n_mels=n_mels).to(device)

        # Projeções para atenção cruzada
        # query_proj: emb_dim -> key_dim (n_mels + emb_dim)
        self.query_proj = nn.Linear(emb_dim, n_mels + emb_dim).to(device)
        # out_proj: n_mels -> timbre_dim
        self.out_proj = nn.Linear(n_mels, timbre_dim).to(device)

    def forward(self, mel_db: torch.Tensor):
        """
        Args:
            mel_db (Tensor): [B, n_mels, T]
        Returns:
            timbre_emb (Tensor): [B, timbre_dim]
            attn_weights (Tensor): [B, T]
        """
        B, n_mels, T = mel_db.shape

        # 1) Valores (V) da atenção: mel-spectrogram permutado
        V = mel_db.permute(0, 2, 1).contiguous()  # -> [B, T, n_mels]

        # 2) Embedding global do locutor: [B, emb_dim]
        spk_emb = self.speaker_embedding(mel_db)

        # 3) Query (Q): projeta spk_emb para dim de key
        Q = self.query_proj(spk_emb)  # [B, n_mels + emb_dim]

        # 4) Expansão temporal do embedding global: [B, T, emb_dim]
        spk_exp = spk_emb.unsqueeze(1).expand(-1, T, -1)

        # 5) Chaves (K): concatena V e spk_exp -> [B, T, n_mels + emb_dim]
        K = torch.cat([V, spk_exp], dim=-1)

        # 6) Cálculo de atenção scaled dot-product
        # scores: [B, T]
        scores = torch.sum(K * Q.unsqueeze(1), dim=-1) / math.sqrt(K.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]

        # 7) Agregação ponderada dos valores: [B, n_mels]
        timbre_feats = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)

        # 8) Projeção final para embedding de timbre: [B, timbre_dim]
        timbre_emb = self.out_proj(timbre_feats)

        return timbre_emb, attn_weights


def extracao_mel_spectrograma(audio_wav: torch.Tensor, d_model=128, device="cuda") -> (torch.Tensor, torch.Tensor):
    # batch_size = 2

    audio = audio_wav.unsqueeze(0)

    # Inicializa o extrator de timbre
    extractor = TimbreExtractor(emb_dim=d_model, device=device)
    mel_spec, _ = extractor(audio)

    # print("Forma do mel-espectrograma:", timbre_db.shape)         # Ex: (2, 80, frames)
    # print("Forma do embedding global:", global_embedding.shape)  # Ex: (2, 128)

    return mel_spec


def extracao_global_speaker(mel_spec: torch.Tensor, d_model=128, device="cuda") -> torch.Tensor:
    print("Entrada", mel_spec.shape)
    # Inicializa o extrator de timbre
    extractor = TimbreExtractor(emb_dim=d_model, device=device)
    _, global_embedding = extractor(mel_spec)
    print(global_embedding.shape)

    print(global_embedding.max(), global_embedding.min())

    return global_embedding
