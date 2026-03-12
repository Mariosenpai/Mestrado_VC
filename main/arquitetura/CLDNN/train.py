# cldnn_flowmatching.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from tqdm.auto import tqdm

from main.arquitetura.CLDNN.model import CLDNNEncoder, CondUNet, sample_linear_path

import torch
import torch.nn.functional as F


# ---------------------------
#  Training & sampling loops
# ---------------------------
def train_epoch(dataloader, cldnn: CLDNNEncoder, vel_model: CondUNet,
                optim, device, log_every=100):
    cldnn.train()
    vel_model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Treinamento", total=len(dataloader))

    for i, batch in enumerate(pbar):

        # mel_in, mel_target: (B, T, F)
        audio, mel, mel_noise, sr, sentence = batch

        pred, groud_truth = generate_mel(cldnn, mel, mel_noise, vel_model,device)

        loss = torch.nn.functional.mse_loss(pred, groud_truth)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/(i+1))

    return total_loss / (i + 1)


def generate_mel(cldnn, mel_noise, mel, vel_model: CondUNet, device):
    '''
    Gera o mel espectograma a partir do mel_noise

    a função retorna o mel gerado (pred) e o mel esperado (groud_truth)
    :param mel_noise:
    :param mel:
    :param vel_model:
    :return:
    '''
    # mel_in, mel_target: (B, T, F)

    mel_in = mel_noise.to(device)
    mel_target = mel.to(device)

    # get cond embedding from CLDNN (per-frame or global)
    cond = cldnn(mel_in)  # (B, T_cond, cond_dim)
    # for simplicity make cond per-utterance via mean (FiLM handles this)
    # but cond can be per-frame if you upsample/align it
    # convert mel_target to image shape (B, C=1, F, T)
    x1 = mel_target.permute(0, 2, 1).unsqueeze(1).contiguous()
    x0 = torch.randn_like(x1)  # noise source

    B = x1.shape[0]
    t = torch.rand(B, device=device)  # random times in [0,1]

    x_t, groud_truth = sample_linear_path(x0, x1, t)

    pred = vel_model(x_t, t, cond)  # (B, C, F, T)

    target_T = groud_truth.size(-1)  # 259
    pred = pred[..., :target_T]

    return pred, groud_truth


@torch.no_grad()
def generate(mel_cond_input: torch.Tensor, cldnn: CLDNNEncoder, vel_model: CondUNet,
             device, num_steps=50):
    """
    Given mel_cond_input (B, T, F) produce mel_generated (B, T, F)
    Steps:
      - cond = cldnn(mel_cond_input)
      - sample x0 ~ N(0,I)
      - integrate dx/dt = u_theta(t,x,cond) from t=0 to 1
    """
    cldnn.eval()
    vel_model.eval()
    cond = cldnn(mel_cond_input.to(device))
    # prepare shapes
    B, T, F = mel_cond_input.shape
    x = torch.randn(B, 1, F, T, device=device)  # initial noise
    dt = 1.0 / num_steps
    t = torch.zeros(B, device=device)

    for step in range(num_steps):
        # simple Euler (you can replace by RK4 for better quality)
        t_mid = t + dt * 0.5
        u = vel_model(x, t, cond)  # shape (B,1,F,T)
        x = x + dt * u
        t = t + dt
        t = t.clamp(0.0, 1.0)
    # x is predicted x1 (mel image)
    mel_gen = x.squeeze(1).permute(0, 2, 1).contiguous()  # (B, T, F)
    return mel_gen


def cldnn_main():
    cldnn = CLDNNEncoder(n_mels=F, conv_channels=(32, 64), gru_hidden=128, gru_layers=2, proj_dim=256).to(device)
    vel = CondUNet(in_ch=1, base_ch=48, time_emb_dim=128, cond_dim=256).to(device)

    optim = torch.optim.Adam(list(cldnn.parameters()) + list(vel.parameters()), lr=2e-4)


# ---------------------------
#  Example minimal usage (pseudocode)
# ---------------------------
if __name__ == "__main__":
    # small smoke test shapes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    B = 2
    T = 260
    F = 80
    cldnn = CLDNNEncoder(n_mels=F, conv_channels=(32, 64), gru_hidden=128, gru_layers=2, proj_dim=256).to(device)
    vel = CondUNet(in_ch=1, base_ch=48, time_emb_dim=128, cond_dim=256).to(device)
    optim = torch.optim.Adam(list(cldnn.parameters()) + list(vel.parameters()), lr=2e-4)


    # fake dataloader
    class FakeDS:
        def __iter__(self):
            for _ in range(20):
                mel_in = torch.randn(B, T, F).to(device)
                mel_t = torch.randn(B, T, F).to(device)
                yield torch.tensor(mel_in), -1, torch.tensor(mel_t), -1

        def __len__(self):
            return 20


    dl = FakeDS()

    for ep in range(2):
        loss = train_epoch(dl, cldnn, vel, optim, device)
        print(f"epoch {ep} avg loss {loss:.6f}")

    # generate example
    mel_cond = torch.randn(B, T, F)
    mel_out = generate(mel_cond, cldnn, vel, device, num_steps=40)
    print("Generated mel shape:", mel_out.shape)  # (B, T, F)
