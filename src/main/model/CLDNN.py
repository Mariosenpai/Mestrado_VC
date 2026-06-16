import math
from typing import Optional, Tuple

import torch
from torch import nn

# cldnn_flowmatching.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

class CLDNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldnn = CLDNNEncoder(
            n_mels = 80,
            conv_channels = (32, 64),
            gru_hidden = 256,
            gru_layers = 3,
            proj_dim = 256,
            bidirectional = True,
            causal = False,
            conv_pool =None
        )
        self.condUnet = CondUNet(in_ch=1, base_ch=64, time_emb_dim=128, cond_dim=256)


class ConvBlock(nn.Module):
    """
    Bloco Conv2D simples: Conv2d -> BatchNorm2d -> ReLU -> (Optional MaxPool2d)
    Input shape: (B, 1, F, T)  where F = freq bins (ex: 80), T = frames (time)
    """

    def __init__(self, in_ch, out_ch, kernel=(5, 5), stride=(1, 1), padding=(2, 2),
                 use_pool: bool = False, pool_kernel=(1, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.use_pool = use_pool
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel) if use_pool else None

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_pool:
            x = self.pool(x)
        return x


class CLDNNEncoder(nn.Module):
    """
    CLDNN encoder (ConvStack -> Bi-GRU stack -> FC projection).
    Returns per-frame embeddings with shape (B, T_out, proj_dim).

    Args:
        n_mels: number of mel frequency bins (F), typically 80.
        conv_channels: list of channel sizes for conv blocks, e.g. [32, 64]
        gru_hidden: hidden size per direction for Bi-GRU (default 256)
        gru_layers: number of GRU layers (default 3)
        proj_dim: final projection size (default 256)
        bidirectional: whether to use Bi-GRU (default True)
        causal: if True use uni-directional GRU (for low-latency scenarios)
    """

    def __init__(self,
                 n_mels: int = 80,
                 conv_channels=(32, 64),
                 gru_hidden: int = 256,
                 gru_layers: int = 3,
                 proj_dim: int = 256,
                 bidirectional: bool = True,
                 causal: bool = False,
                 conv_pool:
                 Optional[list] = None):
        super().__init__()

        assert len(conv_channels) >= 1, "conv_channels must contain at least one element"
        self.n_mels = n_mels
        self.conv_channels = conv_channels
        self.proj_dim = proj_dim
        self.bidirectional = bidirectional and (not causal)
        self.rnn_directions = 2 if self.bidirectional else 1

        # Build conv stack: first block with 5x5, second with 3x3 (common pattern)
        # conv_pool: list of booleans whether to apply pooling after each block
        if conv_pool is None:
            conv_pool = [False] * len(conv_channels)

        conv_blocks = []
        in_ch = 1  # input channel for mel image
        for i, out_ch in enumerate(conv_channels):
            kernel = (5, 5) if i == 0 else (3, 3)
            padding = (kernel[0] // 2, kernel[1] // 2)
            use_pool = conv_pool[i] if i < len(conv_pool) else False
            pool_kernel = (1, 2) if use_pool else (1, 1)
            conv_blocks.append(ConvBlock(in_ch, out_ch, kernel=kernel, padding=padding,
                                         use_pool=use_pool, pool_kernel=pool_kernel))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*conv_blocks)

        # after convs we will collapse frequency+channel into feature dim for RNN
        # but frequency dimension depends on pooling/stride; compute dynamically in forward.

        # Bi-GRU stack
        rnn_input_size = conv_channels[-1] * n_mels  # placeholder; we'll adapt in forward
        # create GRU but with input_size set later via a small wrapper if needed.
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        # We'll instantiate a GRU with a dummy input size now and replace in forward if mismatch
        # However easier: create an nn.GRU in __init__ with input_size = conv_channels[-1] * n_mels
        # and allow user to ensure pooling choices keep freq dimension stable.
        # To avoid hard assumptions, we will create RNN lazily in first forward pass.
        self._rnn = None

        # Projection MLP: two FC layers mapping RNN hidden dim -> proj_dim
        rnn_out_dim = self.rnn_directions * self.gru_hidden
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

        # store flags for lazy rnn creation
        self._gru_bidirectional = self.bidirectional
        self._causal = causal
        self._built = False

    def _build_rnn(self, feat_dim):
        """Build the GRU module when feat_dim (input dim to RNN) is known."""
        self._rnn = nn.GRU(input_size=feat_dim,
                           hidden_size=self.gru_hidden,
                           num_layers=self.gru_layers,
                           batch_first=True,
                           bidirectional=self._gru_bidirectional).to("cuda")
        self._built = True

    def forward(self, mel: torch.Tensor):
        """
        mel: tensor (B, T, F) OR (B, 1, F, T). We accept (B, T, F) as common.
        Returns: embeddings (B, T_out, proj_dim)
        """
        # Normalize input shape to (B, 1, F, T)
        if mel.dim() == 3:
            # (B, T, F) -> (B, 1, F, T)
            mel = mel.permute(0, 2, 1).unsqueeze(1)
        elif mel.dim() == 4:
            # assume (B, C=1, F, T)
            pass
        else:
            raise ValueError("mel must have shape (B, T, F) or (B, C, F, T)")

        # conv stack -> (B, C_out, F_out, T_out)
        x = self.conv_stack(mel)

        B, Cc, Fp, Tp = x.shape

        # Prepare RNN input: collapse channel+freq -> feat, keep time Tp
        # We want shape (B, Tp, feat_dim)
        feat = x.permute(0, 3, 1, 2).contiguous()  # (B, T_out, C, F)
        feat = feat.view(B, Tp, Cc * Fp)  # (B, T_out, feat_dim)
        feat_dim = Cc * Fp

        # lazy build RNN if needed
        if not self._built:
            self._build_rnn(feat_dim)

        # RNN forward
        # If causal==True and model was intended for streaming, we already set uni-directional.
        rnn_out, _ = self._rnn(feat)  # (B, T_out, rnn_out_dim)
        # project per-frame to proj_dim
        emb = self.proj(rnn_out)  # (B, T_out, proj_dim)

        return emb


# ---------------------------
#  Time embedding helper
# ---------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor):
        """
        t: (...,) values in [0,1]
        returns: (..., dim)
        """
        # sinusoidal positional embeddings
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.lin(emb)


# ---------------------------
#  FiLM conditioning bridge
# ---------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)  # scale and shift

    def forward(self, cond):
        # cond: (B, T_cond, cond_dim) or (B, cond_dim)
        # returns scale, shift shapes to be broadcasted to (B, C, H, W)
        if cond.dim() == 3:
            # aggregate over time (simple mean) -> per-utterance cond
            cond = cond.mean(dim=1)
        params = self.fc(cond)  # (B, channels*2)
        scale, shift = params.chunk(2, dim=-1)
        return scale.unsqueeze(-1).unsqueeze(-1), shift.unsqueeze(-1).unsqueeze(-1)


# ---------------------------
#  Conditional U-Net velocity model
# ---------------------------
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            conv3x3(c, c),
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True),
            conv3x3(c, c),
            nn.GroupNorm(8, c)
        )

    def forward(self, x):
        return x + self.net(x)


class CondUNet(nn.Module):
    """
    Simple U-Net that predicts velocity u_theta(t, x, cond)
    Input x: (B, 1, F, T) or (B, C, F, T)
    cond: (B, T_cond, cond_dim) or (B, cond_dim)
    """

    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128, cond_dim=256):
        super().__init__()
        self.inc = conv3x3(in_ch, base_ch)
        self.down1 = nn.Sequential(conv3x3(base_ch, base_ch * 2, stride=(1, 2)), ResidualBlock(base_ch * 2))
        self.down2 = nn.Sequential(conv3x3(base_ch * 2, base_ch * 4, stride=(1, 2)), ResidualBlock(base_ch * 4))
        self.mid = ResidualBlock(base_ch * 4)

        # up blocks (use conv transpose or upsample+conv)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(1, 2), mode='nearest'),
                                 conv3x3(base_ch * 4, base_ch * 2),
                                 ResidualBlock(base_ch * 2))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(1, 2), mode='nearest'),
                                 conv3x3(base_ch * 2, base_ch),
                                 ResidualBlock(base_ch))

        self.outc = nn.Sequential(nn.ReLU(inplace=True), conv3x3(base_ch, in_ch))

        # conditioning
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_fc = nn.Linear(time_emb_dim, base_ch * 4)
        self.film1 = FiLM(cond_dim, base_ch * 4)  # applied at mid
        self.film2 = FiLM(cond_dim, base_ch * 2)  # applied at up2
        self.film3 = FiLM(cond_dim, base_ch)  # applied at up1

    def forward(self, x, t, cond):
        """
        x: (B, C_in, F, T)
        t: (B,) in [0,1]
        cond: (B, T_cond, cond_dim) or (B, cond_dim)
        returns: dx_dt (same shape as x)
        """
        # initial
        x1 = self.inc(x)  # (B, base, F, T)
        d1 = self.down1(x1)  # (B, base*2, F, T/2)
        d2 = self.down2(d1)  # (B, base*4, F, T/4)
        mid = self.mid(d2)

        # time conditioning injected into mid
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        t_fc = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, base*4, 1,1)
        mid = mid + t_fc

        # FiLM at mid
        s, sh = self.film1(cond)
        mid = mid * (1 + s) + sh

        u = self.up2(mid)  # (B, base*2, F, T/2)
        s2, sh2 = self.film2(cond)
        u = u * (1 + s2) + sh2

        u = self.up1(u)  # (B, base, F, T)
        s3, sh3 = self.film3(cond)
        u = u * (1 + s3) + sh3

        out = self.outc(u)  # (B, in_ch, F, T)
        return out


# ---------------------------
#  Flow Matching utilities
# ---------------------------
# def sample_linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Linear interpolation path:
#       x_t = (1-t) * x0 + t * x1
#     derivative wrt t:
#       dx_dt = x1 - x0    (constant, same shape)
#     Args:
#       x0, x1: (B, C, F, T)
#       t: (B,) in [0,1]
#     Returns:
#       x_t: (B, C, F, T)
#       dx_dt: (B, C, F, T)
#     """
#     # expand t to spatial shape
#     B = x0.shape[0]
#     shape = [B] + [1] * (x0.dim() - 1)
#     t_sh = t.view(shape)
#     x_t = (1.0 - t_sh) * x0 + t_sh * x1
#     dx_dt = x1 - x0
#     return x_t, dx_dt
#
#
#
#
# # ---------------------------
# #  Training & sampling loops
# # ---------------------------
# def train_epoch(dataloader, cldnn: CLDNNEncoder, vel_model: CondUNet,
#                 optim, device, log_every=100):
#     cldnn.train()
#     vel_model.train()
#     total_loss = 0.0
#
#     pbar = tqdm(dataloader, desc="Treinamento", total=len(dataloader))
#
#     for i, batch in enumerate(pbar):
#
#         # mel_in, mel_target: (B, T, F)
#         audio, mel, mel_noise, sr, sentence = batch
#
#         pred, groud_truth = generate_mel(cldnn, mel, mel_noise, vel_model,device)
#
#         loss = torch.nn.functional.mse_loss(pred, groud_truth)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#
#         total_loss += loss.item()
#         pbar.set_postfix(loss=total_loss/(i+1))
#
#     return total_loss / (i + 1)
#
#
# def generate_mel(cldnn, mel_noise, mel, vel_model: CondUNet, device):
#     '''
#     Gera o mel espectograma a partir do mel_noise
#
#     a função retorna o mel gerado (pred) e o mel esperado (groud_truth)
#     :param mel_noise:
#     :param mel:
#     :param vel_model:
#     :return:
#     '''
#     # mel_in, mel_target: (B, T, F)
#
#     mel_in = mel_noise.to(device)
#     mel_target = mel.to(device)
#
#     # get cond embedding from CLDNN (per-frame or global)
#     cond = cldnn(mel_in)  # (B, T_cond, cond_dim)
#     # for simplicity make cond per-utterance via mean (FiLM handles this)
#     # but cond can be per-frame if you upsample/align it
#     # convert mel_target to image shape (B, C=1, F, T)
#     x1 = mel_target.permute(0, 2, 1).unsqueeze(1).contiguous()
#     x0 = torch.randn_like(x1)  # noise source
#
#     B = x1.shape[0]
#     t = torch.rand(B, device=device)  # random times in [0,1]
#
#     x_t, groud_truth = sample_linear_path(x0, x1, t)
#
#     pred = vel_model(x_t, t, cond)  # (B, C, F, T)
#
#     target_T = groud_truth.size(-1)  # 259
#     pred = pred[..., :target_T]
#
#     return pred, groud_truth
#
#
# @torch.no_grad()
# def generate(mel_cond_input: torch.Tensor, cldnn: CLDNNEncoder, vel_model: CondUNet,
#              device, num_steps=50):
#     """
#     Given mel_cond_input (B, T, F) produce mel_generated (B, T, F)
#     Steps:
#       - cond = cldnn(mel_cond_input)
#       - sample x0 ~ N(0,I)
#       - integrate dx/dt = u_theta(t,x,cond) from t=0 to 1
#     """
#     cldnn.eval()
#     vel_model.eval()
#     cond = cldnn(mel_cond_input.to(device))
#     # prepare shapes
#     B, T, F = mel_cond_input.shape
#     x = torch.randn(B, 1, F, T, device=device)  # initial noise
#     dt = 1.0 / num_steps
#     t = torch.zeros(B, device=device)
#
#     for step in range(num_steps):
#         # simple Euler (you can replace by RK4 for better quality)
#         t_mid = t + dt * 0.5
#         u = vel_model(x, t, cond)  # shape (B,1,F,T)
#         x = x + dt * u
#         t = t + dt
#         t = t.clamp(0.0, 1.0)
#     # x is predicted x1 (mel image)
#     mel_gen = x.squeeze(1).permute(0, 2, 1).contiguous()  # (B, T, F)
#     return mel_gen
#
#
# def cldnn_main():
#     cldnn = CLDNNEncoder(n_mels=F, conv_channels=(32, 64), gru_hidden=128, gru_layers=2, proj_dim=256).to(device)
#     vel = CondUNet(in_ch=1, base_ch=48, time_emb_dim=128, cond_dim=256).to(device)
#
#     optim = torch.optim.Adam(list(cldnn.parameters()) + list(vel.parameters()), lr=2e-4)
#
#
# # ---------------------------
# #  Example minimal usage (pseudocode)
# # ---------------------------
# if __name__ == "__main__":
#     # small smoke test shapes
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     B = 2
#     T = 260
#     F = 80
#     cldnn = CLDNNEncoder(n_mels=F, conv_channels=(32, 64), gru_hidden=128, gru_layers=2, proj_dim=256).to(device)
#     vel = CondUNet(in_ch=1, base_ch=48, time_emb_dim=128, cond_dim=256).to(device)
#     optim = torch.optim.Adam(list(cldnn.parameters()) + list(vel.parameters()), lr=2e-4)
#
#
#     # fake dataloader
#     class FakeDS:
#         def __iter__(self):
#             for _ in range(20):
#                 mel_in = torch.randn(B, T, F).to(device)
#                 mel_t = torch.randn(B, T, F).to(device)
#                 yield torch.tensor(mel_in), -1, torch.tensor(mel_t), -1
#
#         def __len__(self):
#             return 20
#
#
#     dl = FakeDS()
#
#     for ep in range(2):
#         loss = train_epoch(dl, cldnn, vel, optim, device)
#         print(f"epoch {ep} avg loss {loss:.6f}")
#
#     # generate example
#     mel_cond = torch.randn(B, T, F)
#     mel_out = generate(mel_cond, cldnn, vel, device, num_steps=40)
#     print("Generated mel shape:", mel_out.shape)  # (B, T, F)