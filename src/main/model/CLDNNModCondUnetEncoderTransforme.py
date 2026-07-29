from torch import nn

from src.main.model.CLDNN import CondUNet


class CondUnetEncoderTransforme(CondUNet):

    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128, cond_dim=256,d_model=512):
        super().__init__(in_ch=in_ch, base_ch=base_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=base_ch, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self,  x, t, cond):
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